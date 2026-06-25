"""Rebuild the M3L internal-item to MovieLens movieId bridge.

The M3L interaction file uses internal item ids, while MovieLens metadata uses
movieId. In our local data layout, the MPNet feature matrix follows the M3L
internal item order and the per-movie MPNet JSON files are named by MovieLens
movieId. Matching a small rounded vector fingerprint lets us reconstruct this
bridge without guessing titles or relying on a hidden table.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm


FINGERPRINT_INDICES = [0, 1, 2, 3, 4, 5, 10, 50, 100, 200, 400, 767]


def resolve_existing_path(project_root: Path, explicit: str | None, candidates: list[str], label: str) -> Path:
    """Resolve a user-provided path or the first matching default candidate."""
    if explicit:
        path = Path(explicit)
        path = path if path.is_absolute() else project_root / path
        if path.exists():
            return path
        raise FileNotFoundError(f"{label} not found: {path}")

    for candidate in candidates:
        path = project_root / candidate
        if path.exists():
            return path

    tried = "\n".join(f"  - {project_root / candidate}" for candidate in candidates)
    raise FileNotFoundError(f"Could not find {label}. Tried:\n{tried}")


def vector_fingerprint(vector: np.ndarray, precision: int = 8) -> tuple[float, ...]:
    """Create a compact, stable fingerprint from selected embedding positions."""
    arr = np.asarray(vector, dtype=np.float64).reshape(-1)
    indices = [idx for idx in FINGERPRINT_INDICES if idx < arr.size]
    if not indices:
        raise ValueError("Cannot fingerprint an empty vector.")
    return tuple(np.round(arr[indices], precision))


def read_json_vector(path: Path) -> np.ndarray:
    """Load the vector from an M3L JSON feature file.

    The JSON files usually contain a one-key dictionary such as
    {"movieId": [embedding values]}. The function also accepts a plain list so
    the script remains useful if the local export format changes slightly.
    """
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict):
        vector = next(iter(payload.values()))
    elif isinstance(payload, list):
        vector = payload
    else:
        raise ValueError(f"Unsupported JSON vector format in {path}")

    return np.asarray(vector, dtype=np.float64).reshape(-1)


def build_json_fingerprint_index(json_dir: Path) -> tuple[dict[tuple[float, ...], int], pd.DataFrame]:
    """Index MovieLens movieIds by MPNet vector fingerprint."""
    fingerprint_to_movies: dict[tuple[float, ...], list[int]] = defaultdict(list)
    skipped: list[dict[str, str]] = []

    files = sorted(json_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"No JSON feature files found in {json_dir}")

    for path in tqdm(files, desc="Indexing MovieLens MPNet JSON vectors"):
        if not path.stem.isdigit():
            skipped.append({"file": str(path), "reason": "file stem is not numeric"})
            continue
        try:
            fingerprint_to_movies[vector_fingerprint(read_json_vector(path))].append(int(path.stem))
        except Exception as exc:  # noqa: BLE001 - diagnostics are more helpful than stopping on one bad file.
            skipped.append({"file": str(path), "reason": str(exc)})

    unique_index = {
        fingerprint: movie_ids[0]
        for fingerprint, movie_ids in fingerprint_to_movies.items()
        if len(set(movie_ids)) == 1
    }
    diagnostics = pd.DataFrame(
        [
            {
                "json_files": len(files),
                "unique_fingerprints": len(unique_index),
                "duplicate_fingerprints": sum(len(set(ids)) > 1 for ids in fingerprint_to_movies.values()),
                "skipped_json_files": len(skipped),
            }
        ]
    )
    return unique_index, diagnostics


def rebuild_bridge(matrix_path: Path, json_dir: Path, output_path: Path, diagnostics_path: Path, allow_partial: bool) -> None:
    """Match every M3L matrix row to the MovieLens movieId with the same embedding."""
    matrix = np.load(matrix_path, mmap_mode="r")
    if matrix.ndim != 2:
        raise ValueError(f"Expected a 2D MPNet matrix, found shape {matrix.shape}")

    fingerprint_index, diagnostics = build_json_fingerprint_index(json_dir)
    rows: list[dict[str, int]] = []
    missing_item_ids: list[int] = []

    for item_id in tqdm(range(matrix.shape[0]), desc="Matching M3L item rows to MovieLens movieId"):
        movie_id = fingerprint_index.get(vector_fingerprint(matrix[item_id]))
        if movie_id is None:
            missing_item_ids.append(item_id)
            continue
        rows.append({"item_id": item_id, "movieId": movie_id})

    bridge = pd.DataFrame(rows).sort_values("item_id")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
    bridge.to_csv(output_path, index=False)

    diagnostics = diagnostics.assign(
        matrix_rows=matrix.shape[0],
        matched_rows=len(bridge),
        missing_rows=len(missing_item_ids),
        output=str(output_path),
    )
    diagnostics.to_csv(diagnostics_path, index=False)

    print(f"Wrote bridge: {output_path} ({len(bridge):,} rows)")
    print(f"Wrote diagnostics: {diagnostics_path}")

    if missing_item_ids and not allow_partial:
        preview = ", ".join(map(str, missing_item_ids[:10]))
        raise RuntimeError(
            f"{len(missing_item_ids):,} matrix rows could not be matched. "
            f"First missing item ids: {preview}. Re-run with --allow-partial only for diagnostics."
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct data/interim/m3l_internal_to_movielens.csv from MPNet matrix and JSON feature files."
    )
    parser.add_argument("--project-root", default=".", help="Repository or local project root.")
    parser.add_argument("--mpnet-matrix", default=None, help="Path to m3l-20m/text/mpnet.npy.")
    parser.add_argument("--mpnet-json-dir", default=None, help="Directory containing per-movie TEXT_mpnet JSON files.")
    parser.add_argument("--out", default="data/interim/m3l_internal_to_movielens.csv", help="Output CSV path.")
    parser.add_argument(
        "--diagnostics-out",
        default="data/processed/m3l_bridge_diagnostics_from_script.csv",
        help="Small diagnostics CSV path.",
    )
    parser.add_argument("--allow-partial", action="store_true", help="Write partial output instead of failing on misses.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    matrix_path = resolve_existing_path(
        project_root,
        args.mpnet_matrix,
        [
            "m3l-20m/text/mpnet.npy",
            "data/raw/m3l-20m/text/mpnet.npy",
            "data/raw/M3L_20M/text/mpnet.npy",
        ],
        "MPNet matrix",
    )
    json_dir = resolve_existing_path(
        project_root,
        args.mpnet_json_dir,
        [
            "TEXT_mpnet",
            "data/raw/TEXT_mpnet",
            "data/raw/M3L_10M_20M-main/features_json_format/1_text_feat/mpnet",
            "M3L_10M_20M-main/features_json_format/1_text_feat/mpnet",
        ],
        "MPNet JSON directory",
    )
    output_path = Path(args.out)
    output_path = output_path if output_path.is_absolute() else project_root / output_path
    diagnostics_path = Path(args.diagnostics_out)
    diagnostics_path = diagnostics_path if diagnostics_path.is_absolute() else project_root / diagnostics_path
    rebuild_bridge(matrix_path, json_dir, output_path, diagnostics_path, args.allow_partial)


if __name__ == "__main__":
    main()
