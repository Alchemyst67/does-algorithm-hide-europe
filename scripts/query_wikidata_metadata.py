"""Query Wikidata film metadata used for the cultural-prominence labels.

This script is the script version of the enrichment logic shown in
notebooks/01_data_foundation_movies_db.ipynb. It reads MovieLens IMDb ids,
queries the Wikidata Query Service through property P345, and caches country,
language and production-company metadata under data/interim/.
"""

from __future__ import annotations

import argparse
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm


WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "WU-Data-Algorithmic-Governance-Student-Project/1.0"
OUTPUT_COLUMNS = [
    "imdb_id_str",
    "wikidata_uri",
    "title_wikidata",
    "country",
    "original_language",
    "language_of_work",
    "publication_date",
    "production_company",
    "production_company_country",
    "production_company_hq_country",
]


def resolve_existing_path(project_root: Path, explicit: str | None, candidates: list[str], label: str) -> Path:
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


def read_links_table(path: Path) -> pd.DataFrame:
    """Read MovieLens link(s).csv from either an extracted CSV or the original zip."""
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path, "r") as archive:
            candidates = [
                name
                for name in archive.namelist()
                if name.lower().endswith("/links.csv") or name.lower().endswith("/link.csv")
            ]
            if not candidates:
                raise FileNotFoundError(f"No links.csv or link.csv found inside {path}")
            with archive.open(candidates[0]) as handle:
                return pd.read_csv(handle)
    return pd.read_csv(path)


def imdb_numeric_to_tt(value: object) -> str | None:
    if pd.isna(value):
        return None
    try:
        return f"tt{int(value):07d}"
    except (TypeError, ValueError):
        text = str(value)
        return text if text.startswith("tt") else None


def imdb_ids_from_links(links: pd.DataFrame) -> list[str]:
    if "imdb_id_str" in links.columns:
        ids = links["imdb_id_str"]
    elif "imdbId" in links.columns:
        ids = links["imdbId"].apply(imdb_numeric_to_tt)
    elif "imdb_id" in links.columns:
        ids = links["imdb_id"].apply(imdb_numeric_to_tt)
    else:
        raise KeyError("Expected a MovieLens links table with imdbId or imdb_id_str.")
    return [x for x in pd.Series(ids).dropna().astype(str).unique().tolist() if x.startswith("tt")]


def query_batch(imdb_ids: list[str], strict: bool) -> list[dict[str, str | None]]:
    values = " ".join(f'"{imdb_id}"' for imdb_id in imdb_ids)
    query = f"""
    SELECT ?imdb ?film ?filmLabel
      (GROUP_CONCAT(DISTINCT ?countryLabel; separator="|") AS ?countries)
      (GROUP_CONCAT(DISTINCT ?originalLanguageLabel; separator="|") AS ?originalLanguages)
      (GROUP_CONCAT(DISTINCT ?workLanguageLabel; separator="|") AS ?workLanguages)
      (SAMPLE(?publicationDate) AS ?publicationDate)
      (GROUP_CONCAT(DISTINCT ?productionCompanyLabel; separator="|") AS ?productionCompanies)
      (GROUP_CONCAT(DISTINCT ?companyCountryLabel; separator="|") AS ?productionCompanyCountries)
      (GROUP_CONCAT(DISTINCT ?companyHqCountryLabel; separator="|") AS ?productionCompanyHqCountries)
    WHERE {{
      VALUES ?imdb {{ {values} }}
      ?film wdt:P345 ?imdb .
      OPTIONAL {{ ?film wdt:P495 ?country . }}
      OPTIONAL {{ ?film wdt:P364 ?originalLanguage . }}
      OPTIONAL {{ ?film wdt:P407 ?workLanguage . }}
      OPTIONAL {{ ?film wdt:P577 ?publicationDate . }}
      OPTIONAL {{
        ?film wdt:P272 ?productionCompany .
        OPTIONAL {{ ?productionCompany wdt:P17 ?companyCountry . }}
        OPTIONAL {{ ?productionCompany wdt:P159/wdt:P17 ?companyHqCountry . }}
      }}
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    GROUP BY ?imdb ?film ?filmLabel
    """

    try:
        response = requests.get(
            WIKIDATA_ENDPOINT,
            params={"query": query, "format": "json"},
            headers={"User-Agent": USER_AGENT},
            timeout=90,
        )
        response.raise_for_status()
        bindings = response.json()["results"]["bindings"]
    except Exception as exc:  # noqa: BLE001 - network/SPARQL failures should be visible but cache-safe.
        if strict:
            raise
        print(f"Skipping failed Wikidata batch: {exc}")
        return []

    rows: list[dict[str, str | None]] = []
    for row in bindings:
        rows.append(
            {
                "imdb_id_str": row.get("imdb", {}).get("value"),
                "wikidata_uri": row.get("film", {}).get("value"),
                "title_wikidata": row.get("filmLabel", {}).get("value"),
                "country": row.get("countries", {}).get("value"),
                "original_language": row.get("originalLanguages", {}).get("value"),
                "language_of_work": row.get("workLanguages", {}).get("value"),
                "publication_date": row.get("publicationDate", {}).get("value"),
                "production_company": row.get("productionCompanies", {}).get("value"),
                "production_company_country": row.get("productionCompanyCountries", {}).get("value"),
                "production_company_hq_country": row.get("productionCompanyHqCountries", {}).get("value"),
            }
        )
    return rows


def query_wikidata(imdb_ids: list[str], batch_size: int, sleep_seconds: float, strict: bool) -> pd.DataFrame:
    rows: list[dict[str, str | None]] = []
    for start in tqdm(range(0, len(imdb_ids), batch_size), desc="Querying Wikidata film metadata"):
        rows.extend(query_batch(imdb_ids[start : start + batch_size], strict=strict))
        time.sleep(sleep_seconds)
    return pd.DataFrame(rows, columns=OUTPUT_COLUMNS)


def load_cache(path: Path, force: bool) -> pd.DataFrame:
    if path.exists() and not force:
        cache = pd.read_csv(path)
        for column in OUTPUT_COLUMNS:
            if column not in cache.columns:
                cache[column] = pd.NA
        return cache[OUTPUT_COLUMNS]
    return pd.DataFrame(columns=OUTPUT_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Query and cache Wikidata metadata for MovieLens IMDb ids.")
    parser.add_argument("--project-root", default=".", help="Repository or local project root.")
    parser.add_argument("--links", default=None, help="MovieLens links CSV, link CSV or ml-20m.zip.")
    parser.add_argument(
        "--out",
        default="data/interim/wikidata_movie_metadata_extended.csv",
        help="Output cache CSV path.",
    )
    parser.add_argument("--batch-size", type=int, default=150, help="IMDb ids per SPARQL query.")
    parser.add_argument("--sleep", type=float, default=0.15, help="Pause between requests in seconds.")
    parser.add_argument("--limit", type=int, default=None, help="Optional small limit for testing.")
    parser.add_argument("--force", action="store_true", help="Ignore an existing cache and query from scratch.")
    parser.add_argument("--strict", action="store_true", help="Raise instead of skipping failed Wikidata batches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    links_path = resolve_existing_path(
        project_root,
        args.links,
        [
            "data/raw/ml-20m.zip",
            "data/raw/ml-20m/links.csv",
            "data/raw/ml-20m/link.csv",
            "MovieLens 20M Dataset/link.csv",
            "archive/link.csv",
            "data/raw/links.csv",
            "data/raw/link.csv",
        ],
        "MovieLens links table",
    )
    output_path = Path(args.out)
    output_path = output_path if output_path.is_absolute() else project_root / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    links = read_links_table(links_path)
    imdb_ids = imdb_ids_from_links(links)
    if args.limit is not None:
        imdb_ids = imdb_ids[: args.limit]

    cache = load_cache(output_path, force=args.force)
    cached_ids = set(cache["imdb_id_str"].dropna().astype(str))
    missing = [imdb_id for imdb_id in imdb_ids if imdb_id not in cached_ids]

    print(f"MovieLens IMDb ids: {len(imdb_ids):,}")
    print(f"Cached Wikidata rows: {len(cache):,}")
    print(f"Missing IMDb ids to query: {len(missing):,}")

    if missing:
        new_rows = query_wikidata(missing, args.batch_size, args.sleep, args.strict)
        cache = pd.concat([cache, new_rows], ignore_index=True)
        cache = cache.drop_duplicates("imdb_id_str", keep="first")

    cache.to_csv(output_path, index=False)
    print(f"Wrote Wikidata cache: {output_path} ({len(cache):,} rows)")


if __name__ == "__main__":
    main()
