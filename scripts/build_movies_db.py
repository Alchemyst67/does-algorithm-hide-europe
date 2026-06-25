from __future__ import annotations

from pathlib import Path
import json
import re
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MOVIELENS_DIR = PROJECT_ROOT / "MovieLens 20M Dataset"
M3L_ROOT = PROJECT_ROOT / "M3L_10M_20M-main"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"

for path in [DATA_INTERIM, DATA_PROCESSED, OUTPUTS]:
    path.mkdir(parents=True, exist_ok=True)

RATING_NROWS = 1_000_000

COLORS = {
    "ink": "#243447",
    "blue": "#2F6B9A",
    "green": "#2E8B57",
    "gold": "#C99700",
    "red": "#B84A4A",
    "teal": "#1F8A8A",
    "violet": "#6E5AA8",
    "gray": "#7A869A",
}

EUROPE_COUNTRIES = {
    "Albania", "Andorra", "Armenia", "Austria", "Azerbaijan", "Belarus", "Belgium",
    "Bosnia and Herzegovina", "Bulgaria", "Croatia", "Cyprus", "Czech Republic",
    "Czechia", "Denmark", "Estonia", "Finland", "France", "Georgia", "Germany",
    "Greece", "Hungary", "Iceland", "Ireland", "Italy", "Kosovo", "Latvia",
    "Liechtenstein", "Lithuania", "Luxembourg", "Malta", "Moldova", "Monaco",
    "Montenegro", "Netherlands", "North Macedonia", "Norway", "Poland", "Portugal",
    "Romania", "Russia", "San Marino", "Serbia", "Slovakia", "Slovenia", "Spain",
    "Sweden", "Switzerland", "Turkey", "Ukraine", "United Kingdom", "Vatican City",
}


def is_git_lfs_pointer(path: Path) -> bool:
    """Detect placeholder files so the pipeline never treats pointers as real data."""
    if not path.exists() or not path.is_file():
        return False
    with path.open("rb") as handle:
        head = handle.read(128)
    return head.startswith(b"version https://git-lfs.github.com/spec/v1")


def file_overview(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        rows.append({
            "path": str(path.relative_to(PROJECT_ROOT)) if path.exists() else str(path),
            "exists": path.exists(),
            "size_mb": path.stat().st_size / 1024**2 if path.exists() and path.is_file() else np.nan,
            "is_lfs_pointer": is_git_lfs_pointer(path),
        })
    return pd.DataFrame(rows)


def parse_title_year(title: str) -> pd.Series:
    if pd.isna(title):
        return pd.Series({"clean_title": np.nan, "year": np.nan})
    match = re.search(r"\((\d{4})\)\s*$", str(title))
    year = int(match.group(1)) if match else np.nan
    clean = re.sub(r"\s*\(\d{4}\)\s*$", "", str(title)).strip()
    return pd.Series({"clean_title": clean, "year": year})


def imdb_numeric_to_tt(value) -> str | pd.NA:
    if pd.isna(value):
        return pd.NA
    return "tt" + str(int(value)).zfill(7)


def maybe_read_csv(path: Path, **kwargs) -> pd.DataFrame | None:
    if not path.exists() or is_git_lfs_pointer(path):
        return None
    return pd.read_csv(path, **kwargs)


def read_m3l10_movies(path: Path) -> pd.DataFrame | None:
    if not path.exists() or is_git_lfs_pointer(path):
        return None
    return pd.read_csv(path, sep="::", engine="python", names=["movieId", "title", "genres"])


def savefig(name: str) -> None:
    target = OUTPUTS / name
    plt.savefig(target, dpi=220, bbox_inches="tight")
    print(f"Saved {target}")


def inventory() -> pd.DataFrame:
    paths = [
        MOVIELENS_DIR / "movie.csv",
        MOVIELENS_DIR / "link.csv",
        MOVIELENS_DIR / "rating.csv",
        MOVIELENS_DIR / "tag.csv",
        MOVIELENS_DIR / "genome_scores.csv",
        MOVIELENS_DIR / "genome_tags.csv",
        M3L_ROOT / "3_process_dataset/ml20m/movies.csv",
        M3L_ROOT / "3_process_dataset/ml20m/ratings.csv",
        M3L_ROOT / "3_process_dataset/ml10m/movies.dat",
        M3L_ROOT / "3_process_dataset/ml10m/ratings.dat",
        M3L_ROOT / "1_download_raw/download_text/REPRO_plot_text.tsv",
        M3L_ROOT / "1_download_raw/download_posters/REPRO_poster_links.tsv",
        M3L_ROOT / "1_download_raw/download_trailers/REPRO_trailer_links.tsv",
        PROJECT_ROOT / "TEXT_mpnet",
        PROJECT_ROOT / "IMG_clip-image",
        PROJECT_ROOT / "m3l-20m/text/mpnet.npy",
        PROJECT_ROOT / "m3l-20m/image/clip_image.npy",
    ]
    overview = file_overview(paths)
    overview.to_csv(OUTPUTS / "27_movies_db_file_inventory.csv", index=False)
    return overview


def load_core_movie_metadata() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    movies = pd.read_csv(MOVIELENS_DIR / "movie.csv")
    links = pd.read_csv(MOVIELENS_DIR / "link.csv")

    m3l20_path = M3L_ROOT / "3_process_dataset/ml20m/movies.csv"
    m3l20 = maybe_read_csv(m3l20_path)
    if m3l20 is None:
        # The current project has MovieLens 20M metadata but not Nico's separate M3L metadata folder.
        # We keep the table explicit instead of pretending that an unavailable file was loaded.
        m3l20 = movies.copy()
        m3l20["m3l20_source_status"] = "using MovieLens 20M metadata; separate M3L ml20m/movies.csv not present"
    else:
        m3l20["m3l20_source_status"] = "loaded from M3L_10M_20M-main"

    m3l10 = read_m3l10_movies(M3L_ROOT / "3_process_dataset/ml10m/movies.dat")

    for frame in [movies, m3l20] + ([m3l10] if m3l10 is not None else []):
        parsed = frame["title"].apply(parse_title_year)
        frame[["clean_title", "year"]] = parsed[["clean_title", "year"]]
    links["imdb_tt"] = links["imdbId"].apply(imdb_numeric_to_tt)

    summary = pd.DataFrame([
        {"dataset": "MovieLens movies", "rows": len(movies), "unique_movieId": movies["movieId"].nunique()},
        {"dataset": "MovieLens links", "rows": len(links), "unique_movieId": links["movieId"].nunique()},
        {"dataset": "M3L 20M movie metadata", "rows": len(m3l20), "unique_movieId": m3l20["movieId"].nunique()},
        {
            "dataset": "M3L 10M movie metadata",
            "rows": len(m3l10) if m3l10 is not None else 0,
            "unique_movieId": m3l10["movieId"].nunique() if m3l10 is not None else 0,
        },
    ])
    summary.to_csv(OUTPUTS / "27_movies_db_core_metadata_summary.csv", index=False)
    return movies, links, m3l20, m3l10


def movie_id_overlap(movies: pd.DataFrame, m3l20: pd.DataFrame, m3l10: pd.DataFrame | None) -> pd.DataFrame:
    sets = {
        "movielens": set(movies["movieId"]),
        "m3l20": set(m3l20["movieId"]),
        "m3l10": set(m3l10["movieId"]) if m3l10 is not None else set(),
    }
    rows = []
    for left, right, label in [
        ("movielens", "m3l20", "MovieLens vs M3L 20M metadata"),
        ("movielens", "m3l10", "MovieLens vs M3L 10M metadata"),
        ("m3l20", "m3l10", "M3L 20M metadata vs M3L 10M metadata"),
    ]:
        union = sets[left] | sets[right]
        rows.append({
            "comparison": label,
            "left": len(sets[left]),
            "right": len(sets[right]),
            "intersection": len(sets[left] & sets[right]),
            "jaccard": len(sets[left] & sets[right]) / len(union) if union else np.nan,
        })
    overlap = pd.DataFrame(rows)
    overlap.to_csv(OUTPUTS / "27_movies_db_movieid_overlap.csv", index=False)
    return overlap


def title_mismatches(movies: pd.DataFrame, m3l20: pd.DataFrame) -> pd.DataFrame:
    mismatches = (
        movies[["movieId", "title"]]
        .merge(m3l20[["movieId", "title"]], on="movieId", how="inner", suffixes=("_movielens", "_m3l20"))
    )
    mismatches = mismatches[mismatches["title_movielens"].ne(mismatches["title_m3l20"])]
    mismatches.to_csv(OUTPUTS / "27_movies_db_title_mismatches.csv", index=False)
    return mismatches


def rating_and_tag_stats() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ratings = pd.read_csv(MOVIELENS_DIR / "rating.csv", nrows=RATING_NROWS, parse_dates=["timestamp"])
    tags = pd.read_csv(MOVIELENS_DIR / "tag.csv", parse_dates=["timestamp"])
    rating_stats = ratings.groupby("movieId").agg(
        rating_count=("rating", "size"),
        rating_mean=("rating", "mean"),
        rating_std=("rating", "std"),
        first_rating_at=("timestamp", "min"),
        last_rating_at=("timestamp", "max"),
    ).reset_index()
    tag_stats = tags.groupby("movieId").agg(
        user_tag_count=("tag", "size"),
        unique_user_tags=("tag", "nunique"),
    ).reset_index()
    rating_stats.to_csv(OUTPUTS / "27_movies_db_rating_stats.csv", index=False)
    tag_stats.to_csv(OUTPUTS / "27_movies_db_tag_stats.csv", index=False)
    return ratings, tags, rating_stats, tag_stats


def genome_summary() -> pd.DataFrame:
    genome_tags = pd.read_csv(MOVIELENS_DIR / "genome_tags.csv")
    chunks = []
    for chunk in pd.read_csv(MOVIELENS_DIR / "genome_scores.csv", chunksize=1_000_000):
        top = chunk.sort_values(["movieId", "relevance"], ascending=[True, False]).groupby("movieId").head(5)
        chunks.append(top)
    genome_top = (
        pd.concat(chunks, ignore_index=True)
        .sort_values(["movieId", "relevance"], ascending=[True, False])
        .groupby("movieId")
        .head(5)
        .merge(genome_tags, on="tagId", how="left")
    )
    summary = genome_top.groupby("movieId").agg(
        top_genome_tags=("tag", lambda s: " | ".join(s.astype(str))),
        top_genome_relevance=("relevance", lambda s: " | ".join(f"{v:.3f}" for v in s)),
    ).reset_index()
    summary.to_csv(OUTPUTS / "27_movies_db_genome_summary.csv", index=False)
    return summary


def load_raw_m3l() -> pd.DataFrame:
    plot_path = M3L_ROOT / "1_download_raw/download_text/REPRO_plot_text.tsv"
    poster_path = M3L_ROOT / "1_download_raw/download_posters/REPRO_poster_links.tsv"
    trailer_path = M3L_ROOT / "1_download_raw/download_trailers/REPRO_trailer_links.tsv"
    frames = []
    if plot_path.exists():
        frames.append(pd.read_csv(plot_path, sep="\t").rename(columns={"movie_id": "movieId"}))
    if poster_path.exists():
        frames.append(pd.read_csv(poster_path, sep="\t").rename(columns={"movie_id": "movieId"}))
    if trailer_path.exists():
        frames.append(pd.read_csv(trailer_path, sep="\t").rename(columns={"movie_id": "movieId"}))
    if not frames:
        return pd.DataFrame(columns=["movieId", "plot", "poster_link", "trailer_link"])
    raw = frames[0]
    for frame in frames[1:]:
        raw = raw.merge(frame, on="movieId", how="outer")
    return raw


def feature_inventory_and_coverage(item_map: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    coverage_ids: dict[str, set[int]] = {}

    json_sources = [
        ("text", "mpnet_json", PROJECT_ROOT / "TEXT_mpnet"),
        ("image", "clip_image_json", PROJECT_ROOT / "IMG_clip-image"),
    ]
    for modality, encoder, folder in json_sources:
        files = sorted(folder.glob("*.json")) if folder.exists() else []
        dim = np.nan
        if files:
            with files[0].open() as handle:
                sample = json.load(handle)
            vector = next(iter(sample.values()))
            dim = len(vector) if isinstance(vector, list) else np.nan
        ids = {int(path.stem) for path in files if path.stem.isdigit()}
        col = f"has_{modality}_{encoder}"
        coverage_ids[col] = ids
        rows.append({
            "source_type": "movieId_json",
            "modality": modality,
            "encoder": encoder,
            "files_or_rows": len(files),
            "embedding_dim": dim,
            "id_space": "MovieLens movieId",
        })

    npy_sources = [
        ("text", "clip_text_matrix", PROJECT_ROOT / "m3l-20m/text/clip_text.npy"),
        ("text", "minilm_matrix", PROJECT_ROOT / "m3l-20m/text/minilm.npy"),
        ("text", "mpnet_matrix", PROJECT_ROOT / "m3l-20m/text/mpnet.npy"),
        ("image", "clip_image_matrix", PROJECT_ROOT / "m3l-20m/image/clip_image.npy"),
        ("image", "vgg_matrix", PROJECT_ROOT / "m3l-20m/image/vgg.npy"),
        ("image", "vit_matrix", PROJECT_ROOT / "m3l-20m/image/vit.npy"),
        ("audio", "ast_matrix", PROJECT_ROOT / "m3l-20m/audio/ast.npy"),
        ("audio", "vggish_matrix", PROJECT_ROOT / "m3l-20m/audio/vggish.npy"),
        ("audio", "whisper_matrix", PROJECT_ROOT / "m3l-20m/audio/whisper.npy"),
        ("video", "mvit_matrix", PROJECT_ROOT / "m3l-20m/video/mvit.npy"),
        ("video", "r2p1d_matrix", PROJECT_ROOT / "m3l-20m/video/r2p1d.npy"),
        ("video", "slowfast_matrix", PROJECT_ROOT / "m3l-20m/video/slowfast.npy"),
    ]
    mapped_items = set(item_map["movieId"].astype(int)) if item_map is not None else set()
    for modality, encoder, path in npy_sources:
        if not path.exists():
            continue
        arr = np.load(path, mmap_mode="r")
        col = f"has_{modality}_{encoder}"
        coverage_ids[col] = mapped_items if item_map is not None and len(item_map) == arr.shape[0] else set()
        rows.append({
            "source_type": "m3l_internal_matrix",
            "modality": modality,
            "encoder": encoder,
            "files_or_rows": int(arr.shape[0]),
            "embedding_dim": int(arr.shape[1]) if arr.ndim == 2 else np.nan,
            "id_space": "M3L item_id mapped to MovieLens movieId",
        })

    all_ids = sorted(set().union(*coverage_ids.values())) if coverage_ids else []
    coverage = pd.DataFrame({"movieId": all_ids})
    for col, ids in coverage_ids.items():
        coverage[col] = coverage["movieId"].isin(ids)
    if len(coverage):
        coverage["feature_family_count"] = coverage.drop(columns=["movieId"]).sum(axis=1)
    else:
        coverage["feature_family_count"] = pd.Series(dtype=int)

    inventory_df = pd.DataFrame(rows).sort_values(["source_type", "modality", "encoder"])
    inventory_df.to_csv(OUTPUTS / "27_movies_db_feature_inventory.csv", index=False)
    coverage.to_csv(OUTPUTS / "27_movies_db_feature_coverage_table.csv", index=False)
    return inventory_df, coverage


def split_pipe(value) -> list[str]:
    if pd.isna(value) or value == "":
        return []
    if isinstance(value, list):
        return value
    return sorted({part.strip().strip("'\"[]") for part in str(value).split("|") if part.strip()})


def load_wikidata_aggregate() -> pd.DataFrame:
    cache = DATA_INTERIM / "wikidata_movie_metadata_extended.csv"
    if not cache.exists():
        return pd.DataFrame(columns=["imdb_tt"])
    wd = pd.read_csv(cache)
    wd = wd.rename(columns={"imdb_id_str": "imdb_tt"})
    for col in [
        "country", "original_language", "language_of_work", "production_company",
        "production_company_country", "production_company_hq_country",
    ]:
        if col in wd.columns:
            wd[col] = wd[col].apply(split_pipe)
    return wd


def build_combined_movies(
    movies: pd.DataFrame,
    links: pd.DataFrame,
    m3l20: pd.DataFrame,
    m3l10: pd.DataFrame | None,
    rating_stats: pd.DataFrame,
    tag_stats: pd.DataFrame,
    genome: pd.DataFrame,
    raw_m3l: pd.DataFrame,
    feature_coverage: pd.DataFrame,
) -> pd.DataFrame:
    m3l10_title = (
        m3l10[["movieId", "title"]].rename(columns={"title": "m3l10_title"})
        if m3l10 is not None else pd.DataFrame(columns=["movieId", "m3l10_title"])
    )
    combined = (
        movies
        .merge(links, on="movieId", how="left")
        .merge(m3l20[["movieId", "title", "m3l20_source_status"]].rename(columns={"title": "m3l20_title"}), on="movieId", how="left")
        .merge(m3l10_title, on="movieId", how="left")
        .merge(rating_stats, on="movieId", how="left")
        .merge(tag_stats, on="movieId", how="left")
        .merge(genome, on="movieId", how="left")
        .merge(raw_m3l, on="movieId", how="left")
        .merge(feature_coverage, on="movieId", how="left")
    )
    combined["imdb_tt"] = combined["imdbId"].apply(imdb_numeric_to_tt)
    combined["in_m3l20_metadata"] = combined["m3l20_title"].notna()
    combined["in_m3l10_metadata"] = combined["m3l10_title"].notna()
    combined["has_plot"] = combined["plot"].notna() if "plot" in combined else False
    combined["has_poster"] = combined["poster_link"].notna() if "poster_link" in combined else False
    combined["has_trailer"] = combined.filter(like="trailer").notna().any(axis=1) if any("trailer" in c for c in combined.columns) else False

    item_map_path = DATA_INTERIM / "m3l_internal_to_movielens.csv"
    if item_map_path.exists():
        item_map = pd.read_csv(item_map_path)
        combined["in_m3l_interaction_items"] = combined["movieId"].isin(item_map["movieId"])
    else:
        combined["in_m3l_interaction_items"] = False

    wd = load_wikidata_aggregate()
    if len(wd):
        combined = combined.merge(wd, on="imdb_tt", how="left")
        combined["has_wikidata_match"] = combined["wikidata_uri"].notna()
        combined["has_country"] = combined["country"].apply(lambda x: isinstance(x, list) and bool(x))
        combined["has_original_language"] = combined["original_language"].apply(lambda x: isinstance(x, list) and bool(x))
        combined["is_european"] = combined["country"].apply(lambda xs: any(country in EUROPE_COUNTRIES for country in xs) if isinstance(xs, list) else False)
        combined["is_non_english"] = combined["original_language"].apply(lambda xs: bool(xs) and "English" not in xs if isinstance(xs, list) else False)
    else:
        combined["has_wikidata_match"] = False
        combined["has_country"] = False
        combined["has_original_language"] = False
        combined["is_european"] = False
        combined["is_non_english"] = False

    bool_cols = [col for col in combined.columns if col.startswith("has_") or col.startswith("in_m3l") or col.startswith("is_")]
    combined[bool_cols] = combined[bool_cols].fillna(False)
    combined["feature_family_count"] = combined["feature_family_count"].fillna(0).astype(int)
    combined.to_csv(DATA_PROCESSED / "combined_movies_db.csv", index=False)
    try:
        combined.to_parquet(DATA_PROCESSED / "combined_movies_db.parquet", index=False)
    except Exception as exc:
        print(f"Parquet write skipped: {exc}")
    return combined


def audit_tables(combined: pd.DataFrame, ratings: pd.DataFrame, tags: pd.DataFrame) -> dict[str, pd.DataFrame]:
    feature_cols = [col for col in combined.columns if col.startswith("has_") and ("json" in col or "matrix" in col)]
    coverage_cols = [
        "in_m3l20_metadata", "in_m3l10_metadata", "in_m3l_interaction_items",
        "has_plot", "has_poster", "has_trailer", "has_wikidata_match",
        "has_country", "has_original_language",
    ] + feature_cols
    coverage_report = (
        combined[coverage_cols]
        .mean()
        .rename("coverage_rate")
        .reset_index()
        .rename(columns={"index": "field"})
        .sort_values("coverage_rate", ascending=False)
    )
    summary_stats = pd.DataFrame({
        "metric": [
            "movies",
            "movies with ratings in loaded sample",
            "movies with user tags",
            "movies with genome tags",
            "movies in M3L interaction item universe",
            "movies with text JSON feature",
            "movies with image JSON feature",
            "movies with all available feature families",
            "movies with Wikidata match",
            "median rating count",
            "mean rating",
        ],
        "value": [
            len(combined),
            combined["rating_count"].notna().sum(),
            combined["user_tag_count"].notna().sum(),
            combined["top_genome_tags"].notna().sum(),
            combined["in_m3l_interaction_items"].sum(),
            combined["has_text_mpnet_json"].sum() if "has_text_mpnet_json" in combined else 0,
            combined["has_image_clip_image_json"].sum() if "has_image_clip_image_json" in combined else 0,
            combined["feature_family_count"].eq(len(feature_cols)).sum() if feature_cols else 0,
            combined["has_wikidata_match"].sum(),
            combined["rating_count"].median(),
            combined["rating_mean"].mean(),
        ],
    })

    audit_base = combined.copy()
    audit_base["rating_count_filled"] = audit_base["rating_count"].fillna(0)
    audit_base["rating_mean_filled"] = audit_base["rating_mean"].fillna(audit_base["rating_mean"].mean())
    c = audit_base["rating_mean"].mean()
    m = audit_base["rating_count_filled"].quantile(0.90)
    v = audit_base["rating_count_filled"]
    r = audit_base["rating_mean_filled"]
    audit_base["weighted_rating_score"] = (v / (v + m) * r) + (m / (v + m) * c)
    audit_base["decade"] = (audit_base["year"] // 10 * 10).astype("Int64").astype(str) + "s"
    audit_base.loc[audit_base["year"].isna(), "decade"] = "unknown"

    genre_base = (
        audit_base.assign(genre=lambda df: df["genres"].fillna("(no genres listed)").str.split("|"))
        .explode("genre")
    )
    genre_audit = slice_audit_table(genre_base, "genre", top_k=100, min_available=100)
    decade_audit = slice_audit_table(audit_base, "decade", top_k=100, min_available=25)
    user_summary, concentration = user_discovery(ratings, tags, combined)

    tables = {
        "summary_stats": summary_stats,
        "coverage_report": coverage_report,
        "genre_audit": genre_audit,
        "decade_audit": decade_audit,
        "user_summary": user_summary,
        "user_concentration": concentration,
    }
    for name, table in tables.items():
        table.to_csv(OUTPUTS / f"27_movies_db_{name}.csv", index=False)
    return tables


def slice_audit_table(data: pd.DataFrame, slice_col: str, top_k: int = 100, min_available: int = 50) -> pd.DataFrame:
    base = data.copy()
    popularity_top = set(base.nlargest(top_k, "rating_count_filled")["movieId"])
    weighted_top = set(base.nlargest(top_k, "weighted_rating_score")["movieId"])
    rows = []
    total_movies = base["movieId"].nunique()
    total_interest = base["rating_count_filled"].sum()
    for label, group in base.groupby(slice_col, dropna=False):
        label = "unknown" if pd.isna(label) else label
        available = group["movieId"].nunique()
        if available < min_available:
            continue
        interest = group["rating_count_filled"].sum()
        popularity_visible = group["movieId"].isin(popularity_top).sum()
        weighted_visible = group["movieId"].isin(weighted_top).sum()
        rows.append({
            slice_col: label,
            "available_movies": available,
            "availability_share": available / total_movies,
            "interest_events": int(interest),
            "interest_share": interest / total_interest if total_interest else np.nan,
            "popularity_topk_movies": int(popularity_visible),
            "popularity_topk_share": popularity_visible / top_k,
            "weighted_topk_movies": int(weighted_visible),
            "weighted_topk_share": weighted_visible / top_k,
            "interest_minus_popularity_visibility": (interest / total_interest) - (popularity_visible / top_k) if total_interest else np.nan,
            "interest_minus_weighted_visibility": (interest / total_interest) - (weighted_visible / top_k) if total_interest else np.nan,
        })
    return pd.DataFrame(rows).sort_values("interest_share", ascending=False)


def user_discovery(ratings: pd.DataFrame, tags: pd.DataFrame, combined: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    user_stats = ratings.groupby("userId").agg(
        rating_count=("movieId", "size"),
        unique_movies=("movieId", "nunique"),
        mean_rating=("rating", "mean"),
        rating_std=("rating", "std"),
        min_rating_at=("timestamp", "min"),
        max_rating_at=("timestamp", "max"),
    ).reset_index()
    user_stats["active_days"] = (user_stats["max_rating_at"] - user_stats["min_rating_at"]).dt.days.clip(lower=0)
    user_stats["rating_std"] = user_stats["rating_std"].fillna(0)
    summary = pd.DataFrame({
        "metric": [
            "users in loaded ratings",
            "ratings in loaded sample",
            "movies rated in loaded sample",
            "median ratings per user",
            "mean ratings per user",
            "90th percentile ratings per user",
            "median user mean rating",
            "median active span in days",
        ],
        "value": [
            user_stats["userId"].nunique(),
            len(ratings),
            ratings["movieId"].nunique(),
            user_stats["rating_count"].median(),
            user_stats["rating_count"].mean(),
            user_stats["rating_count"].quantile(0.90),
            user_stats["mean_rating"].median(),
            user_stats["active_days"].median(),
        ],
    })
    activity = user_stats.sort_values("rating_count", ascending=False).reset_index(drop=True)
    concentration = pd.DataFrame([
        {"user_group": "top 1% most active users", "rating_share": activity.head(max(1, int(len(activity) * 0.01)))["rating_count"].sum() / activity["rating_count"].sum()},
        {"user_group": "top 5% most active users", "rating_share": activity.head(max(1, int(len(activity) * 0.05)))["rating_count"].sum() / activity["rating_count"].sum()},
        {"user_group": "top 10% most active users", "rating_share": activity.head(max(1, int(len(activity) * 0.10)))["rating_count"].sum() / activity["rating_count"].sum()},
    ])
    user_stats.to_csv(OUTPUTS / "27_movies_db_user_stats.csv", index=False)
    return summary, concentration


def plots(combined: pd.DataFrame, tables: dict[str, pd.DataFrame]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")

    coverage = tables["coverage_report"].sort_values("coverage_rate")
    fig, ax = plt.subplots(figsize=(10, max(4, 0.28 * len(coverage))))
    ax.barh(coverage["field"], coverage["coverage_rate"], color=COLORS["blue"])
    ax.set_title("Movies DB coverage by source and feature family", weight="bold")
    ax.set_xlabel("Share of MovieLens movies covered")
    ax.set_xlim(0, 1)
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    for i, value in enumerate(coverage["coverage_rate"]):
        ax.text(min(value + 0.01, 0.96), i, f"{value:.1%}", va="center", fontsize=8)
    fig.tight_layout()
    savefig("27_movies_db_coverage.png")
    plt.close(fig)

    rated = combined.dropna(subset=["rating_mean", "rating_count"]).copy()
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    axes[0].hist(rated["rating_mean"], bins=30, color=COLORS["green"], edgecolor="white")
    axes[0].set_title("Average rating distribution", weight="bold")
    axes[0].set_xlabel("Average rating")
    axes[0].set_ylabel("Movies")
    axes[1].hist(np.log10(rated["rating_count"].clip(lower=1)), bins=35, color=COLORS["gold"], edgecolor="white")
    axes[1].set_title("Rating count distribution", weight="bold")
    axes[1].set_xlabel("log10(rating count)")
    axes[1].set_ylabel("Movies")
    fig.tight_layout()
    savefig("28_movies_db_rating_distribution.png")
    plt.close(fig)

    genre_plot = tables["genre_audit"].head(12).set_index("genre")[
        ["availability_share", "interest_share", "popularity_topk_share", "weighted_topk_share"]
    ].sort_values("interest_share")
    fig, ax = plt.subplots(figsize=(11, 6))
    genre_plot.plot(kind="barh", ax=ax, width=0.78)
    ax.set_title("Genre shares across catalogue, interest and baseline Top-K visibility", weight="bold")
    ax.set_xlabel("Share")
    ax.set_ylabel("")
    ax.xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.legend(["Catalogue availability", "User interest", "Popularity Top-100", "Weighted-rating Top-100"], loc="lower right")
    fig.tight_layout()
    savefig("29_movies_db_genre_interest_visibility.png")
    plt.close(fig)

    concentration = tables["user_concentration"]
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.bar(concentration["user_group"], concentration["rating_share"], color=COLORS["red"])
    ax.set_title("Rating concentration among the most active users", weight="bold")
    ax.set_ylabel("Share of loaded ratings")
    ax.yaxis.set_major_formatter(lambda y, pos: f"{y:.0%}")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    savefig("30_movies_db_user_concentration.png")
    plt.close(fig)


def readme(tables: dict[str, pd.DataFrame], combined: pd.DataFrame) -> None:
    summary = tables["summary_stats"].set_index("metric")["value"].to_dict()
    lines = [
        "# Movies DB Outputs",
        "",
        "Generated by `scripts/build_movies_db.py`.",
        "",
        "## Scope",
        "",
        "The Movies DB is a movie-level integration table built from MovieLens 20M metadata, ratings, tags, genome tags, available M3L feature coverage, M3L interaction-item mapping and cached Wikidata enrichment. It does not fabricate missing plot/poster/trailer data when Nico's raw M3L TSV folders are not present locally.",
        "",
        "## Key Numbers",
        "",
        f"- Movies in combined table: {len(combined):,}",
        f"- Movies with ratings in loaded sample: {int(summary.get('movies with ratings in loaded sample', 0)):,}",
        f"- Movies with user tags: {int(summary.get('movies with user tags', 0)):,}",
        f"- Movies in M3L interaction item universe: {int(summary.get('movies in M3L interaction item universe', 0)):,}",
        f"- Movies with Wikidata match: {int(summary.get('movies with Wikidata match', 0)):,}",
        "",
        "## Main Files",
        "",
        "- `data/processed/combined_movies_db.csv`",
        "- `data/processed/combined_movies_db.parquet` if parquet support is available",
        "- `outputs/27_movies_db_summary_stats.csv`",
        "- `outputs/27_movies_db_coverage_report.csv`",
        "- `outputs/27_movies_db_genre_audit.csv`",
        "- `outputs/27_movies_db_user_summary.csv`",
        "- `outputs/27_movies_db_coverage.png`",
        "- `outputs/28_movies_db_rating_distribution.png`",
        "- `outputs/29_movies_db_genre_interest_visibility.png`",
        "- `outputs/30_movies_db_user_concentration.png`",
        "",
        "## Interpretation Note",
        "",
        "The Movies DB is the data foundation for the cultural-prominence audit. Baseline Top-K visibility plots in this layer are diagnostic leads, not final recommender findings.",
    ]
    (OUTPUTS / "27_movies_db_README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    inv = inventory()
    movies, links, m3l20, m3l10 = load_core_movie_metadata()
    overlap = movie_id_overlap(movies, m3l20, m3l10)
    mismatches = title_mismatches(movies, m3l20)
    ratings, tags, rating_stats, tag_stats = rating_and_tag_stats()
    genome = genome_summary()
    raw_m3l = load_raw_m3l()
    item_map_path = DATA_INTERIM / "m3l_internal_to_movielens.csv"
    item_map = pd.read_csv(item_map_path) if item_map_path.exists() else None
    feature_inventory, feature_coverage = feature_inventory_and_coverage(item_map)
    combined = build_combined_movies(movies, links, m3l20, m3l10, rating_stats, tag_stats, genome, raw_m3l, feature_coverage)
    tables = audit_tables(combined, ratings, tags)
    plots(combined, tables)
    readme(tables, combined)

    print("\nMovies DB pipeline complete.")
    print(tables["summary_stats"].to_string(index=False))
    print("\nMovieId overlap:")
    print(overlap.to_string(index=False))
    print(f"\nTitle mismatches against loaded M3L 20M metadata: {len(mismatches):,}")
    print("\nFeature inventory:")
    print(feature_inventory.to_string(index=False))
    print("\nFile inventory warning rows:")
    print(inv[(~inv["exists"]) | (inv["is_lfs_pointer"])].to_string(index=False))


if __name__ == "__main__":
    main()
