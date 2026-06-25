"""Build an optional deeper Wikidata enrichment for the visibility-DNA analysis.

The main audit labels use country, language and production-company metadata.
This script adds a bounded second layer for interpretation: director
citizenship, filming-location country and award signals. It is intentionally
described as an enrichment subset, not as a replacement for the main model
results.
"""

from __future__ import annotations

import argparse
import ast
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm


WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
USER_AGENT = "WU-Data-Algorithmic-Governance-European-Visibility-DNA/1.0"
WIDE_EUROPE = {
    "Albania",
    "Austria",
    "Belgium",
    "Bosnia and Herzegovina",
    "Bulgaria",
    "Croatia",
    "Cyprus",
    "Czech Republic",
    "Czechia",
    "Denmark",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Iceland",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Luxembourg",
    "Malta",
    "Moldova",
    "Montenegro",
    "Netherlands",
    "North Macedonia",
    "Norway",
    "Poland",
    "Portugal",
    "Romania",
    "Serbia",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
    "Switzerland",
    "Turkey",
    "Ukraine",
    "United Kingdom",
}
OUTPUT_COLUMNS = [
    "wikidata_uri",
    "directors",
    "director_citizenship",
    "filming_location_country",
    "award_count",
    "award_examples",
]


def parse_list(value: object) -> list[str]:
    """Parse list-valued columns stored either as Python-list strings or pipes."""
    if isinstance(value, list):
        return [str(item) for item in value if str(item)]
    if pd.isna(value):
        return []
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(item) for item in parsed if str(item) and str(item).lower() != "nan"]
    except (SyntaxError, ValueError):
        pass
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    return [text] if text and text.lower() != "nan" else []


def qid_from_uri(uri: object) -> str | None:
    if not isinstance(uri, str) or "/entity/" not in uri:
        return None
    qid = uri.rsplit("/", 1)[-1]
    return qid if qid.startswith("Q") else None


def resolve_movie_db(project_root: Path, explicit: str | None) -> Path:
    candidates = [
        explicit,
        "data/processed/combined_movies_db.csv",
        "data/processed/movie_database_preview.csv",
    ]
    for candidate in candidates:
        if not candidate:
            continue
        path = Path(candidate)
        path = path if path.is_absolute() else project_root / path
        if path.exists():
            return path
    raise FileNotFoundError(
        "No movie database found. Run notebooks/01_data_foundation_movies_db.ipynb first "
        "or provide --movie-db with a CSV containing wikidata_uri, country and original_language."
    )


def select_priority_subset(movie_db: pd.DataFrame, limit: int) -> pd.DataFrame:
    """Prioritise rows where deeper metadata is most useful for the audit story."""
    required = {"wikidata_uri", "country", "original_language"}
    missing = required - set(movie_db.columns)
    if missing:
        raise KeyError(f"Movie database is missing required columns: {sorted(missing)}")

    rating_col = "rating_count_train" if "rating_count_train" in movie_db.columns else "rating_count"
    if rating_col not in movie_db.columns:
        movie_db[rating_col] = 0

    data = movie_db.copy()
    data["country_list"] = data["country"].apply(parse_list)
    data["language_list"] = data["original_language"].apply(parse_list)
    data["is_european_wide"] = data["country_list"].apply(lambda xs: any(x in WIDE_EUROPE for x in xs))
    data["is_non_english"] = data["language_list"].apply(lambda xs: bool(xs) and "English" not in xs)
    data["priority_score"] = (
        data["is_european_wide"].astype(int) * 4
        + data["is_non_english"].astype(int) * 2
        + data[rating_col].fillna(0).rank(pct=True)
    )
    priority = data[data["wikidata_uri"].notna()].sort_values(
        ["priority_score", rating_col], ascending=[False, False]
    )
    return priority.head(limit).copy()


def query_wikidata_dna(qids: list[str], batch_size: int, sleep_seconds: float, strict: bool) -> pd.DataFrame:
    rows: list[dict[str, str | int | None]] = []
    headers = {"Accept": "application/sparql-results+json", "User-Agent": USER_AGENT}

    for start in tqdm(range(0, len(qids), batch_size), desc="Querying Wikidata visibility-DNA"):
        batch = qids[start : start + batch_size]
        values = " ".join(f"wd:{qid}" for qid in batch)
        query = f"""
        SELECT ?film
               (GROUP_CONCAT(DISTINCT ?directorLabel; separator="|") AS ?directors)
               (GROUP_CONCAT(DISTINCT ?directorCitizenshipLabel; separator="|") AS ?directorCitizenship)
               (GROUP_CONCAT(DISTINCT ?filmingCountryLabel; separator="|") AS ?filmingCountry)
               (COUNT(DISTINCT ?award) AS ?awardCount)
               (GROUP_CONCAT(DISTINCT ?awardLabel; separator="|") AS ?awardExamples)
        WHERE {{
          VALUES ?film {{ {values} }}
          OPTIONAL {{
            ?film wdt:P57 ?director .
            ?director rdfs:label ?directorLabel .
            FILTER(LANG(?directorLabel) = "en")
            OPTIONAL {{ ?director wdt:P27 ?directorCitizenship . }}
            OPTIONAL {{
              ?directorCitizenship rdfs:label ?directorCitizenshipLabel .
              FILTER(LANG(?directorCitizenshipLabel) = "en")
            }}
          }}
          OPTIONAL {{
            ?film wdt:P915 ?filmingLocation .
            OPTIONAL {{ ?filmingLocation wdt:P17 ?filmingCountry . }}
            OPTIONAL {{
              ?filmingCountry rdfs:label ?filmingCountryLabel .
              FILTER(LANG(?filmingCountryLabel) = "en")
            }}
          }}
          OPTIONAL {{
            ?film wdt:P166 ?award .
            OPTIONAL {{
              ?award rdfs:label ?awardLabel .
              FILTER(LANG(?awardLabel) = "en")
            }}
          }}
        }}
        GROUP BY ?film
        """
        try:
            response = requests.get(
                WIKIDATA_ENDPOINT,
                params={"query": query, "format": "json"},
                headers=headers,
                timeout=60,
            )
            response.raise_for_status()
            bindings = response.json()["results"]["bindings"]
        except Exception as exc:  # noqa: BLE001 - cache should still be useful if one public batch fails.
            if strict:
                raise
            print(f"Skipping failed DNA batch {start}: {exc}")
            time.sleep(2)
            continue

        for binding in bindings:
            rows.append(
                {
                    "wikidata_uri": binding.get("film", {}).get("value"),
                    "directors": binding.get("directors", {}).get("value", ""),
                    "director_citizenship": binding.get("directorCitizenship", {}).get("value", ""),
                    "filming_location_country": binding.get("filmingCountry", {}).get("value", ""),
                    "award_count": int(binding.get("awardCount", {}).get("value", 0)),
                    "award_examples": binding.get("awardExamples", {}).get("value", ""),
                }
            )
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
    parser = argparse.ArgumentParser(description="Query a bounded deeper Wikidata enrichment subset.")
    parser.add_argument("--project-root", default=".", help="Repository or local project root.")
    parser.add_argument("--movie-db", default=None, help="CSV with wikidata_uri, country and original_language.")
    parser.add_argument("--limit", type=int, default=600, help="Priority movies to request.")
    parser.add_argument("--batch-size", type=int, default=25, help="Wikidata QIDs per SPARQL query.")
    parser.add_argument("--sleep", type=float, default=0.2, help="Pause between requests in seconds.")
    parser.add_argument(
        "--out",
        default="data/interim/wikidata_visibility_dna_extra.csv",
        help="Output cache CSV path.",
    )
    parser.add_argument(
        "--status-out",
        default="data/processed/visibility_dna_enrichment_status_from_script.csv",
        help="Small status table path.",
    )
    parser.add_argument("--force", action="store_true", help="Ignore existing cache and query from scratch.")
    parser.add_argument("--strict", action="store_true", help="Raise instead of skipping failed Wikidata batches.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    movie_db_path = resolve_movie_db(project_root, args.movie_db)
    output_path = Path(args.out)
    output_path = output_path if output_path.is_absolute() else project_root / output_path
    status_path = Path(args.status_out)
    status_path = status_path if status_path.is_absolute() else project_root / status_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)

    movie_db = pd.read_csv(movie_db_path)
    priority = select_priority_subset(movie_db, args.limit)
    priority["qid"] = priority["wikidata_uri"].apply(qid_from_uri)
    wanted_uris = set(priority["wikidata_uri"].dropna())
    qids = [qid for qid in priority["qid"].dropna().unique().tolist() if qid]

    cache = load_cache(output_path, args.force)
    cached_uris = set(cache["wikidata_uri"].dropna().astype(str))
    missing_qids = [qid for qid in qids if f"http://www.wikidata.org/entity/{qid}" not in cached_uris]

    print(f"Movie database: {movie_db_path}")
    print(f"Priority movies: {len(priority):,}")
    print(f"Cached DNA rows: {len(cache):,}")
    print(f"Missing QIDs to query: {len(missing_qids):,}")

    if missing_qids:
        new_rows = query_wikidata_dna(missing_qids, args.batch_size, args.sleep, args.strict)
        cache = pd.concat([cache, new_rows], ignore_index=True).drop_duplicates("wikidata_uri", keep="first")

    cache.to_csv(output_path, index=False)
    status = pd.DataFrame(
        [
            {
                "source": "Wikidata visibility-DNA enrichment",
                "status": "queried_or_loaded",
                "rows": len(cache),
                "priority_movies_requested": len(priority),
                "coverage_of_priority_movies": len(wanted_uris & set(cache["wikidata_uri"].dropna()))
                / max(len(wanted_uris), 1),
                "notes": "Director citizenship, filming-location country and award counts for a bounded priority subset.",
            },
            {
                "source": "TMDb provider layer",
                "status": "not_queried_missing_api_key",
                "rows": 0,
                "priority_movies_requested": 0,
                "coverage_of_priority_movies": 0,
                "notes": "TMDb ids exist locally, but API credentials are required. No scraping or synthetic provider data was used.",
            },
        ]
    )
    status.to_csv(status_path, index=False)
    print(f"Wrote DNA cache: {output_path} ({len(cache):,} rows)")
    print(f"Wrote status table: {status_path}")


if __name__ == "__main__":
    main()
