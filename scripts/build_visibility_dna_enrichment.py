from __future__ import annotations

import ast
import time
from pathlib import Path

import pandas as pd
import requests
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "cultural_prominence_audit" / "outputs" / "final_notebook_tables"
DATA_INTERIM.mkdir(parents=True, exist_ok=True)
OUTPUTS.mkdir(parents=True, exist_ok=True)

MOVIE_DB = DATA_PROCESSED / "combined_movies_db.csv"
DNA_CACHE = DATA_INTERIM / "wikidata_visibility_dna_extra.csv"
DNA_STATUS = OUTPUTS / "visibility_dna_enrichment_status.csv"

WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"
WIDE_EUROPE = {
    "Austria", "Germany", "France", "Italy", "Spain", "Portugal", "Netherlands",
    "Belgium", "Luxembourg", "Ireland", "Denmark", "Sweden", "Finland",
    "Poland", "Czech Republic", "Czechia", "Slovakia", "Hungary", "Slovenia",
    "Croatia", "Romania", "Bulgaria", "Greece", "Cyprus", "Malta", "Estonia",
    "Latvia", "Lithuania", "Norway", "Switzerland", "Iceland", "United Kingdom",
    "Ukraine", "Serbia", "Bosnia and Herzegovina", "Montenegro", "North Macedonia",
    "Albania", "Moldova", "Russia", "Turkey",
}


def parse_list(value):
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return value
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(x) for x in parsed if str(x) and str(x).lower() != "nan"]
    except Exception:
        pass
    if "|" in text:
        return [part.strip() for part in text.split("|") if part.strip()]
    return [text] if text and text.lower() != "nan" else []


def qid_from_uri(uri: str) -> str | None:
    if not isinstance(uri, str) or "/entity/" not in uri:
        return None
    qid = uri.rsplit("/", 1)[-1]
    return qid if qid.startswith("Q") else None


def query_wikidata_dna(qids: list[str], batch_size: int = 25, sleep: float = 0.2) -> pd.DataFrame:
    rows: list[dict] = []
    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "WU-Data-Algorithmic-Governance-European-Visibility-DNA/1.0",
    }

    for start in tqdm(range(0, len(qids), batch_size), desc="Wikidata DNA batches"):
        batch = qids[start:start + batch_size]
        values = " ".join(f"wd:{qid}" for qid in batch)
        query = f"""
        SELECT ?film
               (GROUP_CONCAT(DISTINCT ?directorLabel; separator="|") AS ?directors)
               (GROUP_CONCAT(DISTINCT ?directorCitizenshipLabel; separator="|") AS ?director_citizenship)
               (GROUP_CONCAT(DISTINCT ?filmingCountryLabel; separator="|") AS ?filming_location_country)
               (COUNT(DISTINCT ?award) AS ?award_count)
               (GROUP_CONCAT(DISTINCT ?awardLabel; separator="|") AS ?award_examples)
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
                timeout=45,
            )
        except requests.RequestException as exc:
            print(f"Skipping batch {start}: {exc}")
            time.sleep(2)
            continue
        if response.status_code != 200:
            print(f"Skipping batch {start}: HTTP {response.status_code} {response.text[:160]}")
            time.sleep(2)
            continue

        for binding in response.json()["results"]["bindings"]:
            film_uri = binding.get("film", {}).get("value")
            rows.append({
                "wikidata_uri": film_uri,
                "directors": binding.get("directors", {}).get("value", ""),
                "director_citizenship": binding.get("director_citizenship", {}).get("value", ""),
                "filming_location_country": binding.get("filming_location_country", {}).get("value", ""),
                "award_count": int(binding.get("award_count", {}).get("value", 0)),
                "award_examples": binding.get("award_examples", {}).get("value", ""),
            })
        time.sleep(sleep)

    return pd.DataFrame(rows).drop_duplicates("wikidata_uri")


def main(limit: int = 600) -> None:
    if not MOVIE_DB.exists():
        raise FileNotFoundError(f"Missing movie database: {MOVIE_DB}")

    movie_db = pd.read_csv(MOVIE_DB)
    movie_db["country_list"] = movie_db["country"].apply(parse_list)
    movie_db["language_list"] = movie_db["original_language"].apply(parse_list)
    movie_db["is_european_wide"] = movie_db["country_list"].apply(lambda xs: any(x in WIDE_EUROPE for x in xs))
    movie_db["is_non_english"] = movie_db["language_list"].apply(lambda xs: bool(xs) and "English" not in xs)
    movie_db["priority_score"] = (
        movie_db["is_european_wide"].astype(int) * 4
        + movie_db["is_non_english"].astype(int) * 2
        + movie_db["rating_count"].fillna(0).rank(pct=True)
    )
    priority = (
        movie_db[movie_db["wikidata_uri"].notna()]
        .sort_values(["priority_score", "rating_count"], ascending=[False, False])
        .head(limit)
        .copy()
    )
    priority["qid"] = priority["wikidata_uri"].apply(qid_from_uri)
    qids = [qid for qid in priority["qid"].dropna().unique().tolist() if qid]

    if DNA_CACHE.exists():
        cache = pd.read_csv(DNA_CACHE)
    else:
        cache = pd.DataFrame(columns=[
            "wikidata_uri", "directors", "director_citizenship",
            "filming_location_country", "award_count", "award_examples",
        ])
    cached = set(cache["wikidata_uri"].dropna()) if len(cache) else set()
    wanted_uris = set(priority["wikidata_uri"].dropna())
    missing_qids = [qid for qid in qids if f"http://www.wikidata.org/entity/{qid}" not in cached]

    if missing_qids:
        new_rows = query_wikidata_dna(missing_qids)
        cache = pd.concat([cache, new_rows], ignore_index=True).drop_duplicates("wikidata_uri")
        cache.to_csv(DNA_CACHE, index=False)
    elif not DNA_CACHE.exists():
        cache.to_csv(DNA_CACHE, index=False)

    status = pd.DataFrame([
        {
            "source": "Wikidata deeper DNA cache",
            "status": "queried_or_loaded",
            "rows": len(cache),
            "priority_movies_requested": len(priority),
            "coverage_of_priority_movies": len(wanted_uris & set(cache["wikidata_uri"].dropna())) / max(len(wanted_uris), 1),
            "notes": "Director citizenship, filming-location country and award counts for a priority subset; not a full catalogue enrichment.",
        },
        {
            "source": "TMDb details/watch providers",
            "status": "not_queried_missing_api_key",
            "rows": 0,
            "priority_movies_requested": 0,
            "coverage_of_priority_movies": 0,
            "notes": "TMDb IDs exist locally, but TMDb API requires credentials. No scraping or synthetic provider data was used.",
        },
        {
            "source": "LUMIERE admissions",
            "status": "not_integrated_validation_layer",
            "rows": 0,
            "priority_movies_requested": 0,
            "coverage_of_priority_movies": 0,
            "notes": "Recommended as a later confident-match validation subset, not as blind bulk data.",
        },
    ])
    status.to_csv(DNA_STATUS, index=False)
    print(f"Wrote {DNA_CACHE} with {len(cache):,} rows")
    print(f"Wrote {DNA_STATUS}")


if __name__ == "__main__":
    main()
