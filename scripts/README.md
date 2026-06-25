# Supporting Python Scripts

The notebooks are the main deliverable and contain the full research story. These scripts make the most important data-foundation steps reproducible outside Jupyter, so the repository does not depend on hidden preprocessing.

## Script Order

Run the scripts from the repository root after placing the raw data locally:

```bash
python scripts/build_m3l_movielens_bridge.py --project-root .
python scripts/query_wikidata_metadata.py --project-root .
python scripts/build_visibility_dna_enrichment.py --project-root .
```

## What Each Script Does

- `build_m3l_movielens_bridge.py` reconstructs the M3L internal item id to MovieLens `movieId` bridge by matching MPNet vector fingerprints. This is needed because recommender matrices and MovieLens metadata use different item identifiers.
- `query_wikidata_metadata.py` queries Wikidata by MovieLens IMDb ids and caches country, language, publication date and production-company fields in `data/interim/wikidata_movie_metadata_extended.csv`.
- `build_visibility_dna_enrichment.py` optionally adds deeper Wikidata fields for interpretation, such as director citizenship, filming-location country and award signals. This is a bounded interpretive layer, not a replacement for the main audit metrics.

## Raw Data Policy

The scripts do not download or redistribute MovieLens or M3L files. Raw files must be obtained from the original providers listed in `DATA_SOURCES.md` and `data/README_data.md`.

## Why These Scripts Are Included

The project is notebook-first, but the source bridge and Wikidata enrichment are central enough that they should be readable and runnable as plain Python. The derived CSVs in `data/processed/` remain review artifacts; the scripts document how the local enrichment caches are rebuilt from the primary sources.
