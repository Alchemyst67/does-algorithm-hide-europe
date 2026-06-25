# Data Access and Raw-Data Policy


This project uses real data only. No synthetic fallback data is generated.


## Required Sources


- **M3L-20M / Binge Watch**: main MovieLens-based interaction data and multimodal features.

- **MovieLens 20M**: movie metadata, ratings and identifier bridge.

- **Wikidata**: country of origin, original language, production company, director and additional visibility-DNA metadata via SPARQL.


## Raw Data


Raw MovieLens and M3L files are not included in this submission package and should not be pushed to a public repository. Download them from the original providers and place them in `data/raw/` or in the expected extracted local folders.


## Processed Data Included in This Repository


The folder `data/processed/` contains derived CSV tables from the executed notebooks. These tables are sufficient to review the reported results, reproduce the presentation figures, and inspect the country/language/model audit outputs without redistributing the raw MovieLens or M3L archives.

The processed tables include model metrics, country and language visibility metrics, metadata coverage summaries, user-fold robustness summaries, feedback-loop outputs, re-ranking frontiers and the final research-question answer table.


## Wikidata Cache


Wikidata is CC0. Query results are cached under `data/interim/` during local reproduction. The submitted derived outputs document query-based metadata coverage and missingness.
