# Data Access and Raw-Data Policy


This project uses real data only. No synthetic fallback data is generated.


## Required Sources


- **M3L-20M / Binge Watch**: main MovieLens-based interaction data and multimodal features.

- **MovieLens 20M**: movie metadata, ratings and identifier bridge.

- **Wikidata**: country of origin, original language, production company, director and additional visibility-DNA metadata via SPARQL.


## Raw Data


Raw MovieLens and M3L files are not included in this submission package and should not be pushed to a public repository. Download them from the original providers and place them in `data/raw/` or in the expected extracted local folders.


## Wikidata Cache


Wikidata is CC0. The scripts cache query results under `data/interim/` during reproduction. The submitted derived outputs document query-based metadata coverage and missingness.
