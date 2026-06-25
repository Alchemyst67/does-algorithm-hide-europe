# Data Sources


This project uses real datasets only. Raw MovieLens and M3L/Binge Watch files are not redistributed in this public repository. The repository includes derived tables and figures so that the submitted analysis can be reviewed without republishing the original raw archives.


## Source Ledger


| Source | Role in this project | Files or fields used | Citation | Licence / terms | Accessed |
|---|---|---|---|---|---|
| M3L-20M / Binge Watch | Main recommender interaction data and multimodal features | M3L-20M interactions, MPNet plot embeddings, CLIP poster embeddings, M3L item bridge | Spillo et al. (2026), Zenodo, DOI: `10.5281/zenodo.18499145` | CC BY 4.0 according to Zenodo | 25 June 2026 |
| MovieLens 20M | Movie catalogue metadata and identifier bridge | `movies.csv`, `ratings.csv`, `links.csv`, `tags.csv`, `genome_scores.csv`, `genome_tags.csv` | GroupLens Research (2016); Harper & Konstan (2015), DOI: `10.1145/2827872` | Research use under GroupLens terms; raw data not redistributed | 25 June 2026 |
| Wikidata | Country, language, production-company, director and visibility-DNA metadata | SPARQL results cached locally during reproduction | Wikidata contributors (2026); Wikimedia Foundation (2026) | Structured Wikidata data is CC0 | 25 June 2026 |


## Source Notes


### M3L-20M / Binge Watch

M3L-20M is used as the project’s main recommender dataset because it extends MovieLens with multimodal features. The Zenodo record describes M3L-10M and M3L-20M as multimodal extensions of MovieLens-10M and MovieLens-20M with textual, visual, acoustic and video features extracted from movie plots, posters and trailers. For this audit, we use M3L-20M interactions plus MPNet text and CLIP-image features.

Formal citation:

Spillo, G., Petruzzelli, A., Musto, C., de Gemmis, M., Lops, P., & Semeraro, G. (2026). *Binge Watch: Reproducible Multimodal Benchmarks Datasets for Large-Scale Movie Recommendation on MovieLens-10M and 20M* (Version v1) [Dataset]. Zenodo. https://doi.org/10.5281/zenodo.18499145


### MovieLens 20M

MovieLens 20M is used for catalogue metadata, ratings, tags, genome information and the IMDb/TMDb identifier bridge. The official GroupLens README reports 20,000,263 ratings, 465,564 tag applications, 27,278 movies and 138,493 users, and lists the core files used here. MovieLens terms require acknowledgement, prohibit redistribution without separate permission and restrict commercial use without permission.

Formal citations:

GroupLens Research. (2016). *MovieLens 20M Dataset README*. https://files.grouplens.org/datasets/movielens/ml-20m-README.html

Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19. https://doi.org/10.1145/2827872


### Wikidata and Wikidata Query Service

Wikidata is used to enrich movies with cultural metadata that MovieLens does not provide directly: country of origin / production country, original language, production company, director and related visibility-DNA fields. Queries are made through the Wikidata Query Service SPARQL endpoint and cached locally during reproduction. Wikidata structured data is published under CC0.

Formal citations:

Wikidata contributors. (2026). *Wikidata:Data access*. https://www.wikidata.org/wiki/Wikidata:Data_access

Wikimedia Foundation. (2026). *Wikidata Query Service/User Manual*. https://www.mediawiki.org/wiki/Wikidata_Query_Service/User_Manual


## Raw-Data Policy


- Raw MovieLens and M3L files are not committed to this repository.
- Derived CSV tables and figures are included for review and presentation.
- Wikidata query results may be cached locally during reproduction because Wikidata structured data is CC0.
- Every empirical claim in the notebooks should be read together with metadata coverage and missingness diagnostics.
