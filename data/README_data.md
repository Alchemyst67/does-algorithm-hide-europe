# Data Access and Raw-Data Policy


This project uses real data only. No synthetic fallback data is generated.


## Required Sources


- **M3L-20M / Binge Watch**: main MovieLens-based interaction data and multimodal features. Citation: Spillo et al. (2026), *Binge Watch: Reproducible Multimodal Benchmarks Datasets for Large-Scale Movie Recommendation on MovieLens-10M and 20M*, Zenodo, DOI: `10.5281/zenodo.18499145`, CC BY 4.0.

- **MovieLens 20M**: movie metadata, ratings and identifier bridge. Citation: GroupLens Research (2016), *MovieLens 20M Dataset README*, and Harper & Konstan (2015), *The MovieLens Datasets: History and Context*, DOI: `10.1145/2827872`.

- **Wikidata**: country of origin, original language, production company, director and additional visibility-DNA metadata via SPARQL. Citation: Wikidata contributors, *Wikidata:Data access*, and Wikimedia Foundation, *Wikidata Query Service/User Manual*. Wikidata structured data is CC0.

Full source details and BibTeX entries are provided in `../DATA_SOURCES.md` and `../references.bib`.


## Raw Data


Raw MovieLens and M3L files are not included in this submission package and should not be pushed to a public repository. Download them from the original providers and place them in `data/raw/` or in the expected extracted local folders.


## Source Links


- M3L-20M / Binge Watch Zenodo record: <https://zenodo.org/records/18499145>

- MovieLens 20M README: <https://files.grouplens.org/datasets/movielens/ml-20m-README.html>

- MovieLens dataset paper DOI: <https://doi.org/10.1145/2827872>

- Wikidata data access: <https://www.wikidata.org/wiki/Wikidata:Data_access>

- Wikidata Query Service manual: <https://www.mediawiki.org/wiki/Wikidata_Query_Service/User_Manual>


## Processed Data Included in This Repository


The folder `data/processed/` contains derived CSV tables from the executed notebooks. These tables are sufficient to review the reported results, reproduce the presentation figures, and inspect the country/language/model audit outputs without redistributing the raw MovieLens or M3L archives.

The processed tables include model metrics, country and language visibility metrics, metadata coverage summaries, user-fold robustness summaries, feedback-loop outputs, re-ranking frontiers and the final research-question answer table.

The `*_from_notebook.csv` files are small diagnostics written by `notebooks/01_data_foundation_movies_db.ipynb`. They document source inventory, M3L bridge checks, Wikidata join coverage, feature coverage and audit-label counts without redistributing raw MovieLens or M3L files.


## Wikidata Cache


Wikidata is CC0. Query results are cached under `data/interim/` during local reproduction. The submitted derived outputs document query-based metadata coverage and missingness.
