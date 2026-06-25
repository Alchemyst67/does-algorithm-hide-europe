# Does the Algorithm Hide Europe?


**A multimodal audit of cultural prominence bias in movie recommender systems**

Team: Max Priessnitz & Nico Zandomeneghi

Course: Data Science and Artificial Intelligence II: Data and Algorithmic Governance, WU Vienna, 2026


## Objective


This project asks whether movie recommender systems trained on real interaction data under-expose European, non-English and long-tail films in Top-K recommendations, and whether multimodal features or transparent re-ranking can reduce this visibility gap without destroying recommendation utility.


The central argument is: **cultural diversity is not only a catalogue question; it is a ranking visibility problem.**


## Repository Architecture


- `notebooks/01_data_foundation_movies_db.ipynb` — MovieLens/M3L/Wikidata data foundation and coverage checks.

- `notebooks/02_model_pipeline_and_user_fold_robustness.ipynb` — model pipeline and bounded user-fold robustness.

- `notebooks/03_feedback_loop_and_mitigation.ipynb` — feedback-loop stress test and mitigation logic.

- `notebooks/04_final_research_story_executed.ipynb` — main research-story notebook with executed outputs and interpretations.

- `notebooks/05_research_answers_and_roadmap.ipynb` — direct answers to the research questions and evidence map.

- `html/` — HTML exports of the notebooks with saved output state.

- `cultural_prominence_audit/outputs/` — derived tables, figures and final-submission assets.

- `data/processed/` — derived CSV tables used in the notebooks, dashboard and presentation.

- `ui/` — interactive roadmap/dashboard layer for presentation and review.


- `data/README_data.md` — dataset links, access instructions and raw-data redistribution policy.


## Required Data


Raw MovieLens and M3L/Binge Watch files are not redistributed. To reproduce from scratch, place the required local files under `data/raw/` or in the extracted local folders described in `data/README_data.md`:


- M3L-20M / Binge Watch interaction and multimodal features

- MovieLens 20M metadata and ratings

- Wikidata enrichment via SPARQL, cached in `data/interim/`


See `DATA_SOURCES.md`, `data/README_data.md` and `references.bib` for exact source citations, links, access notes and licence constraints.


## Data Citations


The empirical analysis is based on three source layers:

- **M3L-20M / Binge Watch** provides the main recommender-system interaction data and the MPNet / CLIP multimodal feature basis. It is a Zenodo dataset by Spillo, Petruzzelli, Musto, de Gemmis, Lops and Semeraro (2026), DOI: `10.5281/zenodo.18499145`, licensed as CC BY 4.0.

- **MovieLens 20M** provides movie titles, genres, ratings, tags, genome tags and the `links.csv` identifier bridge. The official GroupLens README documents the dataset contents and terms; dataset use should cite Harper and Konstan (2015), DOI: `10.1145/2827872`.

- **Wikidata and the Wikidata Query Service** provide country, language, production-company, director and related metadata through SPARQL queries. Wikidata structured data is CC0; query-service use is documented by Wikimedia.

The full citation and reuse details are kept in `DATA_SOURCES.md`, `CITATION_AND_REUSE_LEDGER.md` and `references.bib`.


## Setup


```bash

uv venv --python 3.12 .venv

uv pip install --python .venv/bin/python -r requirements.txt

jupyter lab

```


If `uv` is not available, use a Python 3.12 virtual environment and install the same `requirements.txt`.


## Reproduction Order


This repository is notebook-first. The executed notebooks and HTML exports contain the analysis path, saved outputs and interpretations. The recommended reading order is:

1. `notebooks/01_data_foundation_movies_db.ipynb`
2. `notebooks/02_model_pipeline_and_user_fold_robustness.ipynb`
3. `notebooks/03_feedback_loop_and_mitigation.ipynb`
4. `notebooks/04_final_research_story_executed.ipynb`
5. `notebooks/05_research_answers_and_roadmap.ipynb`

The main notebook can be re-executed after placing the raw datasets locally:


```bash

jupyter nbconvert --to notebook --execute notebooks/04_final_research_story_executed.ipynb --output rerun_final_story.ipynb --ExecutePreprocessor.timeout=1200

```


## Models


We compare six recommender families: Popularity, ItemKNN, TruncatedSVD collaborative filtering, MPNet text-content recommendation, CLIP-image-content recommendation and a simple Hybrid. A transparent governance-aware re-ranker is then applied as a post-processing mitigation layer.


## Metrics


Utility is measured with NDCG@20, Recall@20, MAP@20 and catalogue coverage. Cultural prominence is measured through discounted Top-K group exposure and PACPG: Preference-Adjusted Cultural Prominence Gap. PACPG compares ranked exposure against transparent targets based on history and relevant test data.


## Main Results


- Popularity is the strongest underexposure baseline for Europe in this run.

- SVD is the best utility model by NDCG@20.

- CLIP-image-content creates the highest Europe-wide exposure, but with low utility.

- English-language exposure is dominant across model outputs.

- France shows the clearest support-passing negative country-level gap; Spain and Germany are also important diagnostic cases.

- The Visibility DNA extension suggests globally compatible Europe is much more visible than stricter local-Europe proxies.

- Re-ranking improves cultural prominence only with a measurable utility trade-off.


## State of the Art and Positioning


The project builds on popularity-bias and long-tail exposure research, provider/item-side fairness, multimodal movie recommendation and recommender feedback-loop work. The gap is that cultural prominence is operationalised as a ranking-layer governance audit with country/language proxy labels rather than only as catalogue share or accuracy.


Key sources include Klimashevskaia et al. (2023), Abdollahpouri et al. (2019), Wang/Jin fairness-survey work, Spillo et al. on M3L/Binge Watch, Lesota/Schedl feedback-loop work, MovieLens documentation, Wikidata documentation and EU AVMSD/DSA governance materials. See `CITATION_AND_REUSE_LEDGER.md` and notebook references.


## Tooling Note


OpenAI Codex/ChatGPT was used as a supporting tool for selected wording, code-review checks and small refactoring suggestions. All empirical claims are tied to executed outputs or cited sources. No synthetic data was generated. Details are in `AI_USE_DISCLOSURE.md`.


## Licence and Raw-Data Policy


This repository does not redistribute raw MovieLens or M3L data. Users must download those datasets from their original providers and respect their terms. Derived figures and CSV tables are included for review of the submitted analysis.

## Presentation Material

Presentation design files are handled separately. This repository remains the reproducible analysis package with notebooks, HTML exports, derived tables, figures and documentation.
