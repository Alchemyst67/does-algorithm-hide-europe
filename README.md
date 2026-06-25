# Does the Algorithm Hide Europe?


**A multimodal audit of cultural prominence bias in movie recommender systems**

Team: Max Priessnitz & Nico [surname]

Course: Data Science and Artificial Intelligence II: Data and Algorithmic Governance, WU Vienna, 2026


## Objective


This project asks whether movie recommender systems trained on real interaction data under-expose European, non-English and long-tail films in Top-K recommendations, and whether multimodal features or transparent re-ranking can reduce this visibility gap without destroying recommendation utility.


The central thesis is: **cultural diversity is not only a catalogue question; it is a ranking visibility problem.**


## Repository Architecture


- `notebooks/00_project_roadmap_and_final_answers.ipynb` — entry notebook with final answers and reading order.

- `notebooks/01_data_foundation_movies_db.ipynb` — MovieLens/M3L/Wikidata data foundation and coverage checks.

- `notebooks/02_model_pipeline_and_user_fold_robustness.ipynb` — model pipeline and bounded user-fold robustness.

- `notebooks/03_feedback_loop_and_mitigation.ipynb` — feedback-loop stress test and mitigation logic.

- `notebooks/04_final_research_story_executed.ipynb` — main thesis-like notebook with executed outputs and interpretations.

- `html/` — HTML exports of the notebooks with saved output state.

- `scripts/` — reproducible pipeline scripts.

- `cultural_prominence_audit/outputs/` — derived tables, figures and final-submission assets.

- `ui/` — interactive roadmap/dashboard layer for presentation and review.


- `data/README_data.md` — dataset links, access instructions and raw-data redistribution policy.


## Required Data


Raw MovieLens and M3L/Binge Watch files are not redistributed. To reproduce from scratch, place the required local files under `data/raw/` or use the extracted folders expected by the scripts:


- M3L-20M / Binge Watch interaction and multimodal features

- MovieLens 20M metadata and ratings

- Wikidata enrichment via SPARQL, cached in `data/interim/`


See `data/README_data.md` for exact links and commands.


## Setup


```bash

uv venv --python 3.12 .venv

uv pip install --python .venv/bin/python -r requirements.txt

jupyter lab

```


If `uv` is not available, use a Python 3.12 virtual environment and install the same `requirements.txt`.


## Reproduction Order


```bash

python scripts/build_movies_db.py

python scripts/run_full_cultural_prominence_audit.py

python scripts/run_recommender_cross_validation.py

python scripts/run_schedl_style_feedback_loop.py

python scripts/build_visibility_dna_enrichment.py

python scripts/build_final_submission_assets.py

```


The main notebook can then be executed with:


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


## AI Use


Generative AI was used for code scaffolding, notebook structuring, documentation wording, visualization/UI drafting and presentation drafting. All research claims are tied to executed outputs or cited sources. No synthetic data was generated. Details are in `AI_USE_DISCLOSURE.md`.


## Licence and Raw-Data Policy


This repository does not redistribute raw MovieLens or M3L data. Users must download those datasets from their original providers and respect their terms. Derived figures/tables are included for review of the submitted analysis.

## Presentation Material

Presentation files are intentionally not included in this GitHub repository. The final slide design will be handled separately, while this repository remains the reproducible analysis/code package.
