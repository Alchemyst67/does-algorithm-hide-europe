# Interactive Audit UI

This folder contains a local web interface for the project **Does the Algorithm Hide Europe?**

The UI is a documentation and exploration layer on top of the executed notebook outputs. It does not read or redistribute raw MovieLens or M3L files.

There are two local pages:

- `index.html` is the broader interactive audit dashboard.
- `roadmap.html` is the new finding-driven presentation roadmap. It explains each step through the question answered, the technique used, why that technique was chosen, the outcome, and the evidence artifact.

## What It Shows

- Full-run scope and dataset scale
- The upgraded "Which Europe Gets Recommended?" research storyline
- A visibility funnel from catalogue access to Top-K and feedback-loop exposure
- Movies DB inspection and coverage diagnostics
- Dataset-foundation evidence from the separate exploration notebook
- Model comparison across utility, coverage and cultural exposure metrics
- Schedl-style feedback-loop dynamics for country, language and popularity drift
- Re-ranking lambda frontier
- Proxy-risk checks for language, production country and US-company involvement
- Notebook figure gallery
- Research question, methodology and reproducibility notes
- Expected technical and governance outcome
- Source and citation ledger

The roadmap page additionally shows:

- a seven-step audit roadmap from governance framing to mitigation,
- an eighth research-extension step for European Film Visibility DNA,
- country-risk findings with scorecard, target-vs-exposure plot and approximate geo map,
- a compact model metric explorer,
- final supported research-question answers,
- technique-choice cards for methods, metrics and governance framing.

## Data Inputs

The app uses `ui/data.js`, generated from:

- `outputs/12_full_model_comparison.csv`
- `outputs/12_full_rerank_tradeoff.csv`
- `outputs/12_full_run_report.csv`
- `outputs/31_feedback_loop_iteration_metrics.csv`
- `outputs/31_feedback_loop_final_summary.csv`
- `outputs/31_schedl_model_ledger.csv`
- `outputs/11_metadata_proxy_risk_table.csv`
- `outputs/27_movies_db_summary_stats.csv`
- `outputs/27_movies_db_coverage_report.csv`
- `outputs/27_movies_db_movieid_overlap.csv`
- `outputs/source_citation_ledger.csv`
- `outputs/data_source_ledger.csv`
- `outputs/dataset_exploration_summary.json`
- `outputs/dataset_exploration_figures.csv`
- selected PNG figures in `outputs/`

The dataset-foundation section is extracted from:

- `/Users/maxpriessnitz/Downloads/dataset_exploration_and_combination.ipynb`

Regenerate those derived exploration outputs before rebuilding the UI data:

```bash
.venv/bin/python scripts/build_movies_db.py
.venv/bin/python scripts/extract_dataset_exploration_for_ui.py
```

Regenerate the embedded UI data after rerunning the audit:

```bash
.venv/bin/python scripts/build_movies_db.py
.venv/bin/python scripts/extract_dataset_exploration_for_ui.py
.venv/bin/python scripts/run_schedl_style_feedback_loop.py
.venv/bin/python scripts/build_interactive_ui_data.py
```

The feedback-loop panel follows Lesota, Geiger, Walder, Kowald and Schedl (2024) in spirit: recommend, simulate one accepted item, append it to the user profile, retrain and repeat. MovieLens/M3L has no user-country field, so the UI phrases this as user-history calibration rather than literal local-country representation.

The storyline section is intentionally strict about evidence. It distinguishes supported outputs from partial or not-yet-claimed analyses, especially for granular country/region exposure, the Spain case study and religion/theme exploration.

## Run Locally

From the project root:

```bash
python3 -m http.server 8010 --bind 127.0.0.1
```

Open:

```text
http://127.0.0.1:8010/ui/
```

Open the roadmap directly:

```text
http://127.0.0.1:8010/ui/roadmap.html
```

## Governance Note

The UI is suitable for presentation and documentation because it only uses derived outputs. Raw data folders and archives must stay local unless the course explicitly permits local-only sharing.
