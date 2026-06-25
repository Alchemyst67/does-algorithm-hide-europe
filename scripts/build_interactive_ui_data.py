from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
UI_DIR = PROJECT_ROOT / "ui"
UI_DIR.mkdir(exist_ok=True)


def read_csv(name: str):
    path = OUTPUTS / name
    if not path.exists():
        return []
    return pd.read_csv(path).replace({float("nan"): None}).to_dict(orient="records")


def read_json(name: str):
    path = OUTPUTS / name
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    exploration = read_json("dataset_exploration_summary.json")
    data = {
        "generatedAt": pd.Timestamp.now().isoformat(timespec="seconds"),
        "scope": {
            "title": "Does the Algorithm Hide Europe?",
            "subtitle": "A multimodal audit of cultural prominence bias in movie recommender systems",
            "team": "Max Priessnitz & Nico [surname]",
            "course": "Data Science and Artificial Intelligence II: Data and Algorithmic Governance",
            "run": "Full audit run: 5,000 real users, 12,012 candidate items, no synthetic fallback",
            "rawDataPolicy": "Raw MovieLens and M3L files are not redistributed; the UI uses derived outputs only.",
        },
        "datasetScale": [
            {"label": "M3L users", "value": 138_493, "display": "138,493"},
            {"label": "M3L items", "value": 19_009, "display": "19,009"},
            {"label": "Ratings", "value": 18_777_965, "display": "18.8M"},
            {"label": "Sparsity", "value": 99.29, "display": "99.29%"},
        ],
        "runReport": read_csv("12_full_run_report.csv"),
        "models": read_csv("12_full_model_comparison.csv"),
        "rerank": read_csv("12_full_rerank_tradeoff.csv"),
        "proxyRisk": read_csv("11_metadata_proxy_risk_table.csv"),
        "cvRunReport": read_csv("15_cv_run_report.csv"),
        "cvFoldResults": read_csv("15_cv_fold_results.csv"),
        "cvModelSummary": read_csv("15_cv_model_summary.csv"),
        "feedbackRunReport": read_csv("31_feedback_loop_run_report.csv"),
        "feedbackModelLedger": read_csv("31_schedl_model_ledger.csv"),
        "feedbackIterationMetrics": read_csv("31_feedback_loop_iteration_metrics.csv"),
        "feedbackFinalSummary": read_csv("31_feedback_loop_final_summary.csv"),
        "moviesDbSummary": read_csv("27_movies_db_summary_stats.csv"),
        "moviesDbCoverage": read_csv("27_movies_db_coverage_report.csv"),
        "moviesDbOverlap": read_csv("27_movies_db_movieid_overlap.csv"),
        "moviesDbUserConcentration": read_csv("27_movies_db_user_concentration.csv"),
        "moviesDbInventory": read_csv("27_movies_db_file_inventory.csv"),
        "dataSources": read_csv("data_source_ledger.csv"),
        "libraries": read_csv("library_ledger.csv"),
        "citations": read_csv("source_citation_ledger.csv"),
        "exploration": exploration,
        "assets": {
            "researchGap": "../outputs/00_research_gap_matrix.png",
            "datasetScale": "../outputs/01_dataset_scale.png",
            "pipeline": "../outputs/02_data_pipeline.png",
            "joinFunnel": "../outputs/03_join_funnel.png",
            "longTail": "../outputs/04_popularity_long_tail.png",
            "catalogueShare": "../outputs/05_catalogue_vs_interaction_share.png",
            "coverage": "../outputs/06_metadata_feature_coverage.png",
            "frontier": "../outputs/14_full_utility_vs_prominence_frontier.png",
            "groupExposure": "../outputs/12_full_group_exposure_by_model.png",
            "accuracyFairness": "../outputs/13_full_accuracy_fairness_summary.png",
            "workplan": "../outputs/10_workplan_gantt.png",
            "proxyRisk": "../outputs/11_language_country_company_caveats.png",
            "cvStability": "../outputs/15_cv_metric_stability.png",
            "moviesDbCoverage": "../outputs/27_movies_db_coverage.png",
            "moviesDbRatingDistribution": "../outputs/28_movies_db_rating_distribution.png",
            "moviesDbGenreVisibility": "../outputs/29_movies_db_genre_interest_visibility.png",
            "moviesDbUserConcentration": "../outputs/30_movies_db_user_concentration.png",
            "feedbackDynamics": "../outputs/31_feedback_loop_representation_dynamics.png",
            "feedbackJsd": "../outputs/32_feedback_loop_jsd_heatmap.png",
            "feedbackShift": "../outputs/33_feedback_loop_final_shift.png",
            "feedbackLanguageCountry": "../outputs/34_language_country_bias_panels.png",
            "feedbackNotebook": "../notebooks/schedl_feedback_loop_audit.ipynb",
            "moviesDbNotebook": "../notebooks/movies_db_pipeline.ipynb",
            "deck": "../outputs/does_algorithm_hide_europe_proposal_deck.pptx",
            "notebook": "../notebooks/does_algorithm_hide_europe_realdata.ipynb",
            "explorationNotes": "../outputs/dataset_exploration_source_notes.md",
        },
    }

    target = UI_DIR / "data.js"
    target.write_text(
        "window.AUDIT_DATA = " + json.dumps(data, indent=2, ensure_ascii=True) + ";\n",
        encoding="utf-8",
    )
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
