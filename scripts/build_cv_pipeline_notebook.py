from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "recommender_pipeline_cross_validation.ipynb"


def main() -> None:
    cells = [
        new_markdown_cell(
            """# Recommender Pipeline and Cross-Validation
## Does the Algorithm Hide Europe?

**Team:** Max Priessnitz & Nico [surname]  
**Purpose:** focused modelling notebook for the cultural-prominence audit  

This notebook documents the actual recommendation pipeline and the cross-validation robustness check. It is intentionally narrower than the proposal notebook: it focuses on model implementation, validation design, model comparison and interpretation.
"""
        ),
        new_markdown_cell(
            """## 1. Modelling Objective

The modelling task is to produce Top-20 movie recommendations from real M3L/MovieLens interaction histories and then audit whether those rankings make European, non-English and long-tail films visible. The models are not evaluated only by accuracy. They are evaluated by a combined governance frame:

- **Utility:** NDCG@20, Recall@20 and MAP@20.
- **Catalogue use:** Coverage@20.
- **Cultural prominence:** discounted group exposure and PACPG.
- **Robustness:** user-fold cross-validation across non-overlapping user subsets.

The cross-validation step uses the existing M3L train/test split inside each user fold. This preserves the recommender setting: each fold has real user histories and real held-out test items.
"""
        ),
        new_code_cell(
            """from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
from IPython.display import Image, Markdown, display

PROJECT_ROOT = Path("..").resolve()
OUTPUTS = PROJECT_ROOT / "outputs"
SCRIPT = PROJECT_ROOT / "scripts" / "run_recommender_cross_validation.py"

TOP_K = 20
RUN_CV = False  # Set to True to rerun the full fold pipeline from this notebook.
"""
        ),
        new_markdown_cell(
            """## 2. Reproducible Pipeline Contract

The implementation lives in `scripts/run_recommender_cross_validation.py`. Keeping the heavy model code in a script gives us a reliable rerun path, while this notebook keeps the reasoning, outputs and interpretation readable.

The script performs these steps per fold:

1. Load real M3L interactions and cached processed movie metadata.
2. Select eligible users with at least five train interactions and at least one test interaction.
3. Split sampled users into three non-overlapping folds.
4. Build a fold-specific sparse user-item matrix and candidate universe.
5. Fit/recompute Popularity, ItemKNN, SVD, MPNet-content, CLIP-image-content and Hybrid scores.
6. Apply governance-aware re-ranking to the Hybrid candidate list.
7. Evaluate utility, coverage, exposure and PACPG.

No synthetic fallback is used.
"""
        ),
        new_code_cell(
            """if RUN_CV:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    print(completed.stdout[-4000:])
else:
    print("Using cached cross-validation outputs. Set RUN_CV=True to rerun the fold pipeline.")
"""
        ),
        new_markdown_cell(
            """## 3. Cross-Validation Protocol

This is a **user-fold robustness check**, not a random interaction split. A random interaction split would mix user histories in a way that is less faithful to recommendation. Instead, the fold unit is the user: each fold rebuilds the model pipeline on a different set of real users and evaluates against their M3L test items.
"""
        ),
        new_code_cell(
            """cv_run_report = pd.read_csv(OUTPUTS / "15_cv_run_report.csv")
cv_fold_results = pd.read_csv(OUTPUTS / "15_cv_fold_results.csv")
cv_model_summary = pd.read_csv(OUTPUTS / "15_cv_model_summary.csv")

display(cv_run_report)
"""
        ),
        new_markdown_cell(
            """## 4. Model Ledger

| Model | Implementation | Role in the audit |
|---|---|---|
| Popularity | Global log-popularity score from training interactions. | Mainstream-concentration baseline. |
| ItemKNN | Cosine similarity over the item-user matrix, retaining strongest neighbours. | Transparent collaborative filtering baseline. |
| SVD | TruncatedSVD over the sparse user-item matrix. | Lightweight latent-factor baseline. |
| MPNet-content | User profile from mean text embeddings of train items. | Text-aware content recommender. |
| CLIP-image-content | User profile from mean poster embeddings of train items. | Visual-content recommender. |
| Hybrid | 0.50 SVD + 0.25 MPNet + 0.25 CLIP after row-wise standardisation. | Main multimodal model. |
| Hybrid + re-ranking | Post-processes the Hybrid Top-100 candidate pool with a cultural-prominence bonus. | Governance intervention with explicit lambda. |
"""
        ),
        new_code_cell(
            """cv_model_summary["Absolute PACPG mean"] = (
    cv_model_summary["PACPG European mean"].abs()
    + cv_model_summary["PACPG Non-English mean"].abs()
    + cv_model_summary["PACPG Long-tail mean"].abs()
)
cv_model_summary["Absolute PACPG std proxy"] = (
    cv_model_summary["PACPG European std"].fillna(0).abs()
    + cv_model_summary["PACPG Non-English std"].fillna(0).abs()
    + cv_model_summary["PACPG Long-tail std"].fillna(0).abs()
)

summary_cols = [
    "Model",
    f"NDCG@{TOP_K} mean",
    f"NDCG@{TOP_K} std",
    f"Recall@{TOP_K} mean",
    f"MAP@{TOP_K} mean",
    f"Coverage@{TOP_K} mean",
    "European Exposure@20 mean",
    "Non-English Exposure@20 mean",
    "Long-tail Exposure@20 mean",
    "Absolute PACPG mean",
]
display(cv_model_summary[summary_cols].round(4))
"""
        ),
        new_code_cell(
            """best_utility = cv_model_summary.loc[cv_model_summary[f"NDCG@{TOP_K} mean"].idxmax()]
best_prominence = cv_model_summary.loc[cv_model_summary["Absolute PACPG mean"].idxmin()]
stable_utility = cv_model_summary.loc[cv_model_summary[f"NDCG@{TOP_K} std"].idxmin()]

display(Markdown(
    f"**Interpretation.** The strongest mean utility in the cross-validation run is **{best_utility['Model']}** "
    f"with mean NDCG@{TOP_K}={best_utility[f'NDCG@{TOP_K} mean']:.4f}. "
    f"The lowest mean absolute PACPG is **{best_prominence['Model']}** "
    f"({best_prominence['Absolute PACPG mean']:.4f}). "
    f"The most stable utility by fold standard deviation is **{stable_utility['Model']}**. "
    "The result should be read together with the full-run test evaluation, because cross-validation tests robustness across user subsets while the full run uses the larger bounded audit sample."
))
"""
        ),
        new_markdown_cell(
            """## 5. Stability Visual

The error bars below show fold-level variation. For proposal purposes, this is useful because it prevents us from overclaiming from one convenient sample.
"""
        ),
        new_code_cell(
            """display(Image(filename=str(OUTPUTS / "15_cv_metric_stability.png")))
"""
        ),
        new_markdown_cell(
            """## 6. Fold-Level Audit Table

The fold table is deliberately kept visible. If a model looks strong on average but unstable across folds, the final write-up should treat it more cautiously.
"""
        ),
        new_code_cell(
            """fold_cols = [
    "Fold",
    "Model",
    f"NDCG@{TOP_K}",
    f"Recall@{TOP_K}",
    f"Coverage@{TOP_K}",
    "European Exposure@20",
    "Non-English Exposure@20",
    "PACPG European",
    "PACPG Non-English",
    "PACPG Long-tail",
]
display(cv_fold_results[fold_cols].round(4))
"""
        ),
        new_markdown_cell(
            """## 7. What This Adds to the Proposal

The main proposal already demonstrates an end-to-end audit. This notebook adds model validation discipline:

- the models are rebuilt per fold instead of being treated as static outputs;
- the evaluation includes both recommender utility and cultural-prominence metrics;
- re-ranking is validated as an explicit post-processing intervention;
- the interpretation separates robust patterns from fold-sensitive patterns.

For the final project, the next step is to keep the same pipeline but add stratified analyses by genre, user activity segment and metadata coverage group.
"""
        ),
    ]

    nb = new_notebook(
        cells=cells,
        metadata={
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
    )
    NOTEBOOK_PATH.parent.mkdir(exist_ok=True)
    nbformat.write(nb, NOTEBOOK_PATH)
    print(f"Wrote {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
