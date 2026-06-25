from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "does_algorithm_hide_europe_realdata.ipynb"


def remove_existing_cv_section(cells):
    start = None
    end = None
    for idx, cell in enumerate(cells):
        if cell.cell_type == "markdown" and "<!-- cv-pipeline-section -->" in cell.source:
            start = idx
            continue
        if start is not None and idx > start and cell.cell_type == "markdown" and "<!-- targeted-next-steps -->" in cell.source:
            end = idx
            break
    if start is not None and end is not None:
        del cells[start:end]


def renumber_tail_sections(cells):
    replacements = {
        "## 16. Interpretation and Next Steps": "## 17. Interpretation and Next Steps",
        "## 17. Workplan": "## 18. Workplan",
        "## 18. Proposal Summary": "## 19. Proposal Summary",
        "## 19. References and Library Ledger": "## 20. References and Library Ledger",
        "## 20. Export Outputs": "## 21. Export Outputs",
    }
    for cell in cells:
        if cell.cell_type != "markdown":
            continue
        for old, new in replacements.items():
            cell.source = cell.source.replace(old, new)


def add_manifest_entries(cells):
    extra_outputs = [
        '"15_cv_fold_results.csv"',
        '"15_cv_model_summary.csv"',
        '"15_cv_model_summary_long.csv"',
        '"15_cv_run_report.csv"',
        '"15_cv_metric_stability.png"',
    ]
    for cell in cells:
        if cell.cell_type == "code" and "outputs_manifest = [" in cell.source:
            for entry in extra_outputs:
                if entry not in cell.source:
                    cell.source = cell.source.replace(
                        '    "source_citation_ledger.csv",',
                        f'    {entry},\n    "source_citation_ledger.csv",',
                    )
            break


def cv_cells():
    return [
        new_markdown_cell(
            """<!-- cv-pipeline-section -->
## 16. Cross-Validation and Pipeline Robustness

The modelling layer is now treated as a reproducible notebook pipeline rather than a single one-off run. The full analysis still uses the official M3L train/test labels, but we add a **user-fold robustness check** to test whether the model ranking is stable across different user subsets.

This is not a random interaction split: for each fold, we keep the user histories and M3L test items intact. The fold therefore asks a practical audit question: if we repeat the pipeline for different groups of real users, do the utility and cultural-prominence conclusions remain broadly consistent?

| Pipeline step | What happens | Why it matters |
|---|---|---|
| User fold selection | Eligible users are split into non-overlapping folds. | Tests robustness across users rather than a single convenient sample. |
| Candidate universe | Each fold combines popular items, fold train items and fold test items. | Keeps evaluation feasible while preserving relevant test items. |
| Model fitting | Popularity, ItemKNN, SVD, MPNet, CLIP, Hybrid and Hybrid re-rankers are rebuilt per fold. | Avoids reusing scores from the main run. |
| Evaluation | Each fold reports NDCG@20, Recall@20, MAP@20, Coverage@20, exposure and PACPG. | Aligns cross-validation with the governance metrics, not only accuracy. |
| Interpretation | Mean and fold standard deviation are reported. | Separates robust patterns from sample-sensitive ones. |
"""
        ),
        new_code_cell(
            """cv_run_report = pd.read_csv(OUTPUTS / "15_cv_run_report.csv")
cv_fold_results = pd.read_csv(OUTPUTS / "15_cv_fold_results.csv")
cv_model_summary = pd.read_csv(OUTPUTS / "15_cv_model_summary.csv")

display(cv_run_report)

cv_model_summary["Absolute PACPG mean"] = (
    cv_model_summary["PACPG European mean"].abs()
    + cv_model_summary["PACPG Non-English mean"].abs()
    + cv_model_summary["PACPG Long-tail mean"].abs()
)
cv_model_summary["Absolute PACPG std proxy"] = (
    cv_model_summary["PACPG European std"].fillna(0).abs()
    + cv_model_summary["PACPG Non-English std"].fillna(0).abs()
    + cv_model_summary["PACPG Long-tail std"].fillna(0).abs()
)

cv_view_cols = [
    "Model",
    f"NDCG@{TOP_K} mean",
    f"NDCG@{TOP_K} std",
    f"Recall@{TOP_K} mean",
    f"Coverage@{TOP_K} mean",
    "European Exposure@20 mean",
    "Non-English Exposure@20 mean",
    "Long-tail Exposure@20 mean",
    "Absolute PACPG mean",
    "Absolute PACPG std proxy",
]
display(cv_model_summary[cv_view_cols].round(4))
"""
        ),
        new_code_cell(
            """best_cv_utility = cv_model_summary.loc[cv_model_summary[f"NDCG@{TOP_K} mean"].idxmax()]
best_cv_prominence = cv_model_summary.loc[cv_model_summary["Absolute PACPG mean"].idxmin()]
most_stable_utility = cv_model_summary.loc[cv_model_summary[f"NDCG@{TOP_K} std"].idxmin()]

display(Markdown(
    f"**Cross-validation interpretation.** Across user folds, **{best_cv_utility['Model']}** has the highest mean "
    f"NDCG@{TOP_K} ({best_cv_utility[f'NDCG@{TOP_K} mean']:.4f}, fold std={best_cv_utility[f'NDCG@{TOP_K} std']:.4f}). "
    f"The lowest mean absolute PACPG is achieved by **{best_cv_prominence['Model']}** "
    f"({best_cv_prominence['Absolute PACPG mean']:.4f}). "
    f"The most stable utility score by fold standard deviation is **{most_stable_utility['Model']}**. "
    "We use this as a robustness check, not as a replacement for the full-run test set evaluation."
))
"""
        ),
        new_code_cell(
            """display(Image(filename=str(OUTPUTS / "15_cv_metric_stability.png")))

display(Markdown(
    "**Figure interpretation.** The left panel checks whether ranking utility changes materially across folds; "
    "the right panel checks whether the combined PACPG signal is fold-sensitive. A model is more convincing "
    "when it performs well and has narrow fold variation."
))
"""
        ),
        new_code_cell(
            """fold_metric_view = cv_fold_results[[
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
]].copy()

display(fold_metric_view.round(4))

display(Markdown(
    "**Fold-level interpretation.** This table is intentionally more detailed than the summary above. "
    "It lets us see whether a strong mean is driven by all folds or by a single favourable user subset."
))
"""
        ),
        new_markdown_cell(
            """### How to rerun the robustness check

The cross-validation outputs are generated by:

```bash
.venv/bin/python scripts/run_recommender_cross_validation.py
```

The script uses real M3L interactions and cached processed metadata. It does not query Wikidata and does not create synthetic data. The main settings are deliberately visible in the script: number of folds, users per fold, candidate universe size, Top-K, ItemKNN neighbourhood size, SVD components and re-ranking lambdas.
"""
        ),
    ]


def main():
    nb = nbformat.read(NOTEBOOK_PATH, as_version=4)
    remove_existing_cv_section(nb.cells)
    renumber_tail_sections(nb.cells)
    add_manifest_entries(nb.cells)

    insert_at = None
    for idx, cell in enumerate(nb.cells):
        if cell.cell_type == "markdown" and "<!-- targeted-next-steps -->" in cell.source:
            insert_at = idx
            break
    if insert_at is None:
        raise RuntimeError("Could not find targeted-next-steps marker.")

    for offset, cell in enumerate(cv_cells()):
        nb.cells.insert(insert_at + offset, cell)

    nbformat.write(nb, NOTEBOOK_PATH)
    print(f"Updated {NOTEBOOK_PATH}")


if __name__ == "__main__":
    main()
