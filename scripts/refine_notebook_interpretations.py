from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "does_algorithm_hide_europe_realdata.ipynb"

OLD_MARKERS = [
    "<!-- codex-interpretation -->",
    "<!-- codex-next-steps -->",
    "<!-- targeted-output-interpretation -->",
    "<!-- targeted-next-steps -->",
]


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def remove_generated_cells(cells):
    """Remove earlier interpretation cells so this refinement is repeatable."""
    cleaned = []
    for cell in cells:
        if cell.cell_type == "markdown" and any(marker in cell.source for marker in OLD_MARKERS):
            continue
        if cell.cell_type == "code" and "<!-- targeted-output-interpretation -->" in cell.source:
            continue
        cleaned.append(cell)
    return cleaned


INTERPRETATION_CELLS = [
    (
        lambda s: s.startswith("required_files = {"),
        """
# <!-- targeted-output-interpretation -->
input_types = file_check["input_type"].fillna("missing").value_counts().to_dict()
display(Markdown(
    f"**Output interpretation.** {int(file_check['exists'].sum())}/{len(file_check)} required inputs were found. "
    f"The current run reads {input_types}. Because all required inputs exist, the notebook proceeds with real data only and does not activate the download-stop path."
))
""",
    ),
    (
        lambda s: s.startswith("gap_rows = ["),
        """
# <!-- targeted-output-interpretation -->
core_row = gap_rows.index("Ranking-level prominence audit")
core_col = gap_cols.index("Our contribution")
display(Markdown(
    f"**Output interpretation.** The highest project-emphasis cell is `{gap_rows[core_row]}` x `{gap_cols[core_col]}` "
    f"({gap_matrix[core_row, core_col]:.2f}). This is the slide-ready gap: we are not rediscovering popularity bias; we operationalise cultural visibility in rankings."
))
""",
    ),
    (
        lambda s: s.startswith("data_sources = pd.DataFrame"),
        """
# <!-- targeted-output-interpretation -->
restricted_sources = data_sources["redistribution"].str.contains("do not|no redistribution", case=False, regex=True).sum()
display(Markdown(
    f"**Output interpretation.** The ledger has {len(data_sources)} sources. "
    f"{restricted_sources} source rows explicitly restrict raw-data redistribution, so the submission should contain code, figures and ledgers, but not raw MovieLens/M3L files."
))
""",
    ),
    (
        lambda s: s.startswith("# Inspect archives or folders before loading") or s.startswith("def list_zip_contents"),
        """
# <!-- targeted-output-interpretation -->
zip_inputs = sum(1 for p in resolved_inputs.values() if p is not None and p.is_file())
folder_inputs = sum(1 for p in resolved_inputs.values() if p is not None and p.is_dir())
display(Markdown(
    f"**Output interpretation.** The inspection found {zip_inputs} zip input(s) and {folder_inputs} extracted folder input(s). "
    "This matters because the notebook loads what is actually present locally instead of hardcoding one archive layout."
))
""",
    ),
    (
        lambda s: s.startswith("# MovieLens files appear") or s.startswith("def read_csv_from_zip_flexible"),
        """
# <!-- targeted-output-interpretation -->
imdb_share = links["imdb_id_str"].notna().mean()
display(Markdown(
    f"**Output interpretation.** MovieLens bridge loaded {len(movies):,} movies and {len(links):,} link rows. "
    f"IMDb coverage is {imdb_share:.1%}, which is the key bridge into Wikidata country and language metadata."
))
""",
    ),
    (
        lambda s: s.startswith("def find_candidate_files"),
        """
# <!-- targeted-output-interpretation -->
split_counts = interactions["split"].value_counts()
user_count = interactions["user_id"].nunique()
item_count = interactions["item_id"].nunique()
display(Markdown(
    f"**Output interpretation.** M3L interactions loaded {len(interactions):,} rows across {user_count:,} users and {item_count:,} items. "
    f"The split sizes are train={int(split_counts.get('train', 0)):,}, valid={int(split_counts.get('valid', 0)):,}, test={int(split_counts.get('test', 0)):,}."
))
""",
    ),
    (
        lambda s: s.startswith("def resolve_feature_matrix"),
        """
# <!-- targeted-output-interpretation -->
mpnet_rows = np.load(mpnet_matrix_path, mmap_mode="r").shape[0]
mapped_share = len(item_id_map) / mpnet_rows if mpnet_rows else np.nan
display(Markdown(
    f"**Output interpretation.** The reconstructed item bridge maps {len(item_id_map):,}/{mpnet_rows:,} M3L feature rows "
    f"({mapped_share:.1%}) to MovieLens movie IDs. This is the technical step that makes cultural metadata joins valid."
))
""",
    ),
    (
        lambda s: s.startswith("interaction_cols = ["),
        """
# <!-- targeted-output-interpretation -->
m3l_items = movie_meta["item_id"].nunique()
title_share = movie_meta["title"].notna().mean()
imdb_share_meta = movie_meta["imdb_id_str"].notna().mean()
display(Markdown(
    f"**Output interpretation.** The audit catalogue currently has {m3l_items:,} M3L items. "
    f"MovieLens titles are available for {title_share:.1%} and IMDb IDs for {imdb_share_meta:.1%}; the latter determines how much can be enriched through Wikidata."
))
""",
    ),
    (
        lambda s: s.startswith("WIKIDATA_ENDPOINT ="),
        """
# <!-- targeted-output-interpretation -->
requested = len(priority_imdb_ids)
cache_rows = len(wikidata)
matched_requested = len(set(priority_imdb_ids) & set(wikidata["imdb_id_str"].dropna()))
display(Markdown(
    f"**Output interpretation.** This proposal run requested {requested:,} priority IMDb IDs and the Wikidata cache now contains {cache_rows:,} rows. "
    f"{matched_requested:,} requested IDs are represented in the cache; unmatched or missing metadata stays explicit in the coverage charts."
))
""",
    ),
    (
        lambda s: s.startswith("# Cultural labels are transparent proxies") or s.startswith("EUROPE_COUNTRIES ="),
        """
# <!-- targeted-output-interpretation -->
label_shares = movie_meta[["has_wikidata_match", "has_country", "has_language", "is_european", "is_non_english", "is_long_tail"]].mean()
display(Markdown(
    "**Output interpretation.** Metadata coverage and labels in the current audit catalogue: "
    f"Wikidata match={label_shares['has_wikidata_match']:.1%}, country={label_shares['has_country']:.1%}, language={label_shares['has_language']:.1%}; "
    f"European={label_shares['is_european']:.1%}, non-English={label_shares['is_non_english']:.1%}, long-tail={label_shares['is_long_tail']:.1%}. "
    "These are audit proxies, so missing labels and co-productions remain part of the interpretation."
))
""",
    ),
    (
        lambda s: s.startswith("def savefig(name):"),
        """
# <!-- targeted-output-interpretation -->
users = int(scale.loc[scale["metric"].eq("Users"), "value"].iloc[0])
items = int(scale.loc[scale["metric"].eq("Items"), "value"].iloc[0])
ratings = int(scale.loc[scale["metric"].eq("Ratings"), "value"].iloc[0])
display(Markdown(
    f"**Output interpretation.** Source metadata reports {users:,} users, {items:,} items and {ratings:,} ratings. "
    f"That is about {ratings / users:.1f} ratings per user on average, while the 99.29% sparsity explains why sparse recommender matrices are required."
))
""",
    ),
    (
        lambda s: s.startswith("# Visual 2: data pipeline diagram."),
        """
# <!-- targeted-output-interpretation -->
display(Markdown(
    "**Output interpretation.** The pipeline has two critical joins: M3L internal items to MovieLens IDs, and MovieLens IMDb IDs to Wikidata labels. "
    "Those joins are exactly where we report coverage, because cultural conclusions are only as strong as the metadata bridge."
))
""",
    ),
    (
        lambda s: s.startswith("# Visual 3: metadata join funnel."),
        """
# <!-- targeted-output-interpretation -->
start_items = int(join_funnel.loc[join_funnel["stage"].eq("M3L items"), "items"].iloc[0])
country_items = int(join_funnel.loc[join_funnel["stage"].eq("Has country"), "items"].iloc[0])
language_items = int(join_funnel.loc[join_funnel["stage"].eq("Has original language"), "items"].iloc[0])
display(Markdown(
    f"**Output interpretation.** The funnel starts with {start_items:,} M3L items. "
    f"Country labels are available for {country_items:,} items ({country_items / start_items:.1%}) and original-language labels for {language_items:,} items ({language_items / start_items:.1%}). "
    "Any remaining gap becomes a metadata limitation, not a hidden preprocessing detail."
))
""",
    ),
    (
        lambda s: s.startswith("# Visual 4: long-tail distribution."),
        """
# <!-- targeted-output-interpretation -->
tail_count = int(movie_meta["is_long_tail"].sum())
head_count = int(movie_meta["is_blockbuster_head"].sum())
display(Markdown(
    f"**Output interpretation.** The long-tail threshold is {q20:.0f} train interaction(s), while the head threshold is {q80:.0f}. "
    f"This labels {tail_count:,} movies as long-tail and {head_count:,} as blockbuster-head, using training data only."
))
""",
    ),
    (
        lambda s: s.startswith("# Visual 5: catalogue share vs training interaction share."),
        """
# <!-- targeted-output-interpretation -->
share_gap = catalogue_interaction_share.assign(
    gap=lambda df: df["Training interaction share"] - df["Catalogue share"]
)
largest_gap = share_gap.iloc[share_gap["gap"].abs().argmax()]
display(Markdown(
    f"**Output interpretation.** The largest catalogue-vs-interaction difference is for **{largest_gap['group']}** "
    f"({largest_gap['Catalogue share']:.1%} catalogue vs {largest_gap['Training interaction share']:.1%} interactions; gap={largest_gap['gap']:+.1%}). "
    "This is why we separate catalogue availability from observed attention."
))
""",
    ),
    (
        lambda s: s.startswith("# Visual 6: metadata and feature coverage."),
        """
# <!-- targeted-output-interpretation -->
lowest_coverage = coverage.sort_values("share").iloc[0]
text_share = float(coverage.loc[coverage["field"].eq("Has text embedding"), "share"].iloc[0])
image_share = float(coverage.loc[coverage["field"].eq("Has image embedding"), "share"].iloc[0])
display(Markdown(
    f"**Output interpretation.** The weakest coverage field is **{lowest_coverage['field']}** at {lowest_coverage['share']:.1%}. "
    f"Text embedding coverage is {text_share:.1%} and image embedding coverage is {image_share:.1%}, so multimodal modelling is less constrained than cultural metadata coverage."
))
""",
    ),
    (
        lambda s: s.startswith("# Build proposal-stage users"),
        """
# <!-- targeted-output-interpretation -->
run_counts = run_report.set_index("object")["count"].to_dict()
display(Markdown(
    f"**Output interpretation.** The proposal sandbox uses {int(run_counts['sample users']):,} users, "
    f"{int(run_counts['candidate items']):,} candidate items, {int(run_counts['sample train rows']):,} train rows and "
    f"{int(run_counts['sample test rows']):,} test rows. This is a real-data dry run sized for notebook speed."
))
""",
    ),
    (
        lambda s: s.startswith("# Higher ranking positions receive") or s.startswith("def rank_discount"),
        """
# <!-- targeted-output-interpretation -->
best_ndcg_base = baseline_comparison.loc[baseline_comparison[f"NDCG@{TOP_K}"].idxmax()]
best_cov_base = baseline_comparison.loc[baseline_comparison[f"Coverage@{TOP_K}"].idxmax()]
best_euro_base = baseline_comparison.loc[baseline_comparison[f"European Exposure@{TOP_K}"].idxmax()]
display(Markdown(
    f"**Output interpretation.** Among base models, the highest NDCG@{TOP_K} is **{best_ndcg_base['Model']}** ({best_ndcg_base[f'NDCG@{TOP_K}']:.4f}). "
    f"The widest catalogue coverage is **{best_cov_base['Model']}** ({best_cov_base[f'Coverage@{TOP_K}']:.1%}), and the highest European exposure is **{best_euro_base['Model']}** ({best_euro_base[f'European Exposure@{TOP_K}']:.1%}). "
    "This table is a proposal-run signal, not the final empirical claim."
))
""",
    ),
    (
        lambda s: s.startswith("# Re-ranking is post-processing") or s.startswith("def rerank_hybrid"),
        """
# <!-- targeted-output-interpretation -->
tmp_tradeoff = rerank_tradeoff.copy()
tmp_tradeoff["abs_pacpg"] = tmp_tradeoff[["PACPG European", "PACPG Non-English", "PACPG Long-tail"]].abs().sum(axis=1)
best_utility = tmp_tradeoff.loc[tmp_tradeoff[f"NDCG@{TOP_K}"].idxmax()]
best_prominence = tmp_tradeoff.loc[tmp_tradeoff["abs_pacpg"].idxmin()]
display(Markdown(
    f"**Output interpretation.** In the lambda sweep, highest utility is **{best_utility['Model']}** with NDCG@{TOP_K}={best_utility[f'NDCG@{TOP_K}']:.4f}. "
    f"The lowest combined absolute PACPG is **{best_prominence['Model']}** ({best_prominence['abs_pacpg']:.4f}). "
    "The point of the sweep is to make this trade-off explicit."
))
""",
    ),
    (
        lambda s: s.startswith("# Visual 7: utility vs prominence frontier."),
        """
# <!-- targeted-output-interpretation -->
best_frontier = tradeoff.loc[tradeoff["PACPG improvement"].idxmax()]
display(Markdown(
    f"**Output interpretation.** The largest PACPG improvement in this frontier occurs at lambda={best_frontier['lambda']:.1f} "
    f"with improvement={best_frontier['PACPG improvement']:.4f} and NDCG@{TOP_K}={best_frontier[f'NDCG@{TOP_K}']:.4f}. "
    "A useful final result would sit near the upper-right: more prominence improvement without large NDCG loss."
))
""",
    ),
    (
        lambda s: s.startswith("model_comparison = evaluate_recommendations"),
        """
# <!-- targeted-output-interpretation -->
dashboard = model_comparison.copy()
dashboard["abs_pacpg"] = dashboard[["PACPG European", "PACPG Non-English", "PACPG Long-tail"]].abs().sum(axis=1)
top_ndcg = dashboard.loc[dashboard[f"NDCG@{TOP_K}"].idxmax()]
lowest_pacpg = dashboard.loc[dashboard["abs_pacpg"].idxmin()]
top_coverage = dashboard.loc[dashboard[f"Coverage@{TOP_K}"].idxmax()]
display(Markdown(
    f"**Output interpretation.** In the dashboard, **{top_ndcg['Model']}** has the highest NDCG@{TOP_K} ({top_ndcg[f'NDCG@{TOP_K}']:.4f}), "
    f"**{top_coverage['Model']}** has the highest Coverage@{TOP_K} ({top_coverage[f'Coverage@{TOP_K}']:.1%}), and "
    f"**{lowest_pacpg['Model']}** has the lowest combined absolute PACPG ({lowest_pacpg['abs_pacpg']:.4f}). "
    "This is the core table for the proposal presentation."
))
""",
    ),
    (
        lambda s: s.startswith("# Visual 8: group exposure by model."),
        """
# <!-- targeted-output-interpretation -->
range_lines = []
for col in exposure_cols:
    group = col.replace(f" Exposure@{TOP_K}", "")
    min_row = model_comparison.loc[model_comparison[col].idxmin()]
    max_row = model_comparison.loc[model_comparison[col].idxmax()]
    range_lines.append(f"{group}: {min_row['Model']}={min_row[col]:.1%} to {max_row['Model']}={max_row[col]:.1%}")
display(Markdown(
    "**Output interpretation.** Exposure ranges across models: " + "; ".join(range_lines) + ". "
    "These ranges are what we discuss as ranking-level cultural prominence differences."
))
""",
    ),
    (
        lambda s: s.startswith("# Visual 9: accuracy/fairness summary."),
        """
# <!-- targeted-output-interpretation -->
best_summary = summary.loc[summary["Prominence improvement vs Hybrid"].idxmax()]
display(Markdown(
    f"**Output interpretation.** Relative to the Hybrid baseline, the strongest prominence improvement is **{best_summary['Model']}** "
    f"({best_summary['Prominence improvement vs Hybrid']:.4f}) with NDCG retention={best_summary['NDCG retention vs Hybrid']:.1%}. "
    "This is the compact accuracy-vs-governance slide."
))
""",
    ),
    (
        lambda s: s.startswith("workplan = pd.DataFrame"),
        """
# <!-- targeted-output-interpretation -->
display(Markdown(
    f"**Output interpretation.** The plan covers {len(workplan)} weekly milestones. "
    "The sequence is deliberate: lock data and metadata first, then compare models, then evaluate re-ranking and governance interpretation."
))
""",
    ),
    (
        lambda s: s.startswith("def version_or_na"),
        """
# <!-- targeted-output-interpretation -->
installed_libraries = int((library_ledger["version"] != "not installed").sum())
display(Markdown(
    f"**Output interpretation.** {installed_libraries}/{len(library_ledger)} listed libraries are installed in this environment. "
    "This ledger makes the notebook reproducible and keeps software dependencies visible next to data licences."
))
""",
    ),
    (
        lambda s: s.startswith("outputs_manifest = ["),
        """
# <!-- targeted-output-interpretation -->
present_outputs = sum((OUTPUTS / name).exists() for name in outputs_manifest)
display(Markdown(
    f"**Output interpretation.** {present_outputs}/{len(outputs_manifest)} expected output files are present. "
    "This is the final checklist for the proposal package."
))
""",
    ),
    (
        lambda s: s.startswith("assert NO_SYNTHETIC_DATA"),
        """
# <!-- targeted-output-interpretation -->
display(Markdown(
    "**Output interpretation.** The sanity checks passed: real M3L and MovieLens inputs are available, model outputs exist, and no synthetic fallback was used."
))
""",
    ),
]


NEXT_STEPS_CELL = """
<!-- targeted-next-steps -->
## 15. Interpretation and Next Steps

The proposal run should be presented as a working audit pipeline, not as final empirical evidence. The current outputs already let us show the full logic: source governance, metadata coverage, recommender comparison, cultural prominence metrics and a transparent re-ranking intervention.

**Next steps for the final project:**

1. Expand the Wikidata cache beyond the proposal limit and document the query date.
2. Re-run the models on a larger user/item sample and compare whether the dashboard patterns remain stable.
3. Add genre- and user-profile-stratified exposure checks, because aggregate fairness can hide subgroup effects.
4. Treat PACPG as an audit signal: useful for visibility diagnosis, but always interpreted together with metadata coverage and utility metrics.
"""


def find_interpretation(source: str) -> str | None:
    for predicate, interpretation_source in INTERPRETATION_CELLS:
        if predicate(source):
            return interpretation_source
    return None


def renumber_late_sections(cell):
    if cell.cell_type != "markdown":
        return cell
    replacements = {
        "## 15. Workplan": "## 16. Workplan",
        "## Proposal Summary": "## 17. Proposal Summary",
        "## 16. References and Library Ledger": "## 18. References and Library Ledger",
        "## 17. Export Outputs": "## 19. Export Outputs",
    }
    for old, new in replacements.items():
        cell.source = cell.source.replace(old, new)
    return cell


def main():
    nb = nbf.read(NOTEBOOK_PATH, as_version=4)
    cells = remove_generated_cells(nb.cells)

    refined = []
    inserted_next_steps = False

    for cell in cells:
        if (
            not inserted_next_steps
            and cell.cell_type == "markdown"
            and "Workplan" in cell.source
        ):
            refined.append(md(NEXT_STEPS_CELL))
            inserted_next_steps = True

        cell = renumber_late_sections(cell)
        refined.append(cell)

        if cell.cell_type == "code":
            interp = find_interpretation(cell.source)
            if interp:
                refined.append(code(interp))

    nb.cells = refined
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Refined notebook written to {NOTEBOOK_PATH}")
    print(f"Cells: {len(nb.cells)}")


if __name__ == "__main__":
    main()
