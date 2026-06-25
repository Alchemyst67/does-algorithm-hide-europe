from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "does_algorithm_hide_europe_realdata.ipynb"

INTERPRETATION_MARKER = "<!-- codex-interpretation -->"
NEXT_STEPS_MARKER = "<!-- codex-next-steps -->"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def remove_generated_markdown(cells):
    """Keep reruns idempotent by removing interpretation cells from earlier enhancer runs."""
    return [
        cell
        for cell in cells
        if not (
            cell.cell_type == "markdown"
            and (INTERPRETATION_MARKER in cell.source or NEXT_STEPS_MARKER in cell.source)
        )
    ]


COMMENT_REPLACEMENTS = [
    (
        'plt.style.use("seaborn-v0_8-whitegrid")\n\nPROJECT_ROOT = Path(".").resolve()',
        'plt.style.use("seaborn-v0_8-whitegrid")\n\n'
        "# All paths are rooted in the project folder so the notebook can be rerun on our machines.\n"
        'PROJECT_ROOT = Path(".").resolve()',
    ),
    (
        "SEED = 42\nTOP_K = 20",
        "SEED = 42\n"
        "# These controls separate a fast proposal run from the later full audit.\n"
        "TOP_K = 20",
    ),
    (
        "known_alternatives = {",
        "# We accept both canonical downloaded zip files and the extracted folder layout used in this project.\n"
        "known_alternatives = {",
    ),
    (
        'if not file_check["exists"].all():',
        "# Hard stop: missing real files must not be replaced by synthetic or placeholder data.\n"
        'if not file_check["exists"].all():',
    ),
    (
        "def list_zip_contents(zip_path, max_rows=50):",
        "# Inspect archives or folders before loading so we do not depend on hidden filename assumptions.\n"
        "def list_zip_contents(zip_path, max_rows=50):",
    ),
    (
        "def read_csv_from_zip_flexible(zip_path, candidate_names):",
        "# MovieLens files appear under slightly different names across local layouts; this resolver keeps the bridge robust.\n"
        "def read_csv_from_zip_flexible(zip_path, candidate_names):",
    ),
    (
        "def standardise_interactions(df):",
        "# Rename M3L/MMRec columns into one interaction schema used by every model and metric.\n"
        "def standardise_interactions(df):",
    ),
    (
        "def vector_fingerprint(vec):",
        "# M3L uses internal item IDs; compact MPNet fingerprints let us reconstruct the MovieLens bridge without guessing.\n"
        "def vector_fingerprint(vec):",
    ),
    (
        "def query_wikidata_by_imdb(imdb_ids, batch_size=100, sleep=0.25):",
        "# Wikidata is queried in small cached batches, which keeps reruns polite and reproducible.\n"
        "def query_wikidata_by_imdb(imdb_ids, batch_size=100, sleep=0.25):",
    ),
    (
        "priority_movie_ids = item_priority",
        "# For the proposal run, query the highest-priority movies first; RUN_FULL expands this later.\n"
        "priority_movie_ids = item_priority",
    ),
    (
        "EUROPE_COUNTRIES = {",
        "# Cultural labels are transparent proxies; unknown metadata stays visible instead of being imputed.\n"
        "EUROPE_COUNTRIES = {",
    ),
    (
        'q80 = movie_meta["train_interaction_count"].quantile(0.80)',
        "# Long-tail and head labels use training popularity only, so test data does not leak into the labels.\n"
        'q80 = movie_meta["train_interaction_count"].quantile(0.80)',
    ),
    (
        "# Build proposal-stage users and candidate item universe.\ntrain_user_counts",
        "# Build proposal-stage users and candidate item universe.\n"
        "# We sample eligible users but keep their real train/test interactions unchanged.\n"
        "train_user_counts",
    ),
    (
        "top_pop_items = item_counts",
        "# Candidate items include popular movies plus all sampled train/test items, so evaluation remains feasible.\n"
        "top_pop_items = item_counts",
    ),
    (
        "def mask_seen(scores):",
        "# Seen items are masked so the Top-K list behaves like recommendations for unseen movies.\n"
        "def mask_seen(scores):",
    ),
    (
        "def content_scores_from_features(feature_path):",
        "# A user content profile is the mean feature vector of movies they positively interacted with in training.\n"
        "def content_scores_from_features(feature_path):",
    ),
    (
        "def rank_discount(rank):",
        "# Higher ranking positions receive more visibility weight in all exposure-based metrics.\n"
        "def rank_discount(rank):",
    ),
    (
        "group_columns = {",
        "# PACPG compares exposure with user-level targets derived from history and relevant test items.\n"
        "group_columns = {",
    ),
    (
        "def rerank_hybrid(lambda_value=0.3, k=TOP_K, candidate_k=CANDIDATE_K):",
        "# Re-ranking is post-processing: it changes candidate order without retraining the hybrid model.\n"
        "def rerank_hybrid(lambda_value=0.3, k=TOP_K, candidate_k=CANDIDATE_K):",
    ),
    (
        "final_score = float(rel_lookup[item]) + lambda_value * bonus",
        "# The bonus only helps when an item moves the current list toward the transparent cultural target.\n"
        "                final_score = float(rel_lookup[item]) + lambda_value * bonus",
    ),
    (
        'outputs_manifest = [\n    "01_dataset_scale.png",',
        'outputs_manifest = [\n    "00_research_gap_matrix.png",\n    "01_dataset_scale.png",',
    ),
]


def add_comments_to_code(source: str) -> str:
    """Add short explanatory comments only where they clarify the audit logic."""
    updated = source
    for old, new in COMMENT_REPLACEMENTS:
        if old in updated and new not in updated:
            updated = updated.replace(old, new)
    return updated


INTERPRETATIONS = [
    (
        lambda s: s.startswith("from pathlib import Path"),
        "This confirms the active project directory. All following paths, caches and outputs are anchored here, which keeps our proposal run reproducible.",
    ),
    (
        lambda s: s.startswith("required_files = {"),
        "The file check is our first governance control. If a required input is missing, the notebook stops instead of inventing fallback data.",
    ),
    (
        lambda s: s.startswith("gap_rows = ["),
        "We read this matrix as the conceptual gap: popularity bias is established, while cultural prominence under multimodal recommendation is where our proposal contributes.",
    ),
    (
        lambda s: s.startswith("data_sources = pd.DataFrame"),
        "The ledger makes the data strategy explicit: M3L provides interactions and features, MovieLens provides identifiers, and Wikidata provides cultural labels. Raw licensed data stays out of the submission package.",
    ),
    (
        lambda s: s.startswith("# Inspect archives or folders before loading") or s.startswith("def list_zip_contents"),
        "These listings are a loading guardrail. We inspect the real files first, then write loaders around the observed structure rather than assuming hidden filenames.",
    ),
    (
        lambda s: s.startswith("# MovieLens files appear") or s.startswith("def read_csv_from_zip_flexible"),
        "The bridge output tells us MovieLens titles, genres and IMDb identifiers are available. We use it only to connect M3L items to external metadata, not as a replacement dataset.",
    ),
    (
        lambda s: s.startswith("def find_candidate_files"),
        "The interaction tables prove that the recommender audit starts from real M3L train/validation/test splits. The unique user and item counts define the empirical scale of the project.",
    ),
    (
        lambda s: s.startswith("def resolve_feature_matrix"),
        "This mapping step is technically important: M3L item IDs are internal, so we reconstruct their MovieLens IDs through MPNet feature matches before adding cultural metadata.",
    ),
    (
        lambda s: s.startswith("interaction_cols = ["),
        "This table is the first audit catalogue after the ID bridge. The printed counts show how many items have titles and IMDb IDs available for Wikidata enrichment.",
    ),
    (
        lambda s: s.startswith("WIKIDATA_ENDPOINT ="),
        "Wikidata enrichment turns identifiers into country and language labels. Missing values remain visible because metadata coverage is part of the audit rather than something to hide.",
    ),
    (
        lambda s: s.startswith("# Cultural labels are transparent proxies") or s.startswith("EUROPE_COUNTRIES ="),
        "These rows and shares are our label sanity check. European, non-English and long-tail are transparent proxies, so the final analysis must discuss co-productions, multilingual works and missing labels.",
    ),
    (
        lambda s: s.startswith("def savefig(name):"),
        "The scale chart establishes that the source benchmark is large, while this notebook intentionally uses a bounded proposal run for speed. The chart is context, not a claim from our sample.",
    ),
    (
        lambda s: s.startswith("# Visual 2: data pipeline diagram."),
        "The pipeline visual is the story in one figure: real interactions and multimodal features become an enriched audit dataset, then model comparison and mitigation.",
    ),
    (
        lambda s: s.startswith("# Visual 3: metadata join funnel."),
        "The funnel shows where audit coverage can shrink. Any drop in Wikidata, country or language coverage becomes a limitation to report, not a silent preprocessing detail.",
    ),
    (
        lambda s: s.startswith("# Visual 4: long-tail distribution."),
        "The popularity distribution motivates the project technically. Movie attention is skewed, so accuracy alone can miss whether long-tail works ever become visible.",
    ),
    (
        lambda s: s.startswith("# Visual 5: catalogue share vs training interaction share."),
        "This comparison separates availability from observed attention. If catalogue share and interaction share diverge, ranking-level exposure needs to be audited separately.",
    ),
    (
        lambda s: s.startswith("# Visual 6: metadata and feature coverage."),
        "Coverage tells us which labels and features are usable in the proposal run. It also tells us where the full project needs broader Wikidata querying or clearer caveats.",
    ),
    (
        lambda s: s.startswith("# Build proposal-stage users"),
        "The run report defines our experimental sandbox: sampled users, candidate items and real sparse matrices. This is a speed-bounded proposal setup, not the final empirical run.",
    ),
    (
        lambda s: s.startswith("# Seen items are masked") or s.startswith("def mask_seen"),
        "The popularity baseline is the mainstream-concentration reference point. It shows what happens when recommendations mostly follow aggregate demand.",
    ),
    (
        lambda s: s.startswith("# Model B: ItemKNN"),
        "ItemKNN gives us a transparent collaborative-filtering baseline. It helps separate cultural exposure caused by co-consumption similarity from exposure caused by pure popularity.",
    ),
    (
        lambda s: s.startswith("# Model C: TruncatedSVD"),
        "SVD is a lightweight latent-factor baseline for the proposal stage. It gives us a stronger collaborative model without turning the notebook into a heavy training pipeline.",
    ),
    (
        lambda s: s.startswith("# Models D and E"),
        "The content models test whether plot text and poster images change cultural visibility compared with interaction-only recommenders.",
    ),
    (
        lambda s: s.startswith("# Model F: explicit hybrid."),
        "The hybrid keeps weights simple and readable. That is useful for a proposal because we can explain the mechanism before later tuning it.",
    ),
    (
        lambda s: s.startswith("# Higher ranking positions receive") or s.startswith("def rank_discount"),
        "The baseline comparison already combines utility, coverage and cultural exposure. The key readout is not just which model predicts best, but which model allocates visibility differently.",
    ),
    (
        lambda s: s.startswith("# Re-ranking is post-processing") or s.startswith("def rerank_hybrid"),
        "The lambda table is the mitigation experiment. It shows how strongly the audit objective can intervene in the hybrid ranking and what happens to utility and PACPG.",
    ),
    (
        lambda s: s.startswith("# Visual 7: utility vs prominence frontier."),
        "This frontier is the governance trade-off in visual form. We will use it to ask whether cultural prominence improves and how much ranking utility that costs.",
    ),
    (
        lambda s: s.startswith("model_comparison = evaluate_recommendations"),
        "This dashboard table is the central audit output. Because this is a bounded proposal run, we treat the numbers as a pipeline validation and first signal, not final evidence.",
    ),
    (
        lambda s: s.startswith("# Visual 8: group exposure by model."),
        "The grouped bars make model differences easy to discuss. They answer which algorithms give more or less Top-K exposure to European, non-English and long-tail films.",
    ),
    (
        lambda s: s.startswith("# Visual 9: accuracy/fairness summary."),
        "The scatter condenses the trade-off. The desirable region is high utility retention with lower absolute PACPG.",
    ),
    (
        lambda s: s.startswith("workplan = pd.DataFrame"),
        "The workplan follows the research logic: make the data trustworthy, compare models, test mitigation and then interpret the governance implications.",
    ),
    (
        lambda s: s.startswith("def version_or_na"),
        "The library ledger makes the computational environment auditable. It also keeps software licences visible next to the data-source licences.",
    ),
    (
        lambda s: s.startswith("outputs_manifest = ["),
        "The output manifest is our reproducibility checklist. It records which figures and tables exist and repeats the scope note for the proposal-stage run.",
    ),
    (
        lambda s: s.startswith("assert NO_SYNTHETIC_DATA"),
        "The final checks enforce the hard rules: real data loaded, no synthetic fallback used, and core audit outputs created.",
    ),
]


NEXT_STEPS_CELL = f"""
{NEXT_STEPS_MARKER}
## 15. Interpretation and Next Steps

### What this proposal run already tells us

- The audit architecture works end to end: real M3L interactions, MovieLens identifiers, Wikidata cultural metadata, multimodal features, recommenders, metrics and saved visuals all connect in one notebook.
- The important object is ranking visibility, not just catalogue availability. We therefore compare catalogue share, interaction share, Top-K exposure and user-adjusted prominence targets.
- The first model dashboard is a proposal-stage signal. We can use it to explain the method, but we will not present it as final empirical evidence until the full run is executed.

### What we will say carefully

- "European", "non-English" and "long-tail" are useful audit proxies, not perfect cultural identities.
- Co-productions, multilingual films and missing Wikidata labels can bias results, so metadata coverage is reported as an explicit part of the project.
- Re-ranking is not presented as a moral truth machine. It is a transparent intervention whose utility cost and prominence benefit can be measured.

### Next steps for the full project

1. Expand Wikidata enrichment beyond the proposal limit and freeze the query date.
2. Run the same models on a larger user and item sample, then check whether the patterns are stable.
3. Add genre-stratified and user-profile-stratified views to see whether under-exposure is systematic or concentrated in specific contexts.
4. Tune the hybrid and re-ranking lambda values only after the baseline audit is reproducible.
5. Turn the final notebook results into a short governance discussion: what platforms could monitor, what trade-offs remain, and where the metric should not be overinterpreted.
"""


def interpretation_for(source: str) -> str | None:
    for predicate, text in INTERPRETATIONS:
        if predicate(source):
            return f"{INTERPRETATION_MARKER}\n**Interpretation:** {text}"
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
    source = cell.source
    for old, new in replacements.items():
        source = source.replace(old, new)
    cell.source = source
    return cell


def main():
    nb = nbf.read(NOTEBOOK_PATH, as_version=4)
    base_cells = remove_generated_markdown(nb.cells)

    enhanced_cells = []
    next_steps_inserted = False

    for cell in base_cells:
        if cell.cell_type == "code":
            cell.source = add_comments_to_code(cell.source)

        cell = renumber_late_sections(cell)

        if (
            not next_steps_inserted
            and cell.cell_type == "markdown"
            and cell.source.lstrip().startswith("## 16. Workplan")
        ):
            enhanced_cells.append(md(NEXT_STEPS_CELL))
            next_steps_inserted = True

        enhanced_cells.append(cell)

        if cell.cell_type == "code":
            interpretation = interpretation_for(cell.source)
            if interpretation:
                enhanced_cells.append(md(interpretation))

    nb.cells = enhanced_cells
    nbf.write(nb, NOTEBOOK_PATH)
    print(f"Enhanced notebook written to {NOTEBOOK_PATH}")
    print(f"Cells: {len(base_cells)} -> {len(enhanced_cells)}")


if __name__ == "__main__":
    main()
