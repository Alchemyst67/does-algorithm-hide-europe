from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = PROJECT_ROOT / "final_submission" / "does_algorithm_hide_europe_final_repo"
ZIP_PATH = PROJECT_ROOT / "final_submission" / "does_algorithm_hide_europe_final_submission.zip"


def clean_and_create(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_file(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def copy_tree(src: Path, dst: Path, ignore=None):
    if not src.exists():
        return
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def clean_text(text: str) -> str:
    """Remove incidental quote/indent artifacts from long generated prose blocks."""
    cleaned = []
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith('"'):
            line = line[1:]
        if line.endswith('"'):
            line = line[:-1]
        cleaned.append(line)
    return "\n".join(cleaned).strip()


def make_roadmap_notebook(path: Path):
    cells = [
        new_markdown_cell(
            clean_text(
                """# Does the Algorithm Hide Europe?\n"
            "## Final Project Roadmap and Research-Question Answers\n\n"
            "**Team:** Max Priessnitz & Nico [surname]  \n"
            "**Course:** Data Science and Artificial Intelligence II: Data and Algorithmic Governance  \n"
            "**Purpose:** This short notebook is the entry point for the final submission. It explains the reading order, the architecture, and the main answers. The full evidence is in `04_final_research_story_executed.ipynb`.\n\n"
            "The storyline is deliberately simple: streaming platforms can satisfy catalogue-level diversity while ranking systems still decide what users actually see. We therefore audit cultural prominence at the Top-K ranking layer, not only catalogue availability."""
            )
        ),
        new_markdown_cell(
            clean_text(
                """## Reading order\n\n"
            "1. `00_project_roadmap_and_final_answers.ipynb` — this entry notebook.\n"
            "2. `01_data_foundation_movies_db.ipynb` — MovieLens/M3L/Wikidata data foundation.\n"
            "3. `02_model_pipeline_and_user_fold_robustness.ipynb` — recommender models, metrics and user-fold robustness.\n"
            "4. `03_feedback_loop_and_mitigation.ipynb` — Schedl-inspired feedback-loop stress test and mitigation logic.\n"
            "5. `04_final_research_story_executed.ipynb` — full thesis-like notebook with executed outputs, interpretations and limitations."""
            )
        ),
        new_code_cell(
            """from pathlib import Path\nimport pandas as pd\n\nROOT = Path.cwd()\nTABLES = ROOT / \"cultural_prominence_audit\" / \"outputs\" / \"final_notebook_tables\"\nASSETS = ROOT / \"cultural_prominence_audit\" / \"outputs\" / \"final_submission_assets\"\n\nanswers = pd.read_csv(TABLES / \"final_answer_table.csv\")\nanswers[[\"Question\", \"Short answer\", \"Confidence\", \"Main caveat\"]].head(18)"""
        ),
        new_markdown_cell(
            clean_text(
                """## What we found, in plain English\n\n"
            "- The answer is **model-dependent**, not a single yes/no slogan. Popularity produces the strongest Europe underexposure, while CLIP-image-content shows the highest Europe exposure.\n"
            "- The best ranking-utility model is **SVD**, but the best cultural-visibility model is not necessarily the same model.\n"
            "- **English-language content dominates** Top-20 exposure. MPNet-content is the strongest model for non-English exposure in this bounded run.\n"
            "- European visibility is uneven. France, Spain and Germany show the most discussable support-passing country-level gaps; several smaller countries are near-invisible and must be interpreted with support thresholds.\n"
            "- The stricter **local Europe** proxy is far less visible than globally compatible Europe. This is why we added the Visibility DNA extension.\n"
            "- Re-ranking can increase prominence, but the frontier shows a real utility cost. That trade-off is the governance result, not a hidden moral adjustment."""
            )
        ),
        new_code_cell(
            """model_summary = pd.read_csv(TABLES / \"aggregate_visibility_metrics.csv\")\nmodel_summary[[\"Model\", \"NDCG@20\", \"Recall@20\", \"Coverage@20\", \"Europe wide Exposure@20\", \"Non-English Exposure@20\", \"Europe wide PACPG@20\"]]"""
        ),
        new_code_cell(
            """country_scorecard = pd.read_csv(ASSETS / \"country_problem_scorecard.csv\")\ncountry_scorecard[[\"group_name\", \"mean_exposure_pct\", \"gap_pp\", \"top_genres\", \"problem_type\"]].head(12)"""
        ),
        new_code_cell(
            """company_caveats = pd.read_csv(ASSETS / \"production_company_caveat_summary.csv\")\ncompany_caveats"""
        ),
        new_markdown_cell(
            clean_text(
                """## How to use this submission\n\n"
            "For grading, start with the final presentation and this roadmap notebook. Then open the final research-story notebook for the detailed evidence. The HTML exports contain saved output state; the `.ipynb` files and scripts reproduce the results when the required local raw datasets are available. Raw MovieLens and M3L files are intentionally not redistributed."""
            )
        ),
    ]
    nb = new_notebook(cells=cells, metadata={"kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"}})
    path.parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, path)

    client = NotebookClient(nb, timeout=180, kernel_name="python3")
    client.execute()
    nbformat.write(nb, path)


def write_docs():
    readme = """# Does the Algorithm Hide Europe?\n\n"
    "**A multimodal audit of cultural prominence bias in movie recommender systems**  \n"
    "Team: Max Priessnitz & Nico [surname]  \n"
    "Course: Data Science and Artificial Intelligence II: Data and Algorithmic Governance, WU Vienna, 2026\n\n"
    "## Objective\n\n"
    "This project asks whether movie recommender systems trained on real interaction data under-expose European, non-English and long-tail films in Top-K recommendations, and whether multimodal features or transparent re-ranking can reduce this visibility gap without destroying recommendation utility.\n\n"
    "The central thesis is: **cultural diversity is not only a catalogue question; it is a ranking visibility problem.**\n\n"
    "## Repository Architecture\n\n"
    "- `notebooks/00_project_roadmap_and_final_answers.ipynb` — entry notebook with final answers and reading order.\n"
    "- `notebooks/01_data_foundation_movies_db.ipynb` — MovieLens/M3L/Wikidata data foundation and coverage checks.\n"
    "- `notebooks/02_model_pipeline_and_user_fold_robustness.ipynb` — model pipeline and bounded user-fold robustness.\n"
    "- `notebooks/03_feedback_loop_and_mitigation.ipynb` — feedback-loop stress test and mitigation logic.\n"
    "- `notebooks/04_final_research_story_executed.ipynb` — main thesis-like notebook with executed outputs and interpretations.\n"
    "- `html/` — HTML exports of the notebooks with saved output state.\n"
    "- `scripts/` — reproducible pipeline scripts.\n"
    "- `cultural_prominence_audit/outputs/` — derived tables, figures and final-submission assets.\n"
    "- `ui/` — interactive roadmap/dashboard layer for presentation and review.\n"
    "- `slides/` — final presentation materials.\n"
    "- `data/README_data.md` — dataset links, access instructions and raw-data redistribution policy.\n\n"
    "## Required Data\n\n"
    "Raw MovieLens and M3L/Binge Watch files are not redistributed. To reproduce from scratch, place the required local files under `data/raw/` or use the extracted folders expected by the scripts:\n\n"
    "- M3L-20M / Binge Watch interaction and multimodal features\n"
    "- MovieLens 20M metadata and ratings\n"
    "- Wikidata enrichment via SPARQL, cached in `data/interim/`\n\n"
    "See `data/README_data.md` for exact links and commands.\n\n"
    "## Setup\n\n"
    "```bash\n"
    "uv venv --python 3.12 .venv\n"
    "uv pip install --python .venv/bin/python -r requirements.txt\n"
    "jupyter lab\n"
    "```\n\n"
    "If `uv` is not available, use a Python 3.12 virtual environment and install the same `requirements.txt`.\n\n"
    "## Reproduction Order\n\n"
    "```bash\n"
    "python scripts/build_movies_db.py\n"
    "python scripts/run_full_cultural_prominence_audit.py\n"
    "python scripts/run_recommender_cross_validation.py\n"
    "python scripts/run_schedl_style_feedback_loop.py\n"
    "python scripts/build_visibility_dna_enrichment.py\n"
    "python scripts/build_final_submission_assets.py\n"
    "```\n\n"
    "The main notebook can then be executed with:\n\n"
    "```bash\n"
    "jupyter nbconvert --to notebook --execute notebooks/04_final_research_story_executed.ipynb --output rerun_final_story.ipynb --ExecutePreprocessor.timeout=1200\n"
    "```\n\n"
    "## Models\n\n"
    "We compare six recommender families: Popularity, ItemKNN, TruncatedSVD collaborative filtering, MPNet text-content recommendation, CLIP-image-content recommendation and a simple Hybrid. A transparent governance-aware re-ranker is then applied as a post-processing mitigation layer.\n\n"
    "## Metrics\n\n"
    "Utility is measured with NDCG@20, Recall@20, MAP@20 and catalogue coverage. Cultural prominence is measured through discounted Top-K group exposure and PACPG: Preference-Adjusted Cultural Prominence Gap. PACPG compares ranked exposure against transparent targets based on history and relevant test data.\n\n"
    "## Main Results\n\n"
    "- Popularity is the strongest underexposure baseline for Europe in this run.\n"
    "- SVD is the best utility model by NDCG@20.\n"
    "- CLIP-image-content creates the highest Europe-wide exposure, but with low utility.\n"
    "- English-language exposure is dominant across model outputs.\n"
    "- France shows the clearest support-passing negative country-level gap; Spain and Germany are also important diagnostic cases.\n"
    "- The Visibility DNA extension suggests globally compatible Europe is much more visible than stricter local-Europe proxies.\n"
    "- Re-ranking improves cultural prominence only with a measurable utility trade-off.\n\n"
    "## State of the Art and Positioning\n\n"
    "The project builds on popularity-bias and long-tail exposure research, provider/item-side fairness, multimodal movie recommendation and recommender feedback-loop work. The gap is that cultural prominence is operationalised as a ranking-layer governance audit with country/language proxy labels rather than only as catalogue share or accuracy.\n\n"
    "Key sources include Klimashevskaia et al. (2023), Abdollahpouri et al. (2019), Wang/Jin fairness-survey work, Spillo et al. on M3L/Binge Watch, Lesota/Schedl feedback-loop work, MovieLens documentation, Wikidata documentation and EU AVMSD/DSA governance materials. See `CITATION_AND_REUSE_LEDGER.md` and notebook references.\n\n"
    "## AI Use\n\n"
    "Generative AI was used for code scaffolding, notebook structuring, documentation wording, visualization/UI drafting and presentation drafting. All research claims are tied to executed outputs or cited sources. No synthetic data was generated. Details are in `AI_USE_DISCLOSURE.md`.\n\n"
    "## Licence and Raw-Data Policy\n\n"
    "This repository does not redistribute raw MovieLens or M3L data. Users must download those datasets from their original providers and respect their terms. Derived figures/tables are included for review of the submitted analysis.\n"
    """

    data_readme = """# Data Access and Raw-Data Policy\n\n"
    "This project uses real data only. No synthetic fallback data is generated.\n\n"
    "## Required Sources\n\n"
    "- **M3L-20M / Binge Watch**: main MovieLens-based interaction data and multimodal features.\n"
    "- **MovieLens 20M**: movie metadata, ratings and identifier bridge.\n"
    "- **Wikidata**: country of origin, original language, production company, director and additional visibility-DNA metadata via SPARQL.\n\n"
    "## Raw Data\n\n"
    "Raw MovieLens and M3L files are not included in this submission package and should not be pushed to a public repository. Download them from the original providers and place them in `data/raw/` or in the expected extracted local folders.\n\n"
    "## Wikidata Cache\n\n"
    "Wikidata is CC0. The scripts cache query results under `data/interim/` during reproduction. The submitted derived outputs document query-based metadata coverage and missingness.\n"
    """

    ai = """# Generative AI Use Disclosure\n\n"
    "We used generative AI tools, including OpenAI Codex/ChatGPT, as a coding and writing assistant during the project.\n\n"
    "## What AI Was Used For\n\n"
    "- Scaffolding Python scripts and Jupyter notebook sections.\n"
    "- Refactoring the analysis into a clearer thesis-like storyline.\n"
    "- Drafting markdown explanations, figure interpretations and README documentation.\n"
    "- Drafting interactive UI and presentation structure.\n"
    "- Helping formulate caveats around country, language, production-company and feedback-loop methodology.\n\n"
    "## What AI Was Not Used For\n\n"
    "- No synthetic data was generated.\n"
    "- No unsupported empirical findings were invented.\n"
    "- AI outputs were not treated as sources. Literature and dataset claims are cited separately.\n\n"
    "## Human Responsibility\n\n"
    "Max Priessnitz and Nico [surname] remain responsible for the project framing, interpretation, source selection, validation and final submission. Code and text were reviewed and adapted to match our project language and university requirements.\n"
    """

    citations = """# Citation and Reuse Ledger\n\n"
    "## Datasets and Platforms\n\n"
    "- M3L-20M / Binge Watch dataset, Zenodo record: multimodal MovieLens extension with text, image, audio and video features.\n"
    "- MovieLens 20M / GroupLens: movie metadata, links and ratings.\n"
    "- Wikidata: SPARQL Query Service and CC0 metadata, including country/language/production-company properties.\n"
    "- EU Audiovisual Media Services Directive and digital-strategy materials for cultural-diversity governance framing.\n\n"
    "## Research Literature\n\n"
    "- Klimashevskaia et al. (2023), survey on popularity bias in recommender systems.\n"
    "- Abdollahpouri et al. (2019), popularity-bias unfairness and personalized re-ranking.\n"
    "- Wang/Jin and related fairness surveys on recommender exposure and item/provider fairness.\n"
    "- Spillo et al., Binge Watch / M3L multimodal movie recommendation benchmark.\n"
    "- Lesota, Geiger, Walder, Kowald and Schedl (2024), recommender feedback-loop methodology and country-representation dynamics.\n"
    "- MovieLens dataset documentation and papers.\n\n"
    "## Reused Code\n\n"
    "No third-party project code was copied into the repository. The implementation uses standard public Python and JavaScript libraries listed in `requirements.txt` and package docs. Concepts and methods from the literature are cited in the notebooks and described in the methodology.\n\n"
    "## AI-Assisted Code\n\n"
    "Some code and documentation were drafted with OpenAI Codex/ChatGPT assistance and reviewed by the project team. See `AI_USE_DISCLOSURE.md`.\n"
    """

    checklist = """# Reproducibility Checklist\n\n"
    "- [x] Real data only; no synthetic fallback.\n"
    "- [x] Raw MovieLens/M3L data not redistributed.\n"
    "- [x] Dataset links and licence caveats documented.\n"
    "- [x] Multiple notebooks included with a clear reading order.\n"
    "- [x] HTML exports included with saved outputs.\n"
    "- [x] Models, metrics and caveats explained in markdown.\n"
    "- [x] Country, language, genre and production-company caveats included.\n"
    "- [x] Final research questions answered in a dedicated table.\n"
    "- [x] AI use disclosed.\n"
    "- [x] Source/citation ledger included.\n"
    """

    github_txt = """https://github.com/nicospacez/dsai2\n\n"
    "This is the intended public GitHub repository for the project. If the final-submission folder is used as the release package, copy or push its contents to the public repository before uploading the university submission zip.\n"
    """

    files = {
        "README.md": readme,
        "data/README_data.md": data_readme,
        "AI_USE_DISCLOSURE.md": ai,
        "CITATION_AND_REUSE_LEDGER.md": citations,
        "REPRODUCIBILITY_CHECKLIST.md": checklist,
        "github.txt": github_txt,
        ".gitignore": "__pycache__/\n.ipynb_checkpoints/\n.venv/\ndata/raw/\n*.zip\n.DS_Store\n",
    }
    for rel, text in files.items():
        target = FINAL_ROOT / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(clean_text(text) + "\n", encoding="utf-8")


def export_html(notebook_paths: list[Path]):
    html_dir = FINAL_ROOT / "html"
    html_dir.mkdir(parents=True, exist_ok=True)
    for notebook in notebook_paths:
        subprocess.run(
            [
                "jupyter",
                "nbconvert",
                "--to",
                "html",
                str(notebook),
                "--output-dir",
                str(html_dir),
            ],
            cwd=FINAL_ROOT,
            check=True,
        )


def main():
    clean_and_create(FINAL_ROOT)
    (PROJECT_ROOT / "final_submission").mkdir(exist_ok=True)

    copy_file(PROJECT_ROOT / "requirements.txt", FINAL_ROOT / "requirements.txt")

    notebook_map = {
        "01_data_foundation_movies_db.ipynb": PROJECT_ROOT / "notebooks" / "movies_db_pipeline.ipynb",
        "02_model_pipeline_and_user_fold_robustness.ipynb": PROJECT_ROOT / "notebooks" / "recommender_pipeline_cross_validation.ipynb",
        "03_feedback_loop_and_mitigation.ipynb": PROJECT_ROOT / "notebooks" / "schedl_feedback_loop_audit.ipynb",
        "04_final_research_story_executed.ipynb": PROJECT_ROOT / "notebooks" / "does_algorithm_hide_europe_final_research_story_executed.ipynb",
        "04_final_research_story_source.ipynb": PROJECT_ROOT / "notebooks" / "does_algorithm_hide_europe_final_research_story.ipynb",
    }
    for name, src in notebook_map.items():
        copy_file(src, FINAL_ROOT / "notebooks" / name)

    make_roadmap_notebook(FINAL_ROOT / "notebooks" / "00_project_roadmap_and_final_answers.ipynb")

    copy_tree(PROJECT_ROOT / "scripts", FINAL_ROOT / "scripts", ignore=shutil.ignore_patterns("__pycache__"))
    copy_tree(PROJECT_ROOT / "ui", FINAL_ROOT / "ui")
    copy_tree(PROJECT_ROOT / "cultural_prominence_audit" / "outputs", FINAL_ROOT / "cultural_prominence_audit" / "outputs")

    slides_dir = FINAL_ROOT / "slides"
    slides_dir.mkdir(parents=True, exist_ok=True)
    for deck in [
        PROJECT_ROOT / "DSAI2_proposal.pptx",
        PROJECT_ROOT / "outputs" / "does_algorithm_hide_europe_proposal_deck.pptx",
        PROJECT_ROOT / "outputs" / "does_algorithm_hide_europe_final_presentation.pptx",
    ]:
        if deck.exists():
            copy_file(deck, slides_dir / deck.name)

    write_docs()

    export_html(
        [
            FINAL_ROOT / "notebooks" / "00_project_roadmap_and_final_answers.ipynb",
            FINAL_ROOT / "notebooks" / "01_data_foundation_movies_db.ipynb",
            FINAL_ROOT / "notebooks" / "02_model_pipeline_and_user_fold_robustness.ipynb",
            FINAL_ROOT / "notebooks" / "03_feedback_loop_and_mitigation.ipynb",
            FINAL_ROOT / "notebooks" / "04_final_research_story_executed.ipynb",
        ]
    )

    manifest = {
        "repo_folder": str(FINAL_ROOT),
        "zip_path": str(ZIP_PATH),
        "notebooks": sorted(p.name for p in (FINAL_ROOT / "notebooks").glob("*.ipynb")),
        "html_exports": sorted(p.name for p in (FINAL_ROOT / "html").glob("*.html")),
        "raw_data_included": False,
        "note": "Raw MovieLens/M3L data are excluded; data links and reproduction instructions are in data/README_data.md.",
    }
    (FINAL_ROOT / "submission_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    shutil.make_archive(str(ZIP_PATH.with_suffix("")), "zip", FINAL_ROOT.parent, FINAL_ROOT.name)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
