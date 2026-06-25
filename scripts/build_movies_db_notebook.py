from __future__ import annotations

from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "movies_db_pipeline.ipynb"


def main() -> None:
    cells = [
        new_markdown_cell(
            """# Movies DB Pipeline
## MovieLens + M3L Dataset Exploration and Combination

**Project:** Does the Algorithm Hide Europe?  
**Team:** Max Priessnitz & Nico [surname]  

This notebook is the data-foundation notebook for the project. It builds and documents a combined movie-level database from MovieLens 20M metadata, ratings, tags, genome tags, available M3L feature coverage and cached Wikidata cultural metadata.

The purpose is not to train recommender models. The purpose is to make the movie catalogue, metadata joins, coverage gaps and audit-ready slices explicit before modelling.
"""
        ),
        new_markdown_cell(
            """## 1. Why This Notebook Exists

The earlier exploration notebook showed the right structure:

- inspect local files first;
- use MovieLens `movieId` as the safest join key;
- add ratings, user tags and genome tags;
- add M3L plots/posters/trailers and feature coverage where available;
- enrich through IMDb/Wikidata;
- persist a compact movie-level table;
- interpret coverage, genre, decade and user-activity patterns.

This notebook adapts that structure to the current project directory. If Nico's `M3L_10M_20M-main` raw TSV folder is not present locally, missing plot/poster/trailer files are reported as missing. They are not simulated.
"""
        ),
        new_code_cell(
            """from pathlib import Path
import subprocess
import sys

import pandas as pd
from IPython.display import Image, Markdown, display

PROJECT_ROOT = Path("..").resolve()
OUTPUTS = PROJECT_ROOT / "outputs"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
SCRIPT = PROJECT_ROOT / "scripts" / "build_movies_db.py"

RUN_MOVIES_DB_PIPELINE = False  # Set to True to rebuild all Movies DB outputs from this notebook.
"""
        ),
        new_markdown_cell(
            """## 2. Rebuild or Load Cached Outputs

The heavy lifting lives in `scripts/build_movies_db.py` so the pipeline can be run from terminal, notebook or CI-like checks. The notebook keeps the outputs readable and interpretable.
"""
        ),
        new_code_cell(
            """if RUN_MOVIES_DB_PIPELINE:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=PROJECT_ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    print(completed.stdout[-5000:])
else:
    print("Using cached Movies DB outputs. Set RUN_MOVIES_DB_PIPELINE=True to rebuild.")
"""
        ),
        new_markdown_cell(
            """## 3. Local File Inventory

The first governance step is boring but essential: we must know which data files actually exist. This prevents the notebook from silently mixing available data with assumptions from another machine.
"""
        ),
        new_code_cell(
            """file_inventory = pd.read_csv(OUTPUTS / "27_movies_db_file_inventory.csv")
display(file_inventory)

missing_or_pointer = file_inventory[(~file_inventory["exists"]) | (file_inventory["is_lfs_pointer"])]
display(Markdown(
    f"**Interpretation.** The inventory checks {len(file_inventory)} expected inputs. "
    f"{len(missing_or_pointer)} are missing or Git-LFS pointers in this project directory. "
    "Those files are documented as coverage limitations and are not replaced with synthetic data."
))
"""
        ),
        new_markdown_cell(
            """## 4. Core Movie Metadata and Join Keys

MovieLens `movieId` is the main internal key. IMDb IDs are used only for external cultural metadata enrichment through Wikidata.
"""
        ),
        new_code_cell(
            """core_summary = pd.read_csv(OUTPUTS / "27_movies_db_core_metadata_summary.csv")
movieid_overlap = pd.read_csv(OUTPUTS / "27_movies_db_movieid_overlap.csv")
title_mismatches = pd.read_csv(OUTPUTS / "27_movies_db_title_mismatches.csv")

display(core_summary)
display(movieid_overlap.round(4))
display(title_mismatches.head(20))

display(Markdown(
    f"**Interpretation.** The combined table starts from {int(core_summary.loc[core_summary['dataset'].eq('MovieLens movies'), 'rows'].iloc[0]):,} "
    f"MovieLens movies. The loaded MovieLens/M3L-20M metadata overlap is "
    f"{movieid_overlap.loc[0, 'jaccard']:.1%}. "
    f"The current local run finds {len(title_mismatches):,} title mismatches because the separate Nico M3L metadata folder is not present; "
    "the previous exploration notebook documented 183 mismatches when that folder was available. The methodological conclusion stays the same: use `movieId`, not title strings, for joins."
))
"""
        ),
        new_markdown_cell(
            """## 5. Ratings, Tags and Genome Tags

The pipeline uses the first 1,000,000 MovieLens ratings for a responsive dataset-discovery run, matching the earlier exploration setup. This sample is for metadata discovery and audit preparation, not for final modelling claims.
"""
        ),
        new_code_cell(
            """rating_stats = pd.read_csv(OUTPUTS / "27_movies_db_rating_stats.csv")
tag_stats = pd.read_csv(OUTPUTS / "27_movies_db_tag_stats.csv")
genome_summary = pd.read_csv(OUTPUTS / "27_movies_db_genome_summary.csv")

display(rating_stats.sort_values("rating_count", ascending=False).head(10))
display(tag_stats.sort_values("user_tag_count", ascending=False).head(10))
display(genome_summary.head(10))

display(Markdown(
    f"**Interpretation.** The rating sample covers {rating_stats['movieId'].nunique():,} rated movies. "
    f"User tags cover {tag_stats['movieId'].nunique():,} movies, and genome tags provide compact semantic descriptors for "
    f"{genome_summary['movieId'].nunique():,} movies. These are useful explanatory fields for later audit slices."
))
"""
        ),
        new_markdown_cell(
            """## 6. Multimodal Feature Inventory

Feature coverage is a data-governance issue. If some modalities cover fewer movies, a multimodal recommender may encode catalogue groups unevenly. The table distinguishes MovieLens-ID JSON features from M3L internal matrices mapped back to MovieLens IDs.
"""
        ),
        new_code_cell(
            """feature_inventory = pd.read_csv(OUTPUTS / "27_movies_db_feature_inventory.csv")
coverage_report = pd.read_csv(OUTPUTS / "27_movies_db_coverage_report.csv")

display(feature_inventory)
display(coverage_report.round(4))
display(Image(filename=str(OUTPUTS / "27_movies_db_coverage.png")))

text_coverage = coverage_report.loc[coverage_report["field"].eq("has_text_mpnet_json"), "coverage_rate"]
image_coverage = coverage_report.loc[coverage_report["field"].eq("has_image_clip_image_json"), "coverage_rate"]
text_value = float(text_coverage.iloc[0]) if len(text_coverage) else 0.0
image_value = float(image_coverage.iloc[0]) if len(image_coverage) else 0.0

display(Markdown(
    f"**Interpretation.** MPNet JSON text features cover {text_value:.1%} of MovieLens movies and CLIP-image JSON features cover "
    f"{image_value:.1%}. The M3L interaction matrices cover the smaller recommender item universe rather than the full MovieLens catalogue."
))
"""
        ),
        new_markdown_cell(
            """## 7. Combined Movie-Level Database

The persisted database is the central artifact for downstream analysis. It keeps identifiers, title/year parsing, ratings, tags, genome descriptors, M3L feature coverage, Wikidata cultural metadata and audit labels in one row per MovieLens movie.
"""
        ),
        new_code_cell(
            """combined_movies = pd.read_csv(DATA_PROCESSED / "combined_movies_db.csv")
summary_stats = pd.read_csv(OUTPUTS / "27_movies_db_summary_stats.csv")

display(summary_stats)
display(combined_movies.head())

display(Markdown(
    f"**Interpretation.** The combined Movies DB has {len(combined_movies):,} movie rows and {combined_movies.shape[1]:,} columns. "
    f"{int(summary_stats.loc[summary_stats['metric'].eq('movies with Wikidata match'), 'value'].iloc[0]):,} movies have a Wikidata match in the current cache. "
    "This table is the right basis for cultural labels, metadata coverage checks and audit slices."
))
"""
        ),
        new_markdown_cell(
            """## 8. Rating and Popularity Distributions

These plots explain why long-tail and popularity-sensitive evaluation are necessary. Movie ratings are not evenly distributed across the catalogue.
"""
        ),
        new_code_cell(
            """display(Image(filename=str(OUTPUTS / "28_movies_db_rating_distribution.png")))

display(Markdown(
    "**Interpretation.** Mean ratings are concentrated in the middle-to-high range, while rating counts are heavily skewed. "
    "That skew is exactly why catalogue presence and recommendation visibility must be analysed separately."
))
"""
        ),
        new_markdown_cell(
            """## 9. Audit-Oriented Dataset Discovery

Before model outputs exist, genre and decade slices are only diagnostic leads. They compare catalogue availability, observed user interest and simple popularity/weighted-rating Top-K baselines.
"""
        ),
        new_code_cell(
            """genre_audit = pd.read_csv(OUTPUTS / "27_movies_db_genre_audit.csv")
decade_audit = pd.read_csv(OUTPUTS / "27_movies_db_decade_audit.csv")

display(genre_audit.round(4))
display(decade_audit.round(4))
display(Image(filename=str(OUTPUTS / "29_movies_db_genre_interest_visibility.png")))

largest_gap = genre_audit.sort_values("interest_minus_weighted_visibility", ascending=False).iloc[0]
display(Markdown(
    f"**Interpretation.** The largest positive genre lead versus the weighted Top-100 baseline is **{largest_gap['genre']}** "
    f"({largest_gap['interest_minus_weighted_visibility']:+.1%}). This is an audit lead, not a finding about a trained recommender."
))
"""
        ),
        new_markdown_cell(
            """## 10. User-Oriented Dataset Discovery

Prominence targets should be adjusted for user interest. If a small group of heavy users dominates ratings, aggregate interest can become misleading.
"""
        ),
        new_code_cell(
            """user_summary = pd.read_csv(OUTPUTS / "27_movies_db_user_summary.csv")
user_concentration = pd.read_csv(OUTPUTS / "27_movies_db_user_concentration.csv")

display(user_summary)
display(user_concentration.round(4))
display(Image(filename=str(OUTPUTS / "30_movies_db_user_concentration.png")))

top5 = user_concentration.loc[user_concentration["user_group"].eq("top 5% most active users"), "rating_share"].iloc[0]
display(Markdown(
    f"**Interpretation.** The top 5% most active users contribute {top5:.1%} of the loaded ratings. "
    "This supports preference-adjusted prominence metrics rather than pure catalogue-share targets."
))
"""
        ),
        new_markdown_cell(
            """## 11. What This Changes for the Project

The Movies DB notebook becomes the first pipeline layer:

1. **Movies DB:** combine and validate the movie catalogue.
2. **Audit metadata:** define European, non-English and long-tail labels with coverage caveats.
3. **Recommendation models:** train/evaluate Top-K rankings on real M3L interactions.
4. **Governance dashboard:** compare utility, exposure, PACPG and re-ranking trade-offs.

This separation makes the project easier to defend in class: the data foundation is auditable before any model result is discussed.
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
