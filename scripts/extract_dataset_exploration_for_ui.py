from __future__ import annotations

import base64
import json
from pathlib import Path

import nbformat
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = PROJECT_ROOT / "outputs"
SOURCE_NOTEBOOK = Path("/Users/maxpriessnitz/Downloads/dataset_exploration_and_combination.ipynb")


FIGURE_CELLS = {
    34: {
        "key": "featureCoverage",
        "file": "15_exploration_feature_coverage.png",
        "title": "Movie-level feature coverage",
        "interpretation": (
            "The catalogue-level feature inventory shows that the multimodal audit is feasible: "
            "text and image features cover most movies, while audio/video coverage is more limited "
            "and therefore better treated as optional evidence rather than the core audit signal."
        ),
    },
    35: {
        "key": "ratingDistribution",
        "file": "16_exploration_rating_distribution.png",
        "title": "Rating distribution and popularity",
        "interpretation": (
            "The loaded ratings are highly uneven across movies. This supports the later long-tail "
            "definition and explains why exposure must be evaluated separately from accuracy."
        ),
    },
    37: {
        "key": "genreSummary",
        "file": "17_exploration_genre_summary.png",
        "title": "Genre-level catalogue and rating summary",
        "interpretation": (
            "Genre shares are not mutually exclusive, but they show whether the data contains enough "
            "variation to check if cultural under-exposure is hidden inside broad genre preferences."
        ),
    },
    38: {
        "key": "featureByGenre",
        "file": "18_exploration_feature_by_genre.png",
        "title": "Multimodal feature coverage by genre",
        "interpretation": (
            "Coverage is not only a technical property; it can become a bias source if certain genres "
            "receive weaker feature representations than others."
        ),
    },
    45: {
        "key": "genreAuditLeads",
        "file": "19_exploration_genre_audit_leads.png",
        "title": "Availability, interest and baseline visibility by genre",
        "interpretation": (
            "The genre audit is a diagnostic lead, not an empirical recommender result. It separates "
            "catalogue availability, observed user interest and simple Top-K visibility proxies."
        ),
    },
    47: {
        "key": "decadeAudit",
        "file": "20_exploration_decade_audit.png",
        "title": "Release-decade audit leads",
        "interpretation": (
            "Decades are mutually exclusive, so this view is useful for checking whether recommender "
            "visibility mostly follows recency and popularity rather than catalogue diversity."
        ),
    },
    49: {
        "key": "europeanCheck",
        "file": "21_exploration_european_check.png",
        "title": "European-film check from Wikidata sample",
        "interpretation": (
            "The European-film check demonstrates the label logic, but the small Wikidata sample in "
            "the exploration notebook is a coverage warning rather than a final result."
        ),
    },
    54: {
        "key": "userActivity",
        "file": "22_exploration_user_activity.png",
        "title": "User activity and rating behaviour",
        "interpretation": (
            "User histories differ strongly in length and variation. A prominence audit therefore "
            "needs user-level targets, not only global catalogue shares."
        ),
    },
    56: {
        "key": "activityConcentration",
        "file": "23_exploration_activity_concentration.png",
        "title": "Activity concentration",
        "interpretation": (
            "The most active users contribute a disproportionate share of ratings. This justifies "
            "checking whether aggregate-interest baselines are dominated by heavy users."
        ),
    },
    58: {
        "key": "userSegments",
        "file": "24_exploration_user_segments.png",
        "title": "User segments",
        "interpretation": (
            "Light, medium and heavy users can experience different visibility gaps. The final audit "
            "should therefore keep user-segment diagnostics available."
        ),
    },
    60: {
        "key": "userGenreInterest",
        "file": "25_exploration_user_genre_interest.png",
        "title": "User interest by genre",
        "interpretation": (
            "Genre reach shows whether observed preferences are broad enough to support a fair "
            "comparison between user interest and Top-K exposure."
        ),
    },
    62: {
        "key": "taggingBehaviour",
        "file": "26_exploration_tagging_behaviour.png",
        "title": "Tagging behaviour",
        "interpretation": (
            "Tags are sparse, but they add an explicit-attention signal that helps explain catalogue "
            "interpretation beyond numeric ratings."
        ),
    },
}


def _decode_image_payload(payload: str | list[str]) -> bytes:
    if isinstance(payload, list):
        payload = "".join(payload)
    return base64.b64decode(payload)


def extract_figures(nb) -> list[dict[str, str]]:
    # The source notebook is already executed; we reuse its rendered PNGs as derived evidence.
    figures: list[dict[str, str]] = []
    for cell_index, spec in FIGURE_CELLS.items():
        cell = nb.cells[cell_index]
        image_output = None
        for output in cell.get("outputs", []):
            data = output.get("data", {})
            if "image/png" in data:
                image_output = data["image/png"]
        if image_output is None:
            raise RuntimeError(f"Cell {cell_index} does not contain a PNG output.")

        target = OUTPUTS / spec["file"]
        target.write_bytes(_decode_image_payload(image_output))
        figures.append(
            {
                "key": spec["key"],
                "title": spec["title"],
                "path": f"../outputs/{spec['file']}",
                "cell": cell_index,
                "interpretation": spec["interpretation"],
            }
        )
    return figures


def build_summary(figures: list[dict[str, str]]) -> dict:
    # These values come from the executed exploration notebook outputs and are kept explicit
    # so the UI can explain provenance without rerunning Nico's local directory layout.
    all_feature_share = 19_227 / 27_278
    summary = {
        "sourceNotebook": str(SOURCE_NOTEBOOK),
        "sourceStatus": "Executed local notebook provided by the project group; outputs are used as derived evidence only.",
        "ratingSampleNote": (
            "The imported exploration notebook used RATING_NROWS = 1,000,000 to keep the exploratory run responsive. "
            "These values are therefore dataset-foundation evidence, not final recommender-model results."
        ),
        "kpis": [
            {
                "label": "Catalogue movies",
                "value": 27_278,
                "display": "27,278",
                "note": "MovieLens-style catalogue size in the local exploration notebook.",
            },
            {
                "label": "Archive-M3L20 overlap",
                "value": 1.0,
                "display": "100.0%",
                "note": "The shared movieId bridge is complete between the archive catalogue and M3L-20M metadata.",
            },
            {
                "label": "Archive-M3L10 overlap",
                "value": 0.391459,
                "display": "39.1%",
                "note": "M3L-10M is a smaller slice and should not be treated as catalogue-complete for this audit.",
            },
            {
                "label": "Title mismatches",
                "value": 183,
                "display": "183",
                "note": "Title mismatches make movieId safer than title matching for joins.",
            },
            {
                "label": "All-feature coverage",
                "value": all_feature_share,
                "display": f"{all_feature_share:.1%}",
                "note": "Movies with all 12 local M3L feature families in the exploration notebook.",
            },
            {
                "label": "Loaded rating users",
                "value": 6_743,
                "display": "6,743",
                "note": "Users in the loaded 1M-rating exploration sample.",
            },
            {
                "label": "Top 5% rating share",
                "value": 0.302,
                "display": "30.2%",
                "note": "Heavy-user concentration in the exploration sample.",
            },
        ],
        "tabs": {
            "integration": {
                "label": "Movie-level integration",
                "claim": (
                    "The exploration notebook supports MovieLens movieId as the primary integration key. "
                    "The archive and M3L-20M catalogues overlap completely by movieId, while title mismatches show "
                    "why title-based joins would be fragile."
                ),
                "evidence": [
                    "Archive movies: 27,278 rows and 27,278 unique movieId values.",
                    "Archive versus M3L-20M intersection: 27,278 movies; Jaccard overlap: 1.000.",
                    "Archive versus M3L-10M Jaccard overlap: 0.391, which confirms that M3L-10M is a smaller slice.",
                    "183 title mismatches between archive and M3L-20M make identifier-based joins necessary.",
                ],
                "methodologicalImplication": (
                    "For the final audit, movieId is used as the internal key and IMDb/Wikidata identifiers are "
                    "used only for cultural metadata enrichment. This reduces avoidable join bias."
                ),
            },
            "features": {
                "label": "Feature coverage",
                "claim": (
                    "M3L provides enough multimodal coverage for text- and image-aware recommenders, but feature "
                    "coverage itself must be reported because missing modalities can become a governance issue."
                ),
                "evidence": [
                    "Text features: CLIP-text, MiniLM and MPNet for 25,177 movies.",
                    "Image features: CLIP-image, VGG and ViT for 25,037 movies.",
                    "Audio/video feature families are available for roughly 19.3k movies.",
                    "19,227 movies, or 70.5% of the catalogue, have all 12 feature families in the local exploration.",
                ],
                "methodologicalImplication": (
                    "The UI keeps MPNet text and CLIP-image as core proposal models and treats audio/video as "
                    "possible extensions after coverage and runtime checks."
                ),
            },
            "users": {
                "label": "User behaviour",
                "claim": (
                    "User activity is concentrated and heterogeneous, so prominence cannot be judged only against "
                    "global catalogue shares."
                ),
                "evidence": [
                    "The loaded exploration sample contains 6,743 users, 1,000,000 rating events and 13,950 rated movies.",
                    "Median ratings per user: 70; mean ratings per user: 148.3; 90th percentile: 355.",
                    "The top 5% most active users contribute 30.2% of loaded ratings.",
                    "38.1% of users have a rating standard deviation of at least 1.0.",
                ],
                "methodologicalImplication": (
                    "PACPG compares ranking exposure against observed user interest and relevant test data, which "
                    "is more defensible than a pure catalogue-quota target."
                ),
            },
            "auditLeads": {
                "label": "Audit leads",
                "claim": (
                    "The imported charts identify where visibility gaps may appear, but they do not claim final "
                    "algorithmic bias before the real recommender outputs are evaluated."
                ),
                "evidence": [
                    "Availability, interest and Top-K proxy visibility are explicitly separated.",
                    "Genre-level views are diagnostic because MovieLens genres overlap.",
                    "Release-decade views help detect whether visibility follows recency/popularity concentration.",
                    "The small Wikidata country sample demonstrates the European-label pipeline and its coverage risk.",
                ],
                "methodologicalImplication": (
                    "The final UI should read results as detect -> compare -> mitigate -> interpret, with missingness "
                    "and proxy risk visible beside the performance metrics."
                ),
            },
        },
        "figures": figures,
        "references": [
            {
                "label": "MovieLens / GroupLens",
                "url": "https://files.grouplens.org/datasets/movielens/ml-20m-README.html",
            },
            {
                "label": "M3L-20M / Binge Watch Zenodo record",
                "url": "https://zenodo.org/records/18499145",
            },
            {
                "label": "Wikidata data access",
                "url": "https://www.wikidata.org/wiki/Wikidata:Data_access",
            },
            {
                "label": "Markus Schedl profile",
                "url": "https://www.mschedl.eu/",
            },
        ],
    }
    return summary


def write_notes(summary: dict) -> None:
    lines = [
        "# Dataset Exploration Integration Notes",
        "",
        f"Source notebook: `{summary['sourceNotebook']}`",
        "",
        summary["ratingSampleNote"],
        "",
        "## Key Evidence",
        "",
    ]
    for tab in summary["tabs"].values():
        lines.extend(
            [
                f"### {tab['label']}",
                "",
                tab["claim"],
                "",
                "Evidence:",
            ]
        )
        lines.extend([f"- {item}" for item in tab["evidence"]])
        lines.extend(["", f"Methodological implication: {tab['methodologicalImplication']}", ""])

    (OUTPUTS / "dataset_exploration_source_notes.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    if not SOURCE_NOTEBOOK.exists():
        raise FileNotFoundError(f"Missing source notebook: {SOURCE_NOTEBOOK}")

    OUTPUTS.mkdir(exist_ok=True)
    nb = nbformat.read(SOURCE_NOTEBOOK, as_version=4)
    figures = extract_figures(nb)
    summary = build_summary(figures)

    (OUTPUTS / "dataset_exploration_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(summary["kpis"]).to_csv(OUTPUTS / "dataset_exploration_kpis.csv", index=False)
    pd.DataFrame(figures).to_csv(OUTPUTS / "dataset_exploration_figures.csv", index=False)
    write_notes(summary)
    print(f"Extracted {len(figures)} figures and wrote dataset exploration summary.")


if __name__ == "__main__":
    main()
