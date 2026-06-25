from __future__ import annotations

import ast
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED = PROJECT_ROOT / "data" / "processed"
FINAL_TABLES = PROJECT_ROOT / "cultural_prominence_audit" / "outputs" / "final_notebook_tables"
FINAL_FIGURES = PROJECT_ROOT / "cultural_prominence_audit" / "outputs" / "final_notebook_figures"
SUBMISSION_ASSETS = PROJECT_ROOT / "cultural_prominence_audit" / "outputs" / "final_submission_assets"

SUBMISSION_ASSETS.mkdir(parents=True, exist_ok=True)
FINAL_FIGURES.mkdir(parents=True, exist_ok=True)


def parse_list_cell(value) -> list[str]:
    """Parse Wikidata list-like CSV cells without inventing missing metadata."""
    if pd.isna(value):
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "[]"}:
        return []
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, (list, tuple, set)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass
    return [part.strip() for part in text.replace("|", ",").split(",") if part.strip()]


def explode_multilabel(df: pd.DataFrame, list_col: str, out_col: str) -> pd.DataFrame:
    expanded = df.copy()
    expanded[out_col] = expanded[list_col].apply(parse_list_cell)
    expanded = expanded.explode(out_col)
    return expanded[expanded[out_col].notna() & expanded[out_col].astype(str).ne("")]


def split_genres(value) -> list[str]:
    if pd.isna(value):
        return ["Unknown genre"]
    genres = [g.strip() for g in str(value).split("|") if g.strip()]
    return genres or ["Unknown genre"]


def save_barh(series: pd.Series, title: str, xlabel: str, path: Path, color: str = "#1f8a8a"):
    fig, ax = plt.subplots(figsize=(10.5, max(4.5, 0.32 * len(series))))
    series.sort_values().plot(kind="barh", ax=ax, color=color)
    ax.set_title(title, fontsize=15, fontweight="bold")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("")
    ax.grid(axis="x", alpha=0.25)
    for i, value in enumerate(series.sort_values()):
        label = f"{value:.1%}" if abs(value) <= 1 else f"{value:,.0f}"
        ax.text(value, i, f" {label}", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def build_country_genre_assets(combined: pd.DataFrame):
    country_movies = explode_multilabel(combined, "country", "country_label")
    country_movies["genre"] = country_movies["genres"].apply(split_genres)
    country_genres = country_movies.explode("genre")

    country_genre_stats = (
        country_genres.groupby(["country_label", "genre"])
        .agg(
            movie_count=("movieId", "nunique"),
            median_rating_count=("rating_count", "median"),
            mean_rating=("rating_mean", "mean"),
            with_mpnet_share=("has_text_mpnet_matrix", "mean"),
            with_clip_image_share=("has_image_clip_image_matrix", "mean"),
        )
        .reset_index()
        .sort_values(["movie_count", "country_label"], ascending=[False, True])
    )
    country_genre_stats.to_csv(SUBMISSION_ASSETS / "country_genre_statistics.csv", index=False)

    top_countries = (
        country_movies.groupby("country_label")["movieId"]
        .nunique()
        .sort_values(ascending=False)
        .head(14)
        .index
    )
    top_genres = (
        country_genres.groupby("genre")["movieId"]
        .nunique()
        .sort_values(ascending=False)
        .head(12)
        .index
    )

    heat = (
        country_genres[country_genres["country_label"].isin(top_countries) & country_genres["genre"].isin(top_genres)]
        .pivot_table(index="country_label", columns="genre", values="movieId", aggfunc="nunique", fill_value=0)
        .reindex(index=top_countries, columns=top_genres)
    )
    heat_pct = heat.div(heat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(heat_pct.to_numpy(), cmap="YlGnBu", aspect="auto", vmin=0, vmax=max(0.01, heat_pct.to_numpy().max()))
    ax.set_title("Country-genre mix: which genres define each production-country proxy?", fontsize=15, fontweight="bold")
    ax.set_xlabel("MovieLens genre")
    ax.set_ylabel("Wikidata country of origin / production country")
    ax.set_xticks(range(len(heat_pct.columns)))
    ax.set_xticklabels(heat_pct.columns, rotation=35, ha="right")
    ax.set_yticks(range(len(heat_pct.index)))
    ax.set_yticklabels(heat_pct.index)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Share within selected country rows")
    fig.tight_layout()
    fig.savefig(FINAL_FIGURES / "22_country_genre_mix_heatmap.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    country_genre_leads = (
        country_genre_stats.sort_values(["country_label", "movie_count"], ascending=[True, False])
        .groupby("country_label")
        .head(3)
        .groupby("country_label")
        .agg(
            top_genres=("genre", lambda s: " | ".join(s.astype(str))),
            top_genre_movie_counts=("movie_count", lambda s: " | ".join(str(int(x)) for x in s)),
        )
        .reset_index()
    )
    return country_genre_stats, country_genre_leads


def build_geo_assets(country_genre_leads: pd.DataFrame):
    geo_path = FINAL_TABLES / "country_geo_outcome_table.csv"
    if not geo_path.exists():
        return pd.DataFrame()

    geo = pd.read_csv(geo_path)
    geo = geo.merge(country_genre_leads, left_on="group_name", right_on="country_label", how="left")
    geo = geo.drop(columns=[c for c in ["country_label"] if c in geo.columns])

    geo["gap_pp"] = geo["mean_gap_vs_target"] * 100
    geo["mean_exposure_pct"] = geo["mean_exposure"] * 100
    geo["catalogue_share_pct"] = geo["catalogue_share"] * 100
    geo["train_interaction_share_pct"] = geo["train_interaction_share"] * 100
    geo.to_csv(SUBMISSION_ASSETS / "country_problem_scorecard.csv", index=False)

    fig = px.choropleth(
        geo[~geo["group_name"].str.contains("Unknown", na=False)],
        locations="group_name",
        locationmode="country names",
        color="gap_pp",
        hover_name="group_name",
        hover_data={
            "mean_exposure_pct": ":.2f",
            "visibility_target": ":.3f",
            "support_catalogue_items": ":,",
            "top_genres": True,
            "gap_pp": ":.2f",
        },
        color_continuous_scale="RdBu",
        color_continuous_midpoint=0,
        title="Cultural prominence gap by production-country proxy",
        labels={
            "gap_pp": "Exposure gap vs target (pp)",
            "mean_exposure_pct": "Mean Exposure@20 (%)",
            "visibility_target": "Visibility target",
            "support_catalogue_items": "Catalogue support",
            "top_genres": "Top local genres",
        },
    )
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=20, r=20, t=70, b=20),
        coloraxis_colorbar=dict(title="PACPG-like gap (pp)"),
    )
    fig.write_html(SUBMISSION_ASSETS / "country_visibility_world_map.html", include_plotlyjs="cdn")

    problem = geo.sort_values("mean_gap_vs_target").head(10)
    save_barh(
        problem.set_index("group_name")["mean_gap_vs_target"],
        "Countries with the largest negative visibility gap",
        "Mean gap vs target",
        FINAL_FIGURES / "22_country_problem_ranking.png",
        "#b84a4a",
    )
    return geo


def build_language_country_and_company_assets(combined: pd.DataFrame):
    country_lang = explode_multilabel(combined, "country", "country_label")
    country_lang["language_label"] = country_lang["original_language"].apply(parse_list_cell)
    country_lang = country_lang.explode("language_label")
    country_lang = country_lang[country_lang["language_label"].notna() & country_lang["language_label"].astype(str).ne("")]
    country_lang_stats = (
        country_lang.groupby(["country_label", "language_label"])
        .agg(movie_count=("movieId", "nunique"), train_like_interactions=("rating_count", "sum"))
        .reset_index()
        .sort_values(["movie_count", "train_like_interactions"], ascending=False)
    )
    country_lang_stats.to_csv(SUBMISSION_ASSETS / "country_language_crosswalk.csv", index=False)

    # Production-company country is a caveat layer: it does not override origin,
    # but it tells us where an apparently local film may still involve US firms.
    company = combined.copy()
    company["origin_countries"] = company["country"].apply(parse_list_cell)
    company["company_countries"] = company["production_company_country"].apply(parse_list_cell)
    company["company_hq_countries"] = company["production_company_hq_country"].apply(parse_list_cell)
    company["has_us_origin"] = company["origin_countries"].apply(lambda xs: "United States" in xs)
    company["has_us_company"] = company.apply(
        lambda r: "United States" in set(r["company_countries"]) | set(r["company_hq_countries"]),
        axis=1,
    )
    company["has_european_origin"] = company["is_european"].fillna(False).astype(bool)
    company["is_non_english_flag"] = company["is_non_english"].fillna(False).astype(bool)

    caveat_summary = pd.DataFrame(
        [
            {
                "segment": "European-origin films with US production-company/HQ involvement",
                "movies": int((company["has_european_origin"] & company["has_us_company"]).sum()),
                "share_of_catalogue": float((company["has_european_origin"] & company["has_us_company"]).mean()),
            },
            {
                "segment": "Non-US-origin films with US production-company/HQ involvement",
                "movies": int((~company["has_us_origin"] & company["has_us_company"]).sum()),
                "share_of_catalogue": float((~company["has_us_origin"] & company["has_us_company"]).mean()),
            },
            {
                "segment": "Non-English films with US production-company/HQ involvement",
                "movies": int((company["is_non_english_flag"] & company["has_us_company"]).sum()),
                "share_of_catalogue": float((company["is_non_english_flag"] & company["has_us_company"]).mean()),
            },
            {
                "segment": "Films missing production-company country/HQ metadata",
                "movies": int((company["company_countries"].str.len().eq(0) & company["company_hq_countries"].str.len().eq(0)).sum()),
                "share_of_catalogue": float((company["company_countries"].str.len().eq(0) & company["company_hq_countries"].str.len().eq(0)).mean()),
            },
        ]
    )
    caveat_summary.to_csv(SUBMISSION_ASSETS / "production_company_caveat_summary.csv", index=False)
    return country_lang_stats, caveat_summary


def build_final_brief():
    answer_path = FINAL_TABLES / "final_answer_table.csv"
    if not answer_path.exists():
        return
    answers = pd.read_csv(answer_path)
    key_questions = [
        "Does the algorithm hide Europe?",
        "Which European countries are least visible?",
        "Which languages are most visible?",
        "Does the recommender show local Europe or globally compatible Europe?",
        "Does governance-aware re-ranking help?",
    ]
    selected = answers[answers["Question"].isin(key_questions)].copy()
    lines = [
        "# Final Findings Brief",
        "",
        "This file is generated from the executed final notebook outputs. It is intended as a quick oral-presentation briefing, not as a replacement for the notebook.",
        "",
    ]
    for _, row in selected.iterrows():
        lines.append(f"## {row['Question']}")
        lines.append("")
        lines.append(str(row["Short answer"]))
        lines.append("")
        lines.append(f"Evidence section: {row['Evidence in notebook section']}. Confidence: {row['Confidence']}. Caveat: {row['Main caveat']}.")
        lines.append("")
    (SUBMISSION_ASSETS / "final_findings_brief.md").write_text("\n".join(lines), encoding="utf-8")


def update_figure_ledger():
    ledger_path = FINAL_TABLES / "figure_ledger.csv"
    if not ledger_path.exists():
        return
    ledger = pd.read_csv(ledger_path)
    additions = pd.DataFrame(
        [
            {
                "figure": "22_country_genre_mix_heatmap.png",
                "section": "Final submission extension",
                "interpretation": "Shows whether country-level visibility gaps may be entangled with genre composition rather than only national origin.",
            },
            {
                "figure": "22_country_problem_ranking.png",
                "section": "Final submission extension",
                "interpretation": "Ranks the strongest negative country-level visibility gaps among support-passing countries.",
            },
            {
                "figure": "country_visibility_world_map.html",
                "section": "Final submission extension",
                "interpretation": "Interactive map of mean country prominence gap, used as supplementary HTML evidence.",
            },
        ]
    )
    combined = pd.concat([ledger, additions], ignore_index=True).drop_duplicates(subset=["figure"], keep="last")
    combined.to_csv(ledger_path, index=False)


def main():
    combined_path = PROCESSED / "combined_movies_db.csv"
    if not combined_path.exists():
        raise FileNotFoundError(f"Missing processed movie database: {combined_path}")

    combined = pd.read_csv(combined_path)
    country_genre_stats, country_genre_leads = build_country_genre_assets(combined)
    geo = build_geo_assets(country_genre_leads)
    country_lang_stats, caveat_summary = build_language_country_and_company_assets(combined)
    build_final_brief()
    update_figure_ledger()

    manifest = {
        "generated_assets_dir": str(SUBMISSION_ASSETS.relative_to(PROJECT_ROOT)),
        "country_genre_rows": int(len(country_genre_stats)),
        "country_problem_rows": int(len(geo)),
        "country_language_rows": int(len(country_lang_stats)),
        "production_company_caveat_rows": int(len(caveat_summary)),
        "note": "All assets are derived from real processed MovieLens/M3L/Wikidata data. No synthetic data is generated.",
    }
    (SUBMISSION_ASSETS / "asset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
