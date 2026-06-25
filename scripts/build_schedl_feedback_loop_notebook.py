from __future__ import annotations

from pathlib import Path
import textwrap

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = PROJECT_ROOT / "notebooks"
NOTEBOOKS.mkdir(exist_ok=True)


def md(text: str):
    return nbf.v4.new_markdown_cell(textwrap.dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(textwrap.dedent(text).strip())


def schedl_cells() -> list:
    return [
        md(
            """
            # Schedl-Style Feedback-Loop Audit
            ## Country, language and popularity calibration in movie recommendations

            This notebook extends the cultural prominence audit with the feedback-loop logic from Lesota, Geiger, Walder, Kowald and Schedl (2024), *Oh, Behave! Country Representation Dynamics Created by Feedback Loops in Music Recommender Systems*.

            The original paper studies local and US-produced music. Our MovieLens/M3L setting has no user-country field, so we adapt the logic carefully: **local** becomes the user's observed cultural profile in their initial movie history. A recommender is culturally miscalibrated when its Top-K country/language/popularity distribution drifts away from that profile.
            """
        ),
        md(
            """
            ## What the feedback loop does

            1. Train a recommender on the current user profiles.
            2. Recommend Top-10 unseen movies per user.
            3. Simulate one accepted item per user with rank-biased acceptance probability.
            4. Add the accepted item to the user profile.
            5. Retrain and repeat.

            This is not a replacement for the Top-K audit. It adds the missing dynamic question from the Schedl paper: **does repeated recommendation push user profiles toward or away from European, non-English and non-US content over time?**
            """
        ),
        code(
            """
            from pathlib import Path
            import pandas as pd
            import matplotlib.pyplot as plt
            from IPython.display import Image, Markdown, display

            PROJECT_ROOT = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
            OUTPUTS = PROJECT_ROOT / "outputs"

            run_report = pd.read_csv(OUTPUTS / "31_feedback_loop_run_report.csv")
            model_ledger = pd.read_csv(OUTPUTS / "31_schedl_model_ledger.csv")
            final_summary = pd.read_csv(OUTPUTS / "31_feedback_loop_final_summary.csv")
            iteration_metrics = pd.read_csv(OUTPUTS / "31_feedback_loop_iteration_metrics.csv")

            display(run_report)
            display(model_ledger)
            """
        ),
        code(
            """
            # Focus the table on the cultural shifts that map directly to the project question.
            view_cols = [
                "Model",
                "initial_european_share",
                "recommendation_european_share",
                "recommendation_european_shift",
                "initial_non_english_share",
                "recommendation_non_english_share",
                "recommendation_non_english_shift",
                "initial_us_origin_share",
                "recommendation_us_origin_share",
                "recommendation_us_origin_shift",
                "recommendation_origin_jsd",
                "recommendation_language_jsd",
                "recommendation_popularity_jsd",
            ]
            feedback_table = final_summary[view_cols].sort_values("recommendation_origin_jsd", ascending=False)
            display(feedback_table.round(4))

            strongest_europe_drop = feedback_table.loc[feedback_table["recommendation_european_shift"].idxmin()]
            strongest_non_english_drop = feedback_table.loc[feedback_table["recommendation_non_english_shift"].idxmin()]
            strongest_us_gain = feedback_table.loc[feedback_table["recommendation_us_origin_shift"].idxmax()]
            highest_origin_jsd = feedback_table.loc[feedback_table["recommendation_origin_jsd"].idxmax()]
            highest_popularity_jsd = feedback_table.loc[feedback_table["recommendation_popularity_jsd"].idxmax()]

            interpretation = (
                f\"**Feedback-loop interpretation.** After the simulated loop, **{strongest_europe_drop['Model']}** creates the largest European-origin drop \"
                f\"({strongest_europe_drop['recommendation_european_shift']:+.1%} against users' initial histories). \"
                f\"**{strongest_non_english_drop['Model']}** creates the largest non-English drop \"
                f\"({strongest_non_english_drop['recommendation_non_english_shift']:+.1%}). \"
                f\"The strongest US-origin gain appears for **{strongest_us_gain['Model']}** \"
                f\"({strongest_us_gain['recommendation_us_origin_shift']:+.1%}). \"
                f\"Country/origin miscalibration is highest for **{highest_origin_jsd['Model']}** \"
                f\"(JSD {highest_origin_jsd['recommendation_origin_jsd']:.3f}), while popularity miscalibration is highest for \"
                f\"**{highest_popularity_jsd['Model']}** (JSD {highest_popularity_jsd['recommendation_popularity_jsd']:.3f}).\"
            )
            display(Markdown(interpretation))
            """
        ),
        code(
            """
            for filename, caption in [
                ("31_feedback_loop_representation_dynamics.png", "European and non-English recommendation shares over feedback-loop iterations."),
                ("32_feedback_loop_jsd_heatmap.png", "Origin, language and popularity miscalibration after the simulation."),
                ("33_feedback_loop_final_shift.png", "Final Top-K composition shift relative to initial user histories."),
                ("34_language_country_bias_panels.png", "Country and language bias panels for the final feedback-loop state."),
            ]:
                display(Markdown(f"### {caption}"))
                display(Image(filename=str(OUTPUTS / filename)))
            """
        ),
        md(
            """
            ## Pitch-slide implementation checklist

            | Pitch requirement | Implemented as |
            |---|---|
            | Load -> Enrich -> Train -> Audit -> Mitigate | Main pipeline plus this dynamic feedback-loop extension |
            | Cultural labels | European origin, non-English original language, long-tail, US origin, US-company involvement, co-production and multilingual flags |
            | Models and metrics | Pop, ItemKNN, BPR-MF, LightGCN-style, NeuMF-lite, MultiVAE-lite; shifts and JSD miscalibration |
            | PACPG / prominence framing | Static Top-K audit still reports PACPG; feedback loop adds profile drift and country/language JSD |
            | Visual audit design | Representation dynamics, JSD heatmap, final shift chart, country/language panels |
            | Re-ranking | Still handled in the main Top-K audit; this notebook clarifies the dynamic risk before mitigation |
            """
        ),
        md(
            """
            ## Interpretation caveats

            - MovieLens/M3L does not contain user nationality. We therefore do **not** claim user-local country effects in the literal Schedl sense.
            - We adapt the logic to user-history calibration: recommendation distributions are compared to each user's initial movie profile.
            - LightGCN, NeuMF and MultiVAE are lightweight local variants, because the RecBole stack used in the paper is not installed in this workspace.
            - The strict long-tail label is almost absent in the sampled feedback-loop histories. For this dynamic section, country/language/US-origin and popularity-bin JSD are more informative than the binary long-tail share.

            ## Next steps

            1. Increase iterations from 6 toward 20-100 when runtime allows.
            2. Run RecBole implementations of BPR, LightGCN, NeuMF and MultiVAE for exact comparability with Lesota et al. (2024).
            3. Add a mitigation loop: apply the transparent re-ranker inside each iteration and compare drift against the unmitigated loop.
            4. Report binary and fractional co-production counting side by side.
            """
        ),
        md(
            """
            ## References

            - Lesota, O., Geiger, J., Walder, M., Kowald, D., & Schedl, M. (2024). *Oh, Behave! Country Representation Dynamics Created by Feedback Loops in Music Recommender Systems*. RecSys 2024 / arXiv:2408.11565.
            - Mansoury et al. (2020). *Feedback Loop and Bias Amplification in Recommender Systems*.
            - Steck (2018). *Calibrated Recommendations*.
            - Wikidata properties P495, P364, P407 and P272 for country, language and production-company metadata.
            """
        ),
    ]


def write_standalone_notebook() -> Path:
    nb = nbf.v4.new_notebook()
    nb["cells"] = schedl_cells()
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    target = NOTEBOOKS / "schedl_feedback_loop_audit.ipynb"
    nbf.write(nb, target)
    return target


def append_to_main_notebook() -> Path:
    target = NOTEBOOKS / "does_algorithm_hide_europe_realdata.ipynb"
    nb = nbf.read(target, as_version=4)
    marker = "Schedl-Style Feedback-Loop Audit"
    if any(marker in "".join(cell.get("source", "")) for cell in nb["cells"]):
        return target
    cells = [
        md(
            """
            ## Schedl-Style Feedback-Loop Audit

            The proposal deck promises a model-level cultural prominence audit. We now add a dynamic extension inspired by Lesota et al. and Markus Schedl (2024): recommendation, simulated acceptance, profile update and retraining over repeated iterations.
            """
        ),
        code(
            """
            schedl_final = pd.read_csv(OUTPUTS / "31_feedback_loop_final_summary.csv")
            schedl_view = schedl_final[[
                "Model",
                "recommendation_european_shift",
                "recommendation_non_english_shift",
                "recommendation_us_origin_shift",
                "recommendation_origin_jsd",
                "recommendation_language_jsd",
                "recommendation_popularity_jsd",
            ]].sort_values("recommendation_origin_jsd", ascending=False)
            display(schedl_view.round(4))

            worst_europe = schedl_view.loc[schedl_view["recommendation_european_shift"].idxmin()]
            worst_language = schedl_view.loc[schedl_view["recommendation_non_english_shift"].idxmin()]
            display(Markdown(
                f"**Schedl-style interpretation.** The dynamic loop shows the strongest European-origin drop for "
                f"**{worst_europe['Model']}** ({worst_europe['recommendation_european_shift']:+.1%}) and the strongest "
                f"non-English drop for **{worst_language['Model']}** ({worst_language['recommendation_non_english_shift']:+.1%}). "
                "Because MovieLens has no user-country field, this is a user-history calibration result, not a literal user-local-country claim."
            ))
            """
        ),
        code(
            """
            for filename in [
                "31_feedback_loop_representation_dynamics.png",
                "32_feedback_loop_jsd_heatmap.png",
                "33_feedback_loop_final_shift.png",
                "34_language_country_bias_panels.png",
            ]:
                display(Image(filename=str(OUTPUTS / filename)))
            """
        ),
        md(
            """
            **Methodological caveat.** The original Schedl paper uses user country and artist/track country. MovieLens/M3L does not include user country, so this project uses the user's initial movie-history distribution as the calibration target. This keeps the governance logic but avoids overstating the metadata.
            """
        ),
    ]
    nb["cells"].extend(cells)
    nbf.write(nb, target)
    return target


def main() -> None:
    standalone = write_standalone_notebook()
    main_nb = append_to_main_notebook()
    print(f"Wrote {standalone}")
    print(f"Updated {main_nb}")


if __name__ == "__main__":
    main()
