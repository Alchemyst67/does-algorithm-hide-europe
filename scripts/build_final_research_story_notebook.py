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


cells = [
    md(
        """
        # Does the Algorithm Hide Europe?
        ## Which Europe Gets Recommended? A country, language and visibility audit of multimodal movie recommender systems

        **Team:** Max Priessnitz & Nico  
        **Course:** Data Science and Artificial Intelligence II: Data and Algorithmic Governance  
        **Notebook role:** final executable research story

        This notebook is the main deliverable. It starts from local raw data and cached metadata, builds the analysis data, trains feasible recommender models, computes utility and cultural visibility metrics, and ends with a governance interpretation. Existing aggregate CSV outputs are not used as primary analysis input.
        """
    ),
    md(
        """
        ## 0. Setup, imports and configuration

        **Question.** Which run mode, paths and assumptions are used?

        The notebook defaults to a bounded local run so it can be executed on a laptop. All sample restrictions are recorded in the run manifest. Set `FULL_RUN = True` and increase `SAMPLE_USERS`/candidate limits for a heavier run.
        """
    ),
    code(
        """
        from pathlib import Path
        import os
        import sys
        import json
        import math
        import time
        import platform
        import random
        import re
        import ast
        from collections import Counter, defaultdict
        from itertools import chain

        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt

        from scipy import sparse
        from scipy.spatial.distance import jensenshannon
        from scipy.stats import entropy as scipy_entropy
        from sklearn.decomposition import TruncatedSVD
        from sklearn.metrics.pairwise import cosine_similarity
        from sklearn.preprocessing import normalize, StandardScaler, MultiLabelBinarizer
        from sklearn.neighbors import NearestNeighbors
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import KFold

        try:
            from IPython.display import display, Markdown
        except Exception:
            display = print
            Markdown = lambda x: x

        RANDOM_SEED = 42
        K_VALUES = [5, 10, 20, 50]
        MAIN_K = 20
        CV_FOLDS = 3
        FULL_RUN = False
        SAMPLE_USERS = 400 if not FULL_RUN else 10000
        CV_SAMPLE_USERS = 240 if not FULL_RUN else 3000
        FEEDBACK_SAMPLE_USERS = 180 if not FULL_RUN else 2500
        MAX_RATINGS_FOR_DESCRIPTIVE = 1_000_000 if not FULL_RUN else None
        MAX_CANDIDATE_ITEMS = 1200 if not FULL_RUN else 12000
        CANDIDATE_K = 200
        MIN_COUNTRY_ITEMS = 30
        MIN_LANGUAGE_ITEMS = 30
        MIN_GROUP_INTERACTIONS = 500
        FEEDBACK_ITERATIONS = 10
        LAMBDAS = [round(x / 10, 1) for x in range(0, 11)]
        RUN_WIKIDATA_ENRICHMENT = False
        RUN_CROSS_VALIDATION = True
        RUN_FEEDBACK_LOOP = True

        random.seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        PROJECT_ROOT = Path.cwd().resolve()
        if PROJECT_ROOT.name == "notebooks":
            PROJECT_ROOT = PROJECT_ROOT.parent

        NOTEBOOK_OUT = PROJECT_ROOT / "cultural_prominence_audit" / "outputs"
        FIG_DIR = NOTEBOOK_OUT / "final_notebook_figures"
        TABLE_DIR = NOTEBOOK_OUT / "final_notebook_tables"
        for p in [FIG_DIR, TABLE_DIR]:
            p.mkdir(parents=True, exist_ok=True)

        figure_ledger = []
        model_run_ledger = []
        filter_threshold_ledger = []
        sanity_checks = []

        config = {
            "FULL_RUN": FULL_RUN,
            "SAMPLE_USERS": SAMPLE_USERS,
            "CV_FOLDS": CV_FOLDS,
            "CV_SAMPLE_USERS": CV_SAMPLE_USERS,
            "FEEDBACK_SAMPLE_USERS": FEEDBACK_SAMPLE_USERS,
            "MAX_RATINGS_FOR_DESCRIPTIVE": MAX_RATINGS_FOR_DESCRIPTIVE,
            "MAX_CANDIDATE_ITEMS": MAX_CANDIDATE_ITEMS,
            "K_VALUES": K_VALUES,
            "MAIN_K": MAIN_K,
            "MIN_COUNTRY_ITEMS": MIN_COUNTRY_ITEMS,
            "MIN_LANGUAGE_ITEMS": MIN_LANGUAGE_ITEMS,
            "MIN_GROUP_INTERACTIONS": MIN_GROUP_INTERACTIONS,
            "FEEDBACK_ITERATIONS": FEEDBACK_ITERATIONS,
            "LAMBDAS": LAMBDAS,
            "RUN_WIKIDATA_ENRICHMENT": RUN_WIKIDATA_ENRICHMENT,
            "RUN_CROSS_VALIDATION": RUN_CROSS_VALIDATION,
            "RUN_FEEDBACK_LOOP": RUN_FEEDBACK_LOOP,
        }

        runtime_info = pd.DataFrame([
            {"field": "python", "value": sys.version.split()[0]},
            {"field": "platform", "value": platform.platform()},
            {"field": "project_root", "value": str(PROJECT_ROOT)},
            {"field": "run_mode", "value": "FULL_RUN" if FULL_RUN else "bounded local run"},
            {"field": "figures_dir", "value": str(FIG_DIR)},
            {"field": "tables_dir", "value": str(TABLE_DIR)},
        ])

        display(runtime_info)
        display(pd.DataFrame({"config": config.keys(), "value": [str(v) for v in config.values()]}))
        """
    ),
    code(
        """
        def save_table(df, filename):
            path = TABLE_DIR / filename
            df.to_csv(path, index=False)
            return path

        def save_json(obj, filename):
            path = TABLE_DIR / filename
            with path.open("w", encoding="utf-8") as f:
                json.dump(obj, f, indent=2, default=str)
            return path

        def save_current_figure(filename, section, research_question, source_table, description, caveat=""):
            path = FIG_DIR / filename
            plt.tight_layout()
            plt.savefig(path, dpi=180, bbox_inches="tight")
            figure_ledger.append({
                "figure_file": str(path.relative_to(PROJECT_ROOT)),
                "notebook_section": section,
                "research_question": research_question,
                "source_table": source_table,
                "description": description,
                "caveat": caveat,
            })
            plt.show()
            return path

        def parse_title_year(title):
            if pd.isna(title):
                return pd.Series({"clean_title": np.nan, "release_year": np.nan})
            match = re.search(r"\\((\\d{4})\\)\\s*$", str(title))
            year = int(match.group(1)) if match else np.nan
            clean = re.sub(r"\\s*\\(\\d{4}\\)\\s*$", "", str(title)).strip()
            return pd.Series({"clean_title": clean, "release_year": year})

        def imdb_numeric_to_tt(value):
            if pd.isna(value):
                return pd.NA
            try:
                return "tt" + str(int(value)).zfill(7)
            except Exception:
                return pd.NA

        def split_pipe(value):
            if value is None or (isinstance(value, float) and np.isnan(value)) or pd.isna(value):
                return []
            if isinstance(value, list):
                return [str(x).strip() for x in value if str(x).strip()]
            txt = str(value)
            if txt.startswith("[") and txt.endswith("]"):
                try:
                    parsed = ast.literal_eval(txt)
                    if isinstance(parsed, list):
                        return [str(x).strip() for x in parsed if str(x).strip()]
                except Exception:
                    pass
            return [part.strip() for part in re.split(r"\\s*\\|\\s*", txt) if part.strip() and part.strip().lower() != "nan"]

        def safe_div(a, b):
            return np.nan if b == 0 else a / b

        def add_check(name, passed, details):
            sanity_checks.append({"check": name, "passed": bool(passed), "details": str(details)})

        def barh(series, title, xlabel, filename, section, rq, source_table, caveat="", color="#167f86"):
            fig, ax = plt.subplots(figsize=(9, max(4, 0.32 * len(series))))
            series.sort_values().plot(kind="barh", ax=ax, color=color)
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("")
            return save_current_figure(filename, section, rq, source_table, title, caveat)

        def fmt_pct(x):
            return "n/a" if pd.isna(x) else f"{x:.1%}"

        def fmt_pp(x):
            return "n/a" if pd.isna(x) else f"{x:+.1%}"

        def result_box(title, bullets, caveat=""):
            body = f"**{title}**\\n\\n" + "\\n".join(f"- {bullet}" for bullet in bullets)
            if caveat:
                body += f"\\n\\n*Caveat:* {caveat}"
            display(Markdown(body))

        def label_points(ax, data, x_col, y_col, label_col, max_labels=24, fontsize=8):
            # Country plots need readable labels, but too many labels hide the signal.
            # The caller passes an already sorted frame so the most important points are labelled first.
            if data.empty:
                return
            for _, row in data.head(max_labels).iterrows():
                ax.annotate(
                    str(row[label_col]),
                    (row[x_col], row[y_col]),
                    xytext=(4, 3),
                    textcoords="offset points",
                    fontsize=fontsize,
                    color="#0b1f33",
                )
        """
    ),
    md(
        """
        ## 1. Abstract and research questions

        **Question.** What exactly is this notebook trying to answer?

        **Abstract.** Video-on-demand platforms can satisfy catalogue-level diversity goals while recommendation rankings still concentrate attention on already popular, English-language or US-linked films. This notebook audits cultural visibility in movie recommender systems using MovieLens/M3L interactions, MovieLens identifiers, M3L text/image features and cached Wikidata metadata. The audit treats country and language as proxy labels and evaluates not only accuracy, but ranked exposure, preference-adjusted cultural prominence gaps, intra-European visibility, Spain-origin versus Spanish-language distinctions, feedback-loop drift and re-ranking trade-offs.

        **Main RQ.** How do movie recommender systems transform European catalogue diversity into ranked visibility across countries and languages, and do multimodal features or governance-aware re-ranking mitigate or amplify these visibility gaps?

        **Sub-RQs.**
        1. Is Europe visible in the catalogue, interactions and recommendation rankings?
        2. Which European countries and regions are most and least visible?
        3. Which original languages are most and least visible?
        4. Are Spain-origin films and Spanish-language films proportionally visible?
        5. Do recommender models differ in cultural visibility?
        6. Do multimodal text/image features improve or worsen visibility?
        7. Are visibility gaps explained by popularity, language, US involvement, co-production, genre or release year?
        8. Do feedback loops amplify country/language visibility gaps?
        9. Can transparent re-ranking improve cultural visibility without unacceptable utility loss?
        10. What should platforms report for governance?

        The notebook first answers simple descriptive questions before moving to models: how many films, countries, languages, Spanish films, Spanish-language films and low-support groups exist in the local data.

        **Thesis storyline.** The argument is built like a small empirical thesis:

        1. **Data foundation:** what catalogue, interaction, feature and metadata coverage exists?
        2. **Operationalisation:** how do we define Europe, countries, language and Spain without hiding proxy risk?
        3. **Pre-model reality:** which countries/languages are present and which already receive interaction attention?
        4. **Static ranking audit:** what changes once recommender models rank films?
        5. **Mechanism and robustness:** are gaps linked to popularity, language, US involvement, embeddings, folds or feedback loops?
        6. **Governance answer:** which metrics should a platform report, and what trade-off does re-ranking make visible?

        Every major table or figure below is followed by a direct result statement. The notebook should be read as: **question -> computed evidence -> interpretation -> caveat**.
        """
    ),
    md(
        """
        ## 1b. Governance background and state of the art

        **Question.** Where exactly does this project sit in the literature?

        This project builds on four research streams:

        1. **Popularity bias and long-tail underexposure.** Recommender-system research has repeatedly shown that popular items tend to receive disproportionate exposure, while long-tail items remain underrepresented. This project uses that literature as the technical baseline, not as the novelty claim.
        2. **Exposure fairness and provider/item-side visibility.** Fairness-aware recommender research broadens evaluation beyond user accuracy and asks who or what receives recommendation opportunity. Our project follows this item-side/exposure logic, but applies it to cultural labels.
        3. **Multimodal movie recommendation.** M3L/Binge Watch makes it possible to compare collaborative filtering with MPNet text and CLIP image features on a MovieLens-scale movie dataset. That allows the project to ask whether multimodal features shift cultural visibility.
        4. **Feedback-loop dynamics.** Lesota, Geiger, Walder, Kowald and Schedl show that country representation can drift through repeated recommendation and simulated consumption in music recommenders. We adapt that idea as a lightweight movie-domain stress test.

        **Our gap.** Existing work often evaluates accuracy, general popularity bias, broad provider fairness or country representation in music. This notebook audits **cultural prominence at the ranking layer** in movie recommendation, using country/language proxy labels, Top-K exposure, PACPG and a transparent re-ranking frontier. The governance question is therefore not only whether European works exist in the catalogue, but whether countries and languages receive ranked and repeated visibility.

        **Governance framing.** The AVMSD gives European works catalogue/prominence relevance for audiovisual media services, while DSA-style recommender transparency makes ranking parameters and user-facing control a governance object. This notebook does not claim legal compliance; it turns those concerns into measurable audit questions.
        """
    ),
    code(
        """
        state_of_art = pd.DataFrame([
            {
                "stream": "Popularity bias",
                "source": "Klimashevskaia, Jannach, Elahi & Trattner (2023/2024), A Survey on Popularity Bias in Recommender Systems",
                "role_in_project": "Defines popularity bias and mitigation context; motivates long-tail and exposure metrics.",
                "url": "https://arxiv.org/abs/2308.01118",
            },
            {
                "stream": "Popularity bias / user impact",
                "source": "Abdollahpouri, Mansoury, Burke & Mobasher (2019), The Unfairness of Popularity Bias in Recommendation",
                "role_in_project": "Motivates preference-adjusted evaluation rather than only catalogue-level long-tail counts.",
                "url": "https://arxiv.org/abs/1907.13286",
            },
            {
                "stream": "Re-ranking mitigation",
                "source": "Abdollahpouri, Burke & Mobasher (2019), Managing Popularity Bias in Recommender Systems with Personalized Re-ranking",
                "role_in_project": "Supports transparent post-processing as a practical mitigation family.",
                "url": "https://arxiv.org/abs/1901.07555",
            },
            {
                "stream": "Fairness-aware recommendation",
                "source": "Wang et al. (2022/2023), A Survey on the Fairness of Recommender Systems; Jin et al. (2023), A Survey on Fairness-aware Recommender Systems",
                "role_in_project": "Positions the project as item-side/exposure fairness with cultural proxy labels.",
                "url": "https://arxiv.org/abs/2206.03761 | https://arxiv.org/abs/2306.00403",
            },
            {
                "stream": "Multimodal movie recommendation",
                "source": "Spillo et al. (2026), Binge Watch / M3L-10M and M3L-20M",
                "role_in_project": "Provides the multimodal MovieLens-scale dataset basis for MPNet/CLIP comparison.",
                "url": "https://zenodo.org/records/18499145",
            },
            {
                "stream": "Feedback-loop dynamics",
                "source": "Lesota, Geiger, Walder, Kowald & Schedl (2024), Oh, Behave! Country Representation Dynamics Created by Feedback Loops in Music Recommender Systems",
                "role_in_project": "Inspires the iterative country/language visibility stress test, with explicit limits in our notebook.",
                "url": "https://arxiv.org/abs/2408.11565",
            },
            {
                "stream": "Governance framing",
                "source": "European Commission AVMSD European works page; Digital Services Act recommender transparency materials",
                "role_in_project": "Frames why catalogue share, prominence and recommender transparency are governance-relevant.",
                "url": "https://digital-strategy.ec.europa.eu/en/policies/european-works | https://digital-strategy.ec.europa.eu/en/policies/dsa-impact-platforms",
            },
        ])
        save_table(state_of_art, "state_of_art_ledger.csv")
        display(state_of_art)

        result_box(
            "State-of-the-art position",
            [
                "The project does not claim that popularity bias is new; it uses popularity bias as a known mechanism to test against.",
                "The contribution is the translation of item-side exposure fairness into cultural prominence metrics for movie countries and languages.",
                "M3L enables a defensible multimodal comparison; Schedl/Lesota-style feedback-loop work motivates the dynamic stress-test chapter.",
                "The AVMSD/DSA framing explains why ranked visibility and recommender transparency are governance-relevant, without turning the notebook into a legal compliance assessment.",
            ],
            "Country and language labels are proxies. The notebook audits ranked visibility, not cultural identity or legal compliance."
        )
        """
    ),
    md(
        """
        ## 2. Load raw/local data

        **Question.** What local raw files are available, and which ones are used?

        This cell searches expected local locations and stops if core raw/local files are missing. Cached Wikidata metadata is allowed because it is not an aggregate result table; it is the local metadata enrichment layer keyed by IMDb IDs.
        """
    ),
    code(
        """
        def first_existing(candidates):
            for path in candidates:
                p = PROJECT_ROOT / path
                if p.exists():
                    return p
            return PROJECT_ROOT / candidates[0]

        file_specs = [
            ("MovieLens movies", ["MovieLens 20M Dataset/movie.csv", "data/archive/movie.csv", "archive/movie.csv"], "Movie titles and genres", True),
            ("MovieLens ratings", ["MovieLens 20M Dataset/rating.csv", "data/archive/rating.csv", "archive/rating.csv"], "Raw ratings/interactions with timestamps", True),
            ("MovieLens links", ["MovieLens 20M Dataset/link.csv", "data/archive/link.csv", "archive/link.csv"], "IMDb/TMDb bridge", True),
            ("MovieLens tags", ["MovieLens 20M Dataset/tag.csv", "data/archive/tag.csv", "archive/tag.csv"], "User tags", False),
            ("Genome scores", ["MovieLens 20M Dataset/genome_scores.csv", "data/archive/genome_scores.csv", "archive/genome_scores.csv"], "Genome relevance scores", False),
            ("Genome tags", ["MovieLens 20M Dataset/genome_tags.csv", "data/archive/genome_tags.csv", "archive/genome_tags.csv"], "Genome tag names", False),
            ("M3L interactions", ["m3l-20m/m3l-20m.inter", "data/m3l-20m/m3l-20m.inter"], "M3L split interactions", True),
            ("M3L to MovieLens map", ["data/interim/m3l_internal_to_movielens.csv"], "M3L internal item to MovieLens movieId bridge", True),
            ("MPNet matrix", ["m3l-20m/text/mpnet.npy"], "M3L MPNet plot embeddings", False),
            ("CLIP image matrix", ["m3l-20m/image/clip_image.npy"], "M3L CLIP poster embeddings", False),
            ("MPNet JSON folder", ["TEXT_mpnet"], "Per-movie MPNet JSON feature availability", False),
            ("CLIP image JSON folder", ["IMG_clip-image"], "Per-movie CLIP-image JSON feature availability", False),
            ("Wikidata cache", ["data/interim/wikidata_movie_metadata_extended.csv", "data/interim/wikidata_movie_metadata.csv"], "Cached country/language/production metadata", True),
            ("Wikidata DNA cache", ["data/interim/wikidata_visibility_dna_extra.csv"], "Optional deeper Wikidata DNA metadata: director citizenship, filming location and awards", False),
        ]

        inventory_rows = []
        resolved = {}
        for source, candidates, role, required in file_specs:
            path = first_existing(candidates)
            exists = path.exists()
            resolved[source] = path if exists else None
            size = path.stat().st_size / 1024**2 if exists and path.is_file() else np.nan
            inventory_rows.append({
                "source_name": source,
                "expected_file": " | ".join(candidates),
                "resolved_path": str(path) if exists else "",
                "exists": exists,
                "file_size_mb": round(size, 2) if not np.isnan(size) else np.nan,
                "role": role,
                "used_in_analysis": exists and (required or source in ["MovieLens tags", "Genome scores", "Genome tags", "MPNet matrix", "CLIP image matrix", "MPNet JSON folder", "CLIP image JSON folder"]),
                "required": required,
            })

        data_inventory = pd.DataFrame(inventory_rows)
        save_table(data_inventory, "data_inventory.csv")
        display(data_inventory)

        missing_required = data_inventory.query("required and not exists")
        if len(missing_required):
            display(missing_required)
            raise FileNotFoundError("Missing required raw/local files. Cannot run final research notebook without raw data. No synthetic fallback is allowed.")

        # Core raw/local loads. Ratings are bounded by config for quick local execution.
        movies = pd.read_csv(resolved["MovieLens movies"])
        links = pd.read_csv(resolved["MovieLens links"])
        ratings = pd.read_csv(resolved["MovieLens ratings"], nrows=MAX_RATINGS_FOR_DESCRIPTIVE, parse_dates=["timestamp"])
        tags = pd.read_csv(resolved["MovieLens tags"], parse_dates=["timestamp"]) if resolved["MovieLens tags"] else pd.DataFrame(columns=["userId", "movieId", "tag", "timestamp"])
        m3l_interactions = pd.read_csv(resolved["M3L interactions"], sep="\\t", dtype={"userID": "int32", "itemID": "int32", "rating": "float32", "x_label": "int8"})
        m3l_map = pd.read_csv(resolved["M3L to MovieLens map"])
        m3l_map["item_id"] = m3l_map["item_id"].astype(int)
        m3l_map["movieId"] = m3l_map["movieId"].astype(int)
        # The raw bridge is item-level; the movie database must remain one row per MovieLens movie.
        movie_m3l_map = m3l_map.sort_values("item_id").drop_duplicates("movieId", keep="first").copy()
        bridge_diagnostics = pd.DataFrame([
            {"metric": "M3L bridge item rows", "value": len(m3l_map)},
            {"metric": "unique M3L item_id", "value": m3l_map["item_id"].nunique()},
            {"metric": "unique MovieLens movieId", "value": m3l_map["movieId"].nunique()},
            {"metric": "duplicate MovieLens movieId rows in bridge", "value": int(m3l_map["movieId"].duplicated().sum())},
        ])
        save_table(bridge_diagnostics, "m3l_bridge_diagnostics.csv")
        wikidata_raw = pd.read_csv(resolved["Wikidata cache"]) if resolved["Wikidata cache"] else pd.DataFrame()

        display(pd.DataFrame([
            {"object": "MovieLens movies", "rows": len(movies), "unique_movies": movies.movieId.nunique()},
            {"object": "Loaded MovieLens ratings", "rows": len(ratings), "unique_users": ratings.userId.nunique(), "unique_movies": ratings.movieId.nunique()},
            {"object": "M3L interactions", "rows": len(m3l_interactions), "unique_users": m3l_interactions.userID.nunique(), "unique_items": m3l_interactions.itemID.nunique()},
            {"object": "Wikidata cache rows", "rows": len(wikidata_raw), "unique_imdb": wikidata_raw.imdb_id_str.nunique() if "imdb_id_str" in wikidata_raw else 0},
        ]))

        save_json({"config": config, "runtime_info": runtime_info.to_dict("records"), "data_inventory": data_inventory.to_dict("records")}, "notebook_run_manifest.json")
        """
    ),
    md(
        """
        **Interpretation.** The analysis starts from raw/local MovieLens and M3L files plus cached Wikidata metadata. If `FULL_RUN` is `False`, MovieLens rating-derived descriptive statistics use a bounded rating sample; the M3L interaction file is still loaded as the recommender source. This is an offline recommender audit, not platform UI telemetry.
        """
    ),
    md(
        """
        ## 3. Build the movie-level analysis database

        **Question.** Can we construct one movie-level table with identifiers, ratings, feature coverage and cultural metadata?
        """
    ),
    code(
        """
        movies = movies.join(movies["title"].apply(parse_title_year))
        links["imdb_id_str"] = links["imdbId"].apply(imdb_numeric_to_tt)

        rating_stats = ratings.groupby("movieId").agg(
            rating_count=("rating", "size"),
            mean_rating=("rating", "mean"),
            rating_std=("rating", "std"),
            first_rating_at=("timestamp", "min"),
            last_rating_at=("timestamp", "max"),
        ).reset_index()

        tag_stats = tags.groupby("movieId").agg(
            tag_count=("tag", "size"),
            unique_tag_count=("tag", "nunique"),
            top_tags=("tag", lambda s: " | ".join(pd.Series(s).astype(str).str.lower().value_counts().head(8).index)),
        ).reset_index() if len(tags) else pd.DataFrame(columns=["movieId", "tag_count", "unique_tag_count", "top_tags"])

        if resolved["Genome scores"] and resolved["Genome tags"]:
            genome_tags = pd.read_csv(resolved["Genome tags"])
            genome_chunks = []
            for chunk in pd.read_csv(resolved["Genome scores"], chunksize=1_000_000):
                top = chunk.sort_values(["movieId", "relevance"], ascending=[True, False]).groupby("movieId").head(5)
                genome_chunks.append(top)
            genome_top = pd.concat(genome_chunks, ignore_index=True).merge(genome_tags, on="tagId", how="left")
            genome_summary = genome_top.groupby("movieId").agg(
                genome_tag_coverage=("tagId", "size"),
                top_genome_tags=("tag", lambda s: " | ".join(s.dropna().astype(str).head(5))),
            ).reset_index()
        else:
            genome_summary = pd.DataFrame(columns=["movieId", "genome_tag_coverage", "top_genome_tags"])

        m3l_with_movie = m3l_interactions.merge(m3l_map, left_on="itemID", right_on="item_id", how="left")
        split_name = {0: "train", 1: "validation", 2: "test"}
        m3l_with_movie["split"] = m3l_with_movie["x_label"].map(split_name).fillna("unknown")
        split_counts = m3l_with_movie.groupby(["movieId", "split"]).size().unstack(fill_value=0).reset_index()
        for col in ["train", "validation", "test"]:
            if col not in split_counts:
                split_counts[col] = 0
        split_counts = split_counts.rename(columns={"train": "rating_count_train", "validation": "rating_count_validation", "test": "rating_count_test"})

        mpnet_json_ids = {int(p.stem) for p in resolved["MPNet JSON folder"].glob("*.json")} if resolved["MPNet JSON folder"] and resolved["MPNet JSON folder"].is_dir() else set()
        clip_json_ids = {int(p.stem) for p in resolved["CLIP image JSON folder"].glob("*.json")} if resolved["CLIP image JSON folder"] and resolved["CLIP image JSON folder"].is_dir() else set()
        matrix_item_ids = set(m3l_map["item_id"].astype(int))

        wd = wikidata_raw.copy()
        if len(wd):
            for col in ["country", "original_language", "language_of_work", "production_company", "production_company_country", "production_company_hq_country"]:
                if col not in wd:
                    wd[col] = np.nan
            wikidata_agg = wd.groupby("imdb_id_str").agg(
                wikidata_uri=("wikidata_uri", "first"),
                title_wikidata=("title_wikidata", "first"),
                country=("country", lambda s: sorted(set(chain.from_iterable(split_pipe(x) for x in s.dropna())))),
                original_language=("original_language", lambda s: sorted(set(chain.from_iterable(split_pipe(x) for x in s.dropna())))),
                language_of_work=("language_of_work", lambda s: sorted(set(chain.from_iterable(split_pipe(x) for x in s.dropna())))),
                production_company=("production_company", lambda s: sorted(set(chain.from_iterable(split_pipe(x) for x in s.dropna())))),
                production_company_country=("production_company_country", lambda s: sorted(set(chain.from_iterable(split_pipe(x) for x in s.dropna())))),
                production_company_hq_country=("production_company_hq_country", lambda s: sorted(set(chain.from_iterable(split_pipe(x) for x in s.dropna())))),
                publication_date=("publication_date", "first"),
            ).reset_index()
        else:
            wikidata_agg = pd.DataFrame(columns=["imdb_id_str"])

        movie_db = (
            movies
            .merge(links, on="movieId", how="left")
            .merge(rating_stats, on="movieId", how="left")
            .merge(tag_stats, on="movieId", how="left")
            .merge(genome_summary, on="movieId", how="left")
            .merge(movie_m3l_map, on="movieId", how="left")
            .merge(split_counts, on="movieId", how="left")
            .merge(wikidata_agg, on="imdb_id_str", how="left")
        )
        movie_db["has_mpnet_json"] = movie_db["movieId"].isin(mpnet_json_ids)
        movie_db["has_clip_json"] = movie_db["movieId"].isin(clip_json_ids)
        movie_db["has_mpnet_matrix"] = movie_db["item_id"].isin(matrix_item_ids) & bool(resolved["MPNet matrix"])
        movie_db["has_clip_matrix"] = movie_db["item_id"].isin(matrix_item_ids) & bool(resolved["CLIP image matrix"])
        for col in ["rating_count", "tag_count", "unique_tag_count", "genome_tag_coverage", "rating_count_train", "rating_count_validation", "rating_count_test"]:
            movie_db[col] = movie_db[col].fillna(0)
        for col in ["country", "original_language", "language_of_work", "production_company", "production_company_country", "production_company_hq_country"]:
            movie_db[col] = movie_db[col].apply(lambda x: x if isinstance(x, list) else [])

        join_funnel = pd.DataFrame([
            {"stage": "MovieLens movies", "items": movies.movieId.nunique()},
            {"stage": "with MovieLens links", "items": movie_db["imdb_id_str"].notna().sum()},
            {"stage": "in M3L item map", "items": movie_db["item_id"].notna().sum()},
            {"stage": "with Wikidata match", "items": movie_db["wikidata_uri"].notna().sum()},
            {"stage": "with country metadata", "items": movie_db["country"].map(len).gt(0).sum()},
            {"stage": "with language metadata", "items": movie_db["original_language"].map(len).gt(0).sum()},
            {"stage": "with MPNet matrix", "items": movie_db["has_mpnet_matrix"].sum()},
            {"stage": "with CLIP image matrix", "items": movie_db["has_clip_matrix"].sum()},
        ])
        save_table(join_funnel, "join_funnel.csv")
        save_table(movie_db.head(1000), "movie_database_preview.csv")

        display(movie_db.head())
        display(join_funnel)

        fig, ax = plt.subplots(figsize=(9, 4.8))
        ax.bar(join_funnel["stage"], join_funnel["items"], color="#167f86")
        ax.set_title("MovieLens -> M3L -> Wikidata join funnel")
        ax.set_ylabel("Movies/items")
        ax.tick_params(axis="x", rotation=35)
        save_current_figure("03_join_funnel_final.png", "3. Build movie database", "Data coverage", "join_funnel.csv", "Join funnel from raw/local sources to audit metadata.")

        add_check("unique movieId in movie_db", movie_db["movieId"].is_unique, f"{movie_db['movieId'].duplicated().sum()} duplicate movieId rows")
        add_check("ratings movieIds exist in movie table", set(ratings["movieId"].unique()).issubset(set(movies["movieId"].unique())), "MovieLens rating movieIds checked against movie table")
        """
    ),
    md(
        """
        **Interpretation.** The join funnel is the first governance result: cultural labels are only as strong as the metadata coverage behind them. Unknown country/language is preserved in later sections instead of being dropped.
        """
    ),
    md(
        """
        ## 4. Metadata coverage and missingness

        **Question.** Can we trust country/language analysis, and where are the weak spots?
        """
    ),
    code(
        """
        movie_db["has_country"] = movie_db["country"].map(len).gt(0)
        movie_db["has_language"] = movie_db["original_language"].map(len).gt(0)
        movie_db["has_production_company"] = movie_db["production_company"].map(len).gt(0)
        movie_db["has_release_year"] = movie_db["release_year"].notna()
        movie_db["has_genre"] = movie_db["genres"].notna() & (movie_db["genres"] != "(no genres listed)")
        movie_db["has_rating"] = movie_db["rating_count"].gt(0)
        movie_db["has_tag"] = movie_db["tag_count"].gt(0)
        movie_db["has_genome"] = movie_db["genome_tag_coverage"].gt(0)

        coverage_fields = [
            "has_country", "has_language", "has_production_company", "has_release_year",
            "has_genre", "has_mpnet_matrix", "has_clip_matrix", "has_rating", "has_tag", "has_genome"
        ]
        metadata_coverage = pd.DataFrame({
            "field": coverage_fields,
            "coverage_rate": [movie_db[col].mean() for col in coverage_fields],
            "covered_items": [int(movie_db[col].sum()) for col in coverage_fields],
            "total_items": len(movie_db),
        })
        save_table(metadata_coverage, "metadata_coverage.csv")
        display(metadata_coverage)

        barh(metadata_coverage.set_index("field")["coverage_rate"], "Metadata and feature coverage", "Coverage rate", "04_metadata_coverage_final.png", "4. Metadata coverage", "Data quality", "metadata_coverage.csv")

        movie_db["release_decade"] = (movie_db["release_year"] // 10 * 10).astype("Int64").astype(str) + "s"
        movie_db.loc[movie_db["release_year"].isna(), "release_decade"] = "Unknown"
        movie_db["popularity_rank"] = movie_db["rating_count_train"].rank(method="average", pct=True)
        movie_db["popularity_decile"] = pd.qcut(movie_db["popularity_rank"].fillna(0), 10, labels=False, duplicates="drop")
        q20 = movie_db["rating_count_train"].quantile(0.20)
        q30 = movie_db["rating_count_train"].quantile(0.30)
        q40 = movie_db["rating_count_train"].quantile(0.40)
        q80 = movie_db["rating_count_train"].quantile(0.80)
        movie_db["is_long_tail_20"] = movie_db["rating_count_train"] <= q20
        movie_db["is_long_tail_30"] = movie_db["rating_count_train"] <= q30
        movie_db["is_long_tail_40"] = movie_db["rating_count_train"] <= q40
        movie_db["is_blockbuster_head"] = movie_db["rating_count_train"] >= q80

        coverage_by_decade = movie_db.groupby("release_decade").agg(
            movies=("movieId", "size"),
            country_coverage=("has_country", "mean"),
            language_coverage=("has_language", "mean"),
            mpnet_coverage=("has_mpnet_matrix", "mean"),
            clip_coverage=("has_clip_matrix", "mean"),
        ).reset_index()
        save_table(coverage_by_decade, "metadata_coverage_by_decade.csv")
        display(coverage_by_decade.tail(12))

        coverage_by_pop = movie_db.groupby("popularity_decile", dropna=False).agg(
            movies=("movieId", "size"),
            country_coverage=("has_country", "mean"),
            language_coverage=("has_language", "mean"),
        ).reset_index()
        save_table(coverage_by_pop, "metadata_coverage_by_popularity_decile.csv")
        display(coverage_by_pop)

        unknown_country_share = 1 - movie_db["has_country"].mean()
        unknown_language_share = 1 - movie_db["has_language"].mean()
        display(Markdown(f"**Coverage caveat.** Unknown country share is {unknown_country_share:.1%}; unknown language share is {unknown_language_share:.1%}. Unknowns remain visible as groups in later tables."))
        """
    ),
    md(
        """
        **Interpretation.** Country and original-language metadata are strong enough for an offline visibility audit if missingness is reported. Production-company metadata is weaker and is used only as a caveat mechanism, not as a primary cultural label.
        """
    ),
    md(
        """
        ## 5. Define cultural audit labels

        **Question.** How are Europe, countries, regions, languages and Spain-specific groups operationalised?

        These are audit proxies, not perfect cultural categories. Multi-country and multi-language films use binary labels for flags and fractional credit for distribution tables.
        """
    ),
    code(
        """
        EU27 = {
            "Austria", "Belgium", "Bulgaria", "Croatia", "Cyprus", "Czech Republic", "Czechia", "Denmark", "Estonia",
            "Finland", "France", "Germany", "Greece", "Hungary", "Ireland", "Italy", "Latvia", "Lithuania",
            "Luxembourg", "Malta", "Netherlands", "Poland", "Portugal", "Romania", "Slovakia", "Slovenia", "Spain", "Sweden"
        }
        WIDE_EUROPE = EU27 | {
            "United Kingdom", "Norway", "Switzerland", "Iceland", "Ukraine", "Serbia", "Bosnia and Herzegovina",
            "Montenegro", "North Macedonia", "Albania", "Moldova", "Belarus", "Russia", "Turkey", "Georgia",
            "Armenia", "Azerbaijan", "Kosovo", "Liechtenstein", "Monaco", "San Marino", "Andorra", "Vatican City"
        }
        REGION_MAP = {
            "United Kingdom": "UK/Ireland", "Ireland": "UK/Ireland",
            "France": "France",
            "Germany": "Germany/Austria/Switzerland", "Austria": "Germany/Austria/Switzerland", "Switzerland": "Germany/Austria/Switzerland",
            "Spain": "Spain/Portugal", "Portugal": "Spain/Portugal",
            "Italy": "Italy/Malta", "Malta": "Italy/Malta",
            "Denmark": "Nordics", "Sweden": "Nordics", "Norway": "Nordics", "Finland": "Nordics", "Iceland": "Nordics",
            "Belgium": "Benelux", "Netherlands": "Benelux", "Luxembourg": "Benelux",
            "Poland": "Central Europe", "Czech Republic": "Central Europe", "Czechia": "Central Europe", "Slovakia": "Central Europe", "Hungary": "Central Europe", "Slovenia": "Central Europe",
            "Ukraine": "Eastern Europe", "Belarus": "Eastern Europe", "Russia": "Eastern Europe", "Moldova": "Eastern Europe", "Romania": "Eastern Europe", "Bulgaria": "Eastern Europe",
            "Croatia": "Balkans", "Serbia": "Balkans", "Bosnia and Herzegovina": "Balkans", "Montenegro": "Balkans", "North Macedonia": "Balkans", "Albania": "Balkans", "Greece": "Balkans", "Cyprus": "Balkans", "Kosovo": "Balkans",
        }
        region_mapping = pd.DataFrame([{"country": c, "european_region": r} for c, r in sorted(REGION_MAP.items())])
        save_table(region_mapping, "european_region_mapping.csv")
        display(region_mapping)

        def countries_for(row):
            return row["country"] if len(row["country"]) else ["Unknown country"]

        def languages_for(row):
            return row["original_language"] if len(row["original_language"]) else ["Unknown language"]

        def primary_label(values, unknown):
            vals = [v for v in values if v and not str(v).startswith("Unknown")]
            return vals[0] if vals else unknown

        movie_db["origin_countries"] = movie_db.apply(countries_for, axis=1)
        movie_db["original_languages"] = movie_db.apply(languages_for, axis=1)
        movie_db["primary_origin_country"] = movie_db["origin_countries"].apply(lambda xs: primary_label(xs, "Unknown country"))
        movie_db["primary_original_language"] = movie_db["original_languages"].apply(lambda xs: primary_label(xs, "Unknown language"))
        movie_db["is_european_wide"] = movie_db["origin_countries"].apply(lambda xs: any(x in WIDE_EUROPE for x in xs))
        movie_db["is_eu27"] = movie_db["origin_countries"].apply(lambda xs: any(x in EU27 for x in xs))
        movie_db["european_region"] = movie_db["origin_countries"].apply(lambda xs: sorted({REGION_MAP.get(x, "Other Europe") for x in xs if x in WIDE_EUROPE}) or ["Unknown Europe"])
        movie_db["has_us_origin"] = movie_db["origin_countries"].apply(lambda xs: "United States" in xs)
        movie_db["is_coproduction"] = movie_db["origin_countries"].apply(lambda xs: len([x for x in xs if not x.startswith("Unknown")]) > 1)
        movie_db["num_origin_countries"] = movie_db["origin_countries"].apply(lambda xs: len([x for x in xs if not x.startswith("Unknown")]))
        movie_db["is_english_language"] = movie_db["original_languages"].apply(lambda xs: "English" in xs)
        movie_db["is_non_english"] = movie_db["original_languages"].apply(lambda xs: len([x for x in xs if not x.startswith("Unknown")]) > 0 and "English" not in xs)
        movie_db["is_multilingual"] = movie_db["original_languages"].apply(lambda xs: len([x for x in xs if not x.startswith("Unknown")]) > 1)
        movie_db["has_us_company_involvement"] = movie_db["production_company_country"].apply(lambda xs: "United States" in xs)
        movie_db["is_spain_origin"] = movie_db["origin_countries"].apply(lambda xs: "Spain" in xs)
        movie_db["is_spanish_language"] = movie_db["original_languages"].apply(lambda xs: "Spanish" in xs)
        movie_db["is_european_spanish_language"] = movie_db["is_european_wide"] & movie_db["is_spanish_language"]

        save_table(movie_db[["movieId", "title", "origin_countries", "original_languages", "primary_origin_country", "primary_original_language", "is_european_wide", "is_eu27", "european_region", "is_spain_origin", "is_spanish_language", "is_european_spanish_language"]].head(2000), "labelled_movie_preview.csv")
        display(movie_db[["movieId", "title", "origin_countries", "original_languages", "european_region", "is_spain_origin", "is_spanish_language"]].head())

        add_check("unknown country preserved", movie_db["origin_countries"].apply(lambda xs: "Unknown country" in xs).any(), "Unknown country appears as explicit label when metadata is missing")
        add_check("unknown language preserved", movie_db["original_languages"].apply(lambda xs: "Unknown language" in xs).any(), "Unknown language appears as explicit label when metadata is missing")
        """
    ),
    md(
        """
        **Caveat.** Country of origin, original language and production-company metadata are governance-relevant proxies. They do not define identity, audience or cultural value. The analysis uses them to audit ranked visibility, not to infer sensitive user attributes.
        """
    ),
    md(
        """
        ## 6. Simple descriptive questions before modelling

        **Question.** What is the data reality before algorithms rank anything?
        """
    ),
    code(
        """
        def fractional_distribution(df, label_col, weight_col=None, filter_func=None):
            rows = []
            work = df if filter_func is None else df[filter_func(df)]
            for _, row in work.iterrows():
                labels = row[label_col]
                if not isinstance(labels, list) or not labels:
                    labels = ["Unknown"]
                credit = 1 / len(labels)
                weight = float(row[weight_col]) if weight_col else 1.0
                for label in labels:
                    rows.append({"label": label, "value": credit * weight})
            out = pd.DataFrame(rows)
            if out.empty:
                return pd.DataFrame(columns=["label", "value", "share"])
            out = out.groupby("label", as_index=False)["value"].sum()
            out["share"] = out["value"] / out["value"].sum()
            return out.sort_values("value", ascending=False)

        catalogue_country = fractional_distribution(movie_db, "origin_countries")
        catalogue_language = fractional_distribution(movie_db, "original_languages")
        interaction_country = fractional_distribution(movie_db, "origin_countries", "rating_count_train")
        interaction_language = fractional_distribution(movie_db, "original_languages", "rating_count_train")
        european_country = catalogue_country[catalogue_country["label"].isin(WIDE_EUROPE)]

        add_check("fractional country shares sum correctly", abs(catalogue_country["share"].sum() - 1.0) < 1e-6, f"country share sum={catalogue_country['share'].sum():.6f}")
        add_check("fractional language shares sum correctly", abs(catalogue_language["share"].sum() - 1.0) < 1e-6, f"language share sum={catalogue_language['share'].sum():.6f}")

        save_table(catalogue_country, "catalogue_country_distribution.csv")
        save_table(catalogue_language, "catalogue_language_distribution.csv")
        save_table(interaction_country, "catalogue_vs_interaction_country.csv")
        save_table(interaction_language, "catalogue_vs_interaction_language.csv")

        simple_counts = pd.DataFrame([
            {"question": "movies", "answer": len(movie_db)},
            {"question": "loaded MovieLens users", "answer": ratings["userId"].nunique()},
            {"question": "loaded MovieLens ratings", "answer": len(ratings)},
            {"question": "M3L users", "answer": m3l_interactions["userID"].nunique()},
            {"question": "M3L interactions", "answer": len(m3l_interactions)},
            {"question": "movies with country metadata", "answer": int(movie_db["has_country"].sum())},
            {"question": "movies with language metadata", "answer": int(movie_db["has_language"].sum())},
            {"question": "European-wide films", "answer": int(movie_db["is_european_wide"].sum())},
            {"question": "non-English films", "answer": int(movie_db["is_non_english"].sum())},
            {"question": "Spain-origin films", "answer": int(movie_db["is_spain_origin"].sum())},
            {"question": "Spanish-language films", "answer": int(movie_db["is_spanish_language"].sum())},
            {"question": "European Spanish-language films", "answer": int(movie_db["is_european_spanish_language"].sum())},
        ])
        save_table(simple_counts, "simple_descriptive_counts.csv")
        display(simple_counts)
        display(catalogue_country.head(20))
        display(european_country.head(20))
        display(catalogue_language.head(20))

        barh(catalogue_country.head(20).set_index("label")["share"], "Catalogue country distribution, top 20", "Fractional catalogue share", "06_country_top20.png", "6. Descriptive questions", "RQ2", "catalogue_country_distribution.csv", "Multi-country films use fractional credit.")
        barh(european_country.head(20).set_index("label")["share"], "European catalogue country distribution, top 20", "Fractional catalogue share", "06_european_country_top20.png", "6. Descriptive questions", "RQ2", "catalogue_country_distribution.csv", "Only countries mapped to wider Europe are shown.")
        barh(catalogue_language.head(20).set_index("label")["share"], "Catalogue original-language distribution, top 20", "Fractional catalogue share", "06_language_top20.png", "6. Descriptive questions", "RQ3", "catalogue_language_distribution.csv", "Multi-language films use fractional credit.")

        ci_country = catalogue_country.merge(interaction_country[["label", "share"]], on="label", how="outer", suffixes=("_catalogue", "_interaction")).fillna(0)
        ci_language = catalogue_language.merge(interaction_language[["label", "share"]], on="label", how="outer", suffixes=("_catalogue", "_interaction")).fillna(0)
        save_table(ci_country, "catalogue_vs_interaction_country.csv")
        save_table(ci_language, "catalogue_vs_interaction_language.csv")

        top_countries = ci_country.sort_values("share_interaction", ascending=False).head(15).set_index("label")[["share_catalogue", "share_interaction"]]
        fig, ax = plt.subplots(figsize=(10, 5))
        top_countries.sort_values("share_interaction").plot(kind="barh", ax=ax)
        ax.set_title("Catalogue vs train-interaction share by country")
        ax.set_xlabel("Share")
        save_current_figure("06_catalogue_vs_interaction_country.png", "6. Descriptive questions", "RQ2", "catalogue_vs_interaction_country.csv", "Catalogue availability vs observed interaction share by country.")

        top_languages = ci_language.sort_values("share_interaction", ascending=False).head(15).set_index("label")[["share_catalogue", "share_interaction"]]
        fig, ax = plt.subplots(figsize=(10, 5))
        top_languages.sort_values("share_interaction").plot(kind="barh", ax=ax)
        ax.set_title("Catalogue vs train-interaction share by original language")
        ax.set_xlabel("Share")
        save_current_figure("06_catalogue_vs_interaction_language.png", "6. Descriptive questions", "RQ3", "catalogue_vs_interaction_language.csv", "Catalogue availability vs observed interaction share by language.")

        region_dist = fractional_distribution(movie_db[movie_db["is_european_wide"]], "european_region")
        save_table(region_dist, "european_region_distribution.csv")
        barh(region_dist.set_index("label")["share"], "European region composition", "Fractional catalogue share", "06_european_region_composition.png", "6. Descriptive questions", "RQ2", "european_region_distribution.csv", "Region mapping is an audit grouping, not a cultural truth.")

        origin_group = np.select(
            [movie_db["is_spain_origin"], movie_db["is_european_wide"], movie_db["has_us_origin"]],
            ["Spain-origin", "Other Europe", "US origin"],
            default="Other / unknown",
        )
        movie_db["origin_group"] = origin_group
        pop_origin = pd.crosstab(movie_db["popularity_decile"], movie_db["origin_group"], normalize="columns").fillna(0)
        save_table(pop_origin.reset_index(), "popularity_decile_by_origin_group.csv")
        fig, ax = plt.subplots(figsize=(10, 5))
        pop_origin.plot(kind="bar", ax=ax)
        ax.set_title("Popularity decile by origin group")
        ax.set_xlabel("Popularity decile")
        ax.set_ylabel("Within-group share")
        save_current_figure("06_popularity_decile_by_origin_group.png", "6. Descriptive questions", "RQ7", "popularity_decile_by_origin_group.csv", "Long-tail concentration by origin group.")

        spain_counts = pd.DataFrame([
            {"group": "Spain-origin", "count": int(movie_db["is_spain_origin"].sum()), "share": movie_db["is_spain_origin"].mean()},
            {"group": "Spanish-language", "count": int(movie_db["is_spanish_language"].sum()), "share": movie_db["is_spanish_language"].mean()},
            {"group": "European Spanish-language", "count": int(movie_db["is_european_spanish_language"].sum()), "share": movie_db["is_european_spanish_language"].mean()},
            {"group": "Spain-origin AND Spanish-language", "count": int((movie_db["is_spain_origin"] & movie_db["is_spanish_language"]).sum()), "share": (movie_db["is_spain_origin"] & movie_db["is_spanish_language"]).mean()},
        ])
        save_table(spain_counts, "spain_case_study_counts.csv")
        display(spain_counts)
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(spain_counts["group"], spain_counts["count"], color=["#167f86", "#c99700", "#2f6b9a", "#b84a4a"])
        ax.set_title("Spain-origin vs Spanish-language support")
        ax.set_ylabel("Catalogue movies")
        ax.tick_params(axis="x", rotation=25)
        save_current_figure("06_spain_origin_vs_spanish_language.png", "6. Descriptive questions", "RQ4", "spain_case_study_counts.csv", "Spain-origin and Spanish-language are separate definitions.")

        support_country = ci_country.merge(interaction_country[["label", "value"]].rename(columns={"value": "train_interactions"}), on="label", how="left")
        support_country = support_country.rename(columns={"value": "support_catalogue_items"})
        support_country["threshold_passed"] = (support_country["support_catalogue_items"] >= MIN_COUNTRY_ITEMS) & (support_country["train_interactions"].fillna(0) >= MIN_GROUP_INTERACTIONS)
        support_country["threshold_reason_if_failed"] = np.where(support_country["threshold_passed"], "", "low catalogue item or train-interaction support")
        save_table(support_country, "support_threshold_groups.csv")
        display(support_country.sort_values("support_catalogue_items", ascending=False).head(20))

        top_catalogue_country = catalogue_country.iloc[0]
        top_eu_catalogue_country = european_country.iloc[0] if len(european_country) else pd.Series({"label": "n/a", "share": np.nan})
        top_language = catalogue_language.iloc[0]
        top_interaction_country = interaction_country.iloc[0]
        top_interaction_language = interaction_language.iloc[0]
        non_english_long_tail = movie_db.loc[movie_db["is_non_english"], "is_long_tail_20"].mean()
        europe_long_tail = movie_db.loc[movie_db["is_european_wide"], "is_long_tail_20"].mean()
        low_support_groups = support_country[~support_country["threshold_passed"]]["label"].nunique()

        result_box(
            "Chapter 6 result: the data already has a cultural visibility shape before modelling",
            [
                f"The largest catalogue country signal is {top_catalogue_country['label']} ({fmt_pct(top_catalogue_country['share'])}); the largest European catalogue signal is {top_eu_catalogue_country['label']} ({fmt_pct(top_eu_catalogue_country['share'])}).",
                f"The largest original-language signal is {top_language['label']} ({fmt_pct(top_language['share'])}). This matters because a Europe audit can otherwise become an English-language Europe audit.",
                f"The largest train-interaction country/language signals are {top_interaction_country['label']} ({fmt_pct(top_interaction_country['share'])}) and {top_interaction_language['label']} ({fmt_pct(top_interaction_language['share'])}). Interaction attention is therefore not the same object as catalogue availability.",
                f"Spain-origin films and Spanish-language films are separated from the start: {int(movie_db['is_spain_origin'].sum()):,} Spain-origin films versus {int(movie_db['is_spanish_language'].sum()):,} Spanish-language films.",
                f"Long-tail risk is visible before models: {fmt_pct(europe_long_tail)} of European-wide films and {fmt_pct(non_english_long_tail)} of non-English films fall into the bottom-20% popularity tail.",
                f"{low_support_groups} country labels fail the configured support threshold; these are kept in CSV outputs but not overclaimed in the main interpretation.",
            ],
            "These figures describe the local MovieLens/M3L/Wikidata audit dataset. They are not statements about all streaming catalogues."
        )
        """
    ),
    md(
        """
        **Interpretation.** Spain-origin and Spanish-language are not the same label. A Spanish-language film can be from Spain, Latin America, the US or another context; a Spain-origin film can be multilingual or not Spanish-language in the metadata. This distinction is carried into the recommender audit.
        """
    ),
    md(
        """
        ## 7. Train / validation / test split

        **Question.** Which users and items are evaluated, and is test leakage avoided?

        The notebook uses the M3L split label when available: `0=train`, `1=validation`, `2=test`. Evaluated users must have both train and test interactions. Seen train items are excluded from recommendations.
        """
    ),
    code(
        """
        inter = m3l_with_movie.dropna(subset=["movieId"]).copy()
        inter["movieId"] = inter["movieId"].astype(int)
        inter["user_id"] = inter["userID"].astype(int)
        inter["item_id"] = inter["itemID"].astype(int)

        train_inter = inter[inter["split"].eq("train")].copy()
        valid_inter = inter[inter["split"].eq("validation")].copy()
        test_inter = inter[inter["split"].eq("test")].copy()
        if len(test_inter) == 0:
            # Fallback only when no explicit M3L test split exists.
            inter_sorted = inter.sort_values(["user_id", "item_id"])
            per_user_rank = inter_sorted.groupby("user_id").cumcount()
            per_user_size = inter_sorted.groupby("user_id")["item_id"].transform("size")
            train_inter = inter_sorted[per_user_rank < per_user_size * 0.8].copy()
            test_inter = inter_sorted[per_user_rank >= per_user_size * 0.8].copy()
            valid_inter = pd.DataFrame(columns=inter.columns)
            filter_threshold_ledger.append({"step": "train_test_split", "rule": "random/order fallback", "reason": "No explicit M3L test split found"})

        eligible_users = sorted(set(train_inter["user_id"]) & set(test_inter["user_id"]))
        rng = np.random.default_rng(RANDOM_SEED)
        if len(eligible_users) > SAMPLE_USERS:
            eval_users = sorted(rng.choice(eligible_users, SAMPLE_USERS, replace=False).tolist())
            sample_note = f"Sampled {SAMPLE_USERS} of {len(eligible_users)} eligible users for local runtime."
        else:
            eval_users = eligible_users
            sample_note = f"Using all {len(eligible_users)} eligible users."

        train_eval = train_inter[train_inter["user_id"].isin(eval_users)].copy()
        test_eval = test_inter[test_inter["user_id"].isin(eval_users)].copy()

        item_pop = train_inter.groupby("item_id").size().rename("train_item_count").reset_index()
        top_items = item_pop.nlargest(MAX_CANDIDATE_ITEMS, "train_item_count")["item_id"].astype(int).tolist()
        candidate_items = sorted(set(top_items) | set(test_eval["item_id"].astype(int).unique()))
        candidate_items = [i for i in candidate_items if i in set(m3l_map["item_id"])]

        user_to_idx = {u: i for i, u in enumerate(eval_users)}
        item_to_idx = {it: i for i, it in enumerate(candidate_items)}
        idx_to_item = {i: it for it, i in item_to_idx.items()}
        idx_to_movie = m3l_map.set_index("item_id")["movieId"].to_dict()

        train_small = train_eval[train_eval["item_id"].isin(candidate_items)]
        rows = train_small["user_id"].map(user_to_idx).to_numpy()
        cols = train_small["item_id"].map(item_to_idx).to_numpy()
        values = np.ones(len(train_small), dtype=np.float32)
        train_matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(eval_users), len(candidate_items)))

        train_seen_by_user = train_eval.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(int)) & set(candidate_items)).to_dict()
        raw_test_by_user = test_eval.groupby("user_id")["item_id"].apply(lambda s: set(s.astype(int)) & set(candidate_items)).to_dict()
        test_by_user = {
            u: raw_test_by_user.get(u, set()) - train_seen_by_user.get(u, set())
            for u in set(raw_test_by_user) | set(train_seen_by_user)
        }
        eval_users = [u for u in eval_users if len(test_by_user.get(u, set())) > 0]
        train_test_overlap_count = sum(len(train_seen_by_user.get(u, set()) & test_by_user.get(u, set())) for u in eval_users)

        split_summary = pd.DataFrame([
            {"split": "train_eval", "users": train_eval.user_id.nunique(), "items": train_eval.item_id.nunique(), "interactions": len(train_eval)},
            {"split": "test_eval", "users": test_eval.user_id.nunique(), "items": test_eval.item_id.nunique(), "interactions": len(test_eval)},
            {"split": "candidate_set", "users": len(eval_users), "items": len(candidate_items), "interactions": np.nan},
        ])
        save_table(split_summary, "train_validation_test_split_summary.csv")
        display(split_summary)
        display(Markdown(f"**Split interpretation.** {sample_note} Candidate set contains {len(candidate_items):,} items."))

        add_check("evaluated users have train interactions", train_eval.groupby("user_id").size().reindex(eval_users).fillna(0).gt(0).all(), "Every evaluated user has train interactions")
        add_check("evaluated users have test interactions", pd.Series([len(test_by_user.get(u, set())) for u in eval_users]).gt(0).all(), "Every evaluated user has test candidate interactions")
        add_check("no test data used for training", train_test_overlap_count == 0, f"train/test item overlap after filtering={train_test_overlap_count}")
        """
    ),
    md(
        """
        ## 8. Baseline visibility without algorithms

        **Question.** What would visibility look like before recommender models intervene?
        """
    ),
    code(
        """
        label_by_item = movie_db.dropna(subset=["item_id"]).set_index("item_id").to_dict("index")

        def item_flag(item_id, flag):
            row = label_by_item.get(item_id)
            return bool(row.get(flag, False)) if row else False

        def item_labels(item_id, col, unknown):
            row = label_by_item.get(item_id)
            if not row:
                return [unknown]
            vals = row.get(col, [])
            return vals if isinstance(vals, list) and vals else [unknown]

        base_groups = {
            "Europe wide": lambda item: item_flag(item, "is_european_wide"),
            "EU27": lambda item: item_flag(item, "is_eu27"),
            "US origin": lambda item: item_flag(item, "has_us_origin"),
            "Non-English": lambda item: item_flag(item, "is_non_english"),
            "Long-tail 20": lambda item: item_flag(item, "is_long_tail_20"),
            "Blockbuster-head": lambda item: item_flag(item, "is_blockbuster_head"),
            "Spain-origin": lambda item: item_flag(item, "is_spain_origin"),
            "Spanish-language": lambda item: item_flag(item, "is_spanish_language"),
            "European Spanish-language": lambda item: item_flag(item, "is_european_spanish_language"),
        }

        catalogue_items = set(movie_db.dropna(subset=["item_id"])["item_id"].astype(int))
        train_items_all = train_inter["item_id"].astype(int).tolist()
        test_items_all = list(chain.from_iterable(test_by_user.values()))

        def group_share(items, pred):
            items = list(items)
            if not items:
                return np.nan
            return np.mean([pred(i) for i in items])

        visibility_funnel_rows = []
        for name, pred in base_groups.items():
            user_hist_shares = [group_share(train_seen_by_user.get(u, set()), pred) for u in eval_users]
            relevant_shares = [group_share(test_by_user.get(u, set()), pred) for u in eval_users]
            visibility_funnel_rows.append({
                "group_name": name,
                "group_type": "aggregate",
                "catalogue_share": group_share(catalogue_items, pred),
                "train_interaction_share": group_share(train_items_all, pred),
                "user_history_share": np.nanmean(user_hist_shares),
                "relevant_test_share": np.nanmean(relevant_shares),
                "candidate_set_share": group_share(candidate_items, pred),
                "support_catalogue_items": int(sum(pred(i) for i in catalogue_items)),
                "support_train_interactions": int(sum(pred(i) for i in train_items_all)),
                "support_test_interactions": int(sum(pred(i) for i in test_items_all)),
                "support_users": int(sum(group_share(test_by_user.get(u, set()), pred) > 0 for u in eval_users)),
            })

        visibility_funnel = pd.DataFrame(visibility_funnel_rows)
        base_group_baselines = visibility_funnel.set_index("group_name").to_dict("index")
        save_table(visibility_funnel, "visibility_funnel_baselines.csv")
        display(visibility_funnel)

        fig, ax = plt.subplots(figsize=(11, 5))
        plot_df = visibility_funnel.set_index("group_name")[["catalogue_share", "train_interaction_share", "user_history_share", "relevant_test_share", "candidate_set_share"]]
        plot_df.plot(kind="bar", ax=ax)
        ax.set_title("Visibility funnel before algorithms")
        ax.set_ylabel("Share")
        ax.tick_params(axis="x", rotation=35)
        save_current_figure("08_visibility_funnel_baselines.png", "8. Baseline visibility", "RQ1/RQ4", "visibility_funnel_baselines.csv", "Baseline shares before model ranking.")

        funnel_lookup = visibility_funnel.set_index("group_name")
        result_box(
            "Chapter 8 result: the audit target is not a single baseline",
            [
                f"European-wide films have catalogue share {fmt_pct(funnel_lookup.loc['Europe wide','catalogue_share'])}, train-interaction share {fmt_pct(funnel_lookup.loc['Europe wide','train_interaction_share'])}, user-history share {fmt_pct(funnel_lookup.loc['Europe wide','user_history_share'])} and relevant-test share {fmt_pct(funnel_lookup.loc['Europe wide','relevant_test_share'])}.",
                f"Non-English films have catalogue share {fmt_pct(funnel_lookup.loc['Non-English','catalogue_share'])} but relevant-test share {fmt_pct(funnel_lookup.loc['Non-English','relevant_test_share'])}. This is why PACPG compares exposure to observed user interest, not only to catalogue supply.",
                f"Spain-origin and Spanish-language have different baseline shares: Spain-origin relevant-test share is {fmt_pct(funnel_lookup.loc['Spain-origin','relevant_test_share'])}; Spanish-language relevant-test share is {fmt_pct(funnel_lookup.loc['Spanish-language','relevant_test_share'])}.",
                "The figure answers: what would be visible if rankings followed catalogue, interactions, user history or test relevance instead of model scores?",
            ],
            "Negative PACPG later means underexposure relative to the stricter of user-history and relevant-test baselines."
        )
        """
    ),
    md(
        """
        **Interpretation.** These baselines define the audit target. A negative PACPG later means a group is underexposed relative to observed user history and relevant test share, not merely relative to the catalogue.
        """
    ),
    md(
        """
        ## 9. Metric definitions

        **Question.** How are utility, exposure, gaps and diversity measured?
        """
    ),
    code(
        """
        def rank_discounts(k):
            ranks = np.arange(1, k + 1)
            return 1 / np.log2(ranks + 1)

        def dcg_at_k(recommended, relevant, k):
            rec = recommended[:k]
            return sum((1 / math.log2(i + 2)) for i, item in enumerate(rec) if item in relevant)

        def ndcg_at_k(recommended, relevant, k):
            if not relevant:
                return np.nan
            ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
            return dcg_at_k(recommended, relevant, k) / ideal if ideal else np.nan

        def recall_at_k(recommended, relevant, k):
            if not relevant:
                return np.nan
            return len(set(recommended[:k]) & set(relevant)) / len(relevant)

        def map_at_k(recommended, relevant, k):
            if not relevant:
                return np.nan
            hits = 0
            total = 0
            for i, item in enumerate(recommended[:k], start=1):
                if item in relevant:
                    hits += 1
                    total += hits / i
            return total / min(len(relevant), k)

        def discounted_group_exposure(recs_by_user, pred, k=MAIN_K):
            discounts = rank_discounts(k)
            vals = []
            for recs in recs_by_user.values():
                recs = recs[:k]
                if not recs:
                    continue
                denom = discounts[:len(recs)].sum()
                vals.append(sum(discounts[i] * pred(item) for i, item in enumerate(recs)) / denom)
            return float(np.mean(vals)) if vals else np.nan

        def user_group_share(items_by_user, pred):
            vals = []
            for items in items_by_user.values():
                if len(items):
                    vals.append(np.mean([pred(i) for i in items]))
            return float(np.mean(vals)) if vals else np.nan

        def pacpg(exposure, history_share, relevant_share):
            return exposure - max(history_share, relevant_share)

        def hhi(shares):
            arr = np.array([x for x in shares if x > 0], dtype=float)
            if arr.sum() == 0:
                return np.nan
            arr = arr / arr.sum()
            return float((arr ** 2).sum())

        def normalized_entropy(shares):
            arr = np.array([x for x in shares if x > 0], dtype=float)
            if len(arr) <= 1:
                return 0.0
            arr = arr / arr.sum()
            return float(scipy_entropy(arr) / np.log(len(arr)))

        toy_recs = {1: [1, 2, 3], 2: [3, 2, 1]}
        toy_pred = lambda item: item == 1
        toy_exposure = discounted_group_exposure(toy_recs, toy_pred, k=3)
        toy_pacpg = pacpg(toy_exposure, history_share=0.5, relevant_share=0.25)
        metric_toy = pd.DataFrame([
            {"metric": "discounted exposure", "value": toy_exposure, "expected_behavior": "higher when group items are ranked earlier"},
            {"metric": "PACPG", "value": toy_pacpg, "expected_behavior": "negative means underexposure relative to max(history, relevance)"},
        ])
        save_table(metric_toy, "metric_toy_examples.csv")
        display(metric_toy)
        add_check("exposure values between 0 and 1 in toy example", 0 <= toy_exposure <= 1, f"toy exposure={toy_exposure}")
        add_check("PACPG toy example computed", not pd.isna(toy_pacpg), f"toy PACPG={toy_pacpg}")
        """
    ),
    md(
        """
        ## 10. Recommender models

        **Question.** Which feasible recommenders can be trained and evaluated inside the notebook?

        The notebook implements classical, content-based, hybrid and re-ranked models. Deep models are not faked; if they are not run in this notebook, they are recorded in the model run ledger as excluded from the final notebook comparison.
        """
    ),
    code(
        """
        def mask_seen(scores, seen_indices):
            if seen_indices:
                scores[list(seen_indices)] = -np.inf
            return scores

        user_seen_idx = {
            u: {item_to_idx[i] for i in train_seen_by_user.get(u, set()) if i in item_to_idx}
            for u in eval_users
        }

        def topk_from_scores(score_matrix, model_name, k=MAIN_K):
            recs = {}
            for u in eval_users:
                ui = user_to_idx[u]
                scores = np.asarray(score_matrix[ui]).ravel().astype(float).copy()
                mask_seen(scores, user_seen_idx.get(u, set()))
                if np.isneginf(scores).all():
                    top_idx = []
                else:
                    n = min(k, np.isfinite(scores).sum())
                    top_idx = np.argpartition(-scores, range(n))[:n]
                    top_idx = top_idx[np.argsort(-scores[top_idx])]
                recs[u] = [idx_to_item[int(i)] for i in top_idx]
            model_run_ledger.append({"model": model_name, "status": "executed", "reason": "", "dependency_or_data_issue": "", "included_in_final_comparison": True})
            return recs

        def popularity_model():
            counts = np.asarray(train_matrix.sum(axis=0)).ravel()
            return np.tile(counts, (train_matrix.shape[0], 1))

        def itemknn_model():
            item_user = train_matrix.T.tocsr()
            sim = cosine_similarity(item_user, dense_output=True)
            np.fill_diagonal(sim, 0)
            return train_matrix @ sim

        def svd_model(n_components=64):
            n_components = min(n_components, max(2, min(train_matrix.shape) - 1))
            svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
            user_factors = svd.fit_transform(train_matrix)
            item_factors = svd.components_.T
            return user_factors @ item_factors.T

        def load_embedding_matrix(path, dim_name):
            if not path:
                return None
            arr = np.load(path, mmap_mode="r")
            if arr.shape[0] <= max(candidate_items):
                display(Markdown(f"**Embedding caveat.** {dim_name} matrix has shape {arr.shape}; item IDs exceed row count. Skipping."))
                return None
            emb = np.asarray(arr[candidate_items])
            return normalize(np.nan_to_num(emb))

        mpnet_emb = load_embedding_matrix(resolved["MPNet matrix"], "MPNet")
        clip_emb = load_embedding_matrix(resolved["CLIP image matrix"], "CLIP image")

        def content_scores(emb):
            if emb is None:
                return None
            scores = np.zeros((len(eval_users), len(candidate_items)), dtype=np.float32)
            for u in eval_users:
                ui = user_to_idx[u]
                seen = [item_to_idx[i] for i in train_seen_by_user.get(u, set()) if i in item_to_idx]
                if not seen:
                    continue
                profile = emb[seen].mean(axis=0, keepdims=True)
                profile = normalize(profile)
                scores[ui] = profile @ emb.T
            return scores

        def minmax_rows(matrix):
            m = np.asarray(matrix, dtype=np.float32)
            out = np.zeros_like(m)
            for i in range(m.shape[0]):
                row = m[i]
                finite = np.isfinite(row)
                if finite.any():
                    lo, hi = row[finite].min(), row[finite].max()
                    out[i, finite] = (row[finite] - lo) / (hi - lo + 1e-9)
            return out

        score_matrices = {}
        recs_by_model = {}

        for name, fn in [
            ("Popularity", popularity_model),
            ("ItemKNN", itemknn_model),
            ("SVD", svd_model),
        ]:
            start = time.time()
            try:
                scores = fn()
                score_matrices[name] = scores
                recs_by_model[name] = topk_from_scores(scores, name, k=max(K_VALUES))
                display(Markdown(f"**{name} executed** in {time.time() - start:.1f}s."))
            except Exception as exc:
                model_run_ledger.append({"model": name, "status": "failed", "reason": str(exc), "dependency_or_data_issue": type(exc).__name__, "included_in_final_comparison": False})

        for name, emb in [("MPNet-content", mpnet_emb), ("CLIP-image-content", clip_emb)]:
            if emb is None:
                model_run_ledger.append({"model": name, "status": "skipped", "reason": "Embedding matrix unavailable or incompatible", "dependency_or_data_issue": "missing embeddings", "included_in_final_comparison": False})
                continue
            scores = content_scores(emb)
            score_matrices[name] = scores
            recs_by_model[name] = topk_from_scores(scores, name, k=max(K_VALUES))

        if "SVD" in score_matrices and "MPNet-content" in score_matrices and "CLIP-image-content" in score_matrices:
            hybrid_scores = 0.50 * minmax_rows(score_matrices["SVD"]) + 0.25 * minmax_rows(score_matrices["MPNet-content"]) + 0.25 * minmax_rows(score_matrices["CLIP-image-content"])
            score_matrices["Hybrid"] = hybrid_scores
            recs_by_model["Hybrid"] = topk_from_scores(hybrid_scores, "Hybrid", k=max(K_VALUES))
        else:
            model_run_ledger.append({"model": "Hybrid", "status": "skipped", "reason": "Requires SVD, MPNet and CLIP scores", "dependency_or_data_issue": "missing component scores", "included_in_final_comparison": False})

        for optional in ["BPR-MF", "LightGCN-style", "NeuMF-lite", "MultiVAE-lite"]:
            model_run_ledger.append({"model": optional, "status": "not run in final notebook", "reason": "Deep variants are documented in separate feedback script; final notebook keeps runtime bounded.", "dependency_or_data_issue": "runtime/dependency scope", "included_in_final_comparison": False})

        model_run_ledger_df = pd.DataFrame(model_run_ledger)
        save_table(model_run_ledger_df, "model_run_ledger.csv")
        display(model_run_ledger_df)

        add_check("recommendation list length is correct", all(len(v) <= max(K_VALUES) for v in recs_by_model.get("Popularity", {}).values()), "Top-K lists bounded by max K")
        add_check("seen items excluded from recommendations", all(not (set(recs) & train_seen_by_user.get(u, set())) for u, recs in recs_by_model.get("Popularity", {}).items()), "Popularity recommendations checked for seen item exclusion")
        """
    ),
    md(
        """
        ## 11. Static recommender audit: aggregate groups

        **Question.** Do models underexpose Europe, non-English and long-tail films, and does the best utility model also have the best visibility?
        """
    ),
    code(
        """
        test_items_by_user = {u: test_by_user.get(u, set()) for u in eval_users}
        train_items_by_user = {u: train_seen_by_user.get(u, set()) for u in eval_users}

        def evaluate_model(model_name, recs):
            row = {"Model": model_name}
            for k in K_VALUES:
                row[f"NDCG@{k}"] = np.nanmean([ndcg_at_k(recs.get(u, []), test_items_by_user.get(u, set()), k) for u in eval_users])
                row[f"Recall@{k}"] = np.nanmean([recall_at_k(recs.get(u, []), test_items_by_user.get(u, set()), k) for u in eval_users])
            row["MAP@20"] = np.nanmean([map_at_k(recs.get(u, []), test_items_by_user.get(u, set()), MAIN_K) for u in eval_users])
            row["Coverage@20"] = len(set(chain.from_iterable([r[:MAIN_K] for r in recs.values()]))) / len(candidate_items)
            pop_lookup = item_pop.set_index("item_id")["train_item_count"].to_dict()
            row["Mean recommendation popularity@20"] = np.mean([pop_lookup.get(i, 0) for i in chain.from_iterable([r[:MAIN_K] for r in recs.values()])])
            for group, pred in base_groups.items():
                exp = discounted_group_exposure(recs, pred, MAIN_K)
                baseline = base_group_baselines[group]
                hist = baseline["user_history_share"]
                rel = baseline["relevant_test_share"]
                cat = baseline["catalogue_share"]
                inter_share = baseline["train_interaction_share"]
                row[f"{group} Exposure@20"] = exp
                row[f"{group} CatalogueGap@20"] = exp - cat
                row[f"{group} InteractionGap@20"] = exp - inter_share
                row[f"{group} RelevantTestGap@20"] = exp - rel
                row[f"{group} PACPG@20"] = pacpg(exp, hist, rel)
            return row

        model_metrics = pd.DataFrame([evaluate_model(name, recs) for name, recs in recs_by_model.items()])
        save_table(model_metrics, "model_utility_metrics.csv")
        aggregate_cols = ["Model", "NDCG@20", "Recall@20", "MAP@20", "Coverage@20", "Europe wide Exposure@20", "Non-English Exposure@20", "Long-tail 20 Exposure@20", "Europe wide PACPG@20", "Non-English PACPG@20", "Long-tail 20 PACPG@20"]
        aggregate_visibility = model_metrics[[c for c in aggregate_cols if c in model_metrics.columns]]
        save_table(aggregate_visibility, "aggregate_visibility_metrics.csv")
        display(aggregate_visibility)

        fig, ax = plt.subplots(figsize=(9, 5))
        ax.scatter(model_metrics["NDCG@20"], model_metrics["Europe wide Exposure@20"], s=90, color="#167f86")
        for _, row in model_metrics.iterrows():
            ax.text(row["NDCG@20"], row["Europe wide Exposure@20"], row["Model"], fontsize=9)
        ax.set_title("Accuracy is not visibility: NDCG@20 vs European exposure")
        ax.set_xlabel("NDCG@20")
        ax.set_ylabel("European Exposure@20")
        save_current_figure("11_accuracy_vs_european_exposure.png", "11. Static recommender audit", "RQ1/RQ5", "aggregate_visibility_metrics.csv", "Utility and European exposure are plotted together.")

        exp_plot = model_metrics.set_index("Model")[["Europe wide Exposure@20", "Non-English Exposure@20", "Long-tail 20 Exposure@20"]]
        fig, ax = plt.subplots(figsize=(10, 5))
        exp_plot.plot(kind="bar", ax=ax)
        ax.set_title("Aggregate cultural exposure by model")
        ax.set_ylabel("Discounted Exposure@20")
        ax.tick_params(axis="x", rotation=25)
        save_current_figure("11_aggregate_exposure_by_model.png", "11. Static recommender audit", "RQ1/RQ5", "aggregate_visibility_metrics.csv", "Aggregate group exposure by model.")

        best_utility = model_metrics.sort_values("NDCG@20", ascending=False).iloc[0]
        best_europe = model_metrics.sort_values("Europe wide Exposure@20", ascending=False).iloc[0]
        best_non_english = model_metrics.sort_values("Non-English Exposure@20", ascending=False).iloc[0]
        most_negative_europe_pacpg = model_metrics.sort_values("Europe wide PACPG@20").iloc[0]
        most_negative_non_english_pacpg = model_metrics.sort_values("Non-English PACPG@20").iloc[0]
        best_coverage = model_metrics.sort_values("Coverage@20", ascending=False).iloc[0]
        result_box(
            "Chapter 11 result: model quality and cultural visibility are different outcomes",
            [
                f"Best utility model: {best_utility['Model']} with NDCG@20={best_utility['NDCG@20']:.4f}.",
                f"Highest European Exposure@20: {best_europe['Model']} with {fmt_pct(best_europe['Europe wide Exposure@20'])}.",
                f"Highest Non-English Exposure@20: {best_non_english['Model']} with {fmt_pct(best_non_english['Non-English Exposure@20'])}.",
                f"Most negative Europe PACPG@20: {most_negative_europe_pacpg['Model']} ({fmt_pp(most_negative_europe_pacpg['Europe wide PACPG@20'])}).",
                f"Most negative Non-English PACPG@20: {most_negative_non_english_pacpg['Model']} ({fmt_pp(most_negative_non_english_pacpg['Non-English PACPG@20'])}).",
                f"Broadest catalogue coverage: {best_coverage['Model']} with Coverage@20={fmt_pct(best_coverage['Coverage@20'])}.",
                "The scatter plot is the key governance result here: high NDCG does not automatically imply balanced cultural prominence.",
            ],
            "This is an offline sampled audit. The result supports model comparison, not a claim about any real streaming platform interface."
        )
        """
    ),
    md(
        """
        ## 12. Which Europe gets recommended?

        **Question.** Which European countries and regions are visible or underexposed in Top-K rankings?
        """
    ),
    code(
        """
        top_eu_countries = european_country[european_country["value"] >= MIN_COUNTRY_ITEMS].head(20)["label"].tolist()
        region_labels = sorted(set(chain.from_iterable(movie_db["european_region"])))

        def label_exposure_table(labels, label_col, group_type, min_items, min_interactions, unknown):
            support_cache = {}
            for label in labels:
                mask = movie_db[label_col].apply(lambda xs, lab=label: lab in xs if isinstance(xs, list) else False)
                label_item_set = set(movie_db.loc[mask & movie_db["item_id"].notna(), "item_id"].astype(int))
                pred_cached = lambda item, items=label_item_set: item in items
                support_cache[label] = {
                    "item_set": label_item_set,
                    "support_catalogue_items": int(mask.sum()),
                    "support_train_interactions": int(movie_db.loc[mask, "rating_count_train"].sum()),
                    "support_test_interactions": int(movie_db.loc[mask, "rating_count_test"].sum()),
                    "history_share": user_group_share(train_items_by_user, pred_cached),
                    "relevant_test_share": user_group_share(test_items_by_user, pred_cached),
                    "support_users": int(sum(len(test_by_user.get(u, set()) & label_item_set) > 0 for u in eval_users)),
                }
            rows = []
            for model, recs in recs_by_model.items():
                for label in labels:
                    cached = support_cache[label]
                    label_item_set = cached["item_set"]
                    pred = lambda item, items=label_item_set: item in items
                    exposure = discounted_group_exposure(recs, pred, MAIN_K)
                    hist = cached["history_share"]
                    rel = cached["relevant_test_share"]
                    cat_items = cached["support_catalogue_items"]
                    train_support = cached["support_train_interactions"]
                    rows.append({
                        "group_name": label,
                        "group_type": group_type,
                        "model_name": model,
                        "exposure_at_20": exposure,
                        "history_share": hist,
                        "relevant_test_share": rel,
                        "PACPG_at_20": pacpg(exposure, hist, rel),
                        "support_catalogue_items": cat_items,
                        "support_train_interactions": train_support,
                        "support_test_interactions": cached["support_test_interactions"],
                        "support_users": cached["support_users"],
                        "support_recommendation_slots": int(sum(pred(i) for i in chain.from_iterable([r[:MAIN_K] for r in recs.values()]))),
                        "metadata_coverage_rate": movie_db["has_country"].mean() if group_type in ["country", "region"] else movie_db["has_language"].mean(),
                        "unknown_share": 1 - (movie_db["has_country"].mean() if group_type in ["country", "region"] else movie_db["has_language"].mean()),
                        "threshold_passed": (cat_items >= min_items) and (train_support >= min_interactions),
                        "threshold_reason_if_failed": "" if (cat_items >= min_items and train_support >= min_interactions) else "below support threshold",
                    })
            return pd.DataFrame(rows)

        country_visibility = label_exposure_table(top_eu_countries + ["Unknown country"], "origin_countries", "country", MIN_COUNTRY_ITEMS, MIN_GROUP_INTERACTIONS, "Unknown country")
        region_visibility = label_exposure_table(region_labels, "european_region", "region", MIN_COUNTRY_ITEMS, MIN_GROUP_INTERACTIONS, "Unknown Europe")
        save_table(country_visibility, "country_visibility_metrics.csv")
        save_table(region_visibility, "region_visibility_metrics.csv")
        display(country_visibility.query("threshold_passed").sort_values("PACPG_at_20").head(20))

        heat = country_visibility.query("threshold_passed").pivot_table(index="group_name", columns="model_name", values="PACPG_at_20", aggfunc="mean").fillna(0)
        if not heat.empty:
            fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(heat))))
            im = ax.imshow(heat.values, aspect="auto", cmap="coolwarm", vmin=-np.nanmax(abs(heat.values)), vmax=np.nanmax(abs(heat.values)))
            ax.set_xticks(range(len(heat.columns)))
            ax.set_xticklabels(heat.columns, rotation=35, ha="right")
            ax.set_yticks(range(len(heat.index)))
            ax.set_yticklabels(heat.index)
            ax.set_title("Country PACPG@20 by model")
            fig.colorbar(im, ax=ax, label="PACPG@20")
            save_current_figure("12_country_pacpg_heatmap.png", "12. Which Europe", "RQ2", "country_visibility_metrics.csv", "Country-level PACPG heatmap for support-passing countries.", "Low-support countries are excluded from the main heatmap but saved in CSV.")

        region_heat = region_visibility.query("threshold_passed").pivot_table(index="group_name", columns="model_name", values="PACPG_at_20", aggfunc="mean").fillna(0)
        if not region_heat.empty:
            fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(region_heat))))
            im = ax.imshow(region_heat.values, aspect="auto", cmap="coolwarm", vmin=-np.nanmax(abs(region_heat.values)), vmax=np.nanmax(abs(region_heat.values)))
            ax.set_xticks(range(len(region_heat.columns)))
            ax.set_xticklabels(region_heat.columns, rotation=35, ha="right")
            ax.set_yticks(range(len(region_heat.index)))
            ax.set_yticklabels(region_heat.index)
            ax.set_title("Region PACPG@20 by model")
            fig.colorbar(im, ax=ax, label="PACPG@20")
            save_current_figure("12_region_pacpg_heatmap.png", "12. Which Europe", "RQ2", "region_visibility_metrics.csv", "Region-level PACPG heatmap.")

        below_threshold = pd.concat([
            country_visibility[~country_visibility["threshold_passed"]],
            region_visibility[~region_visibility["threshold_passed"]],
        ], ignore_index=True)
        save_table(below_threshold, "below_threshold_groups.csv")
        display(Markdown(f"**Support caveat.** {below_threshold[['group_name','group_type']].drop_duplicates().shape[0]} country/region groups are below the configured support threshold and are not used for strong model claims."))

        country_pass = country_visibility.query("threshold_passed").copy()
        region_pass = region_visibility.query("threshold_passed").copy()
        if len(country_pass):
            country_summary = country_pass.groupby("group_name").agg(
                mean_exposure=("exposure_at_20", "mean"),
                mean_pacpg=("PACPG_at_20", "mean"),
                worst_pacpg=("PACPG_at_20", "min"),
                best_exposure=("exposure_at_20", "max"),
                history_share=("history_share", "max"),
                relevant_test_share=("relevant_test_share", "max"),
                support_users=("support_users", "max"),
                support_catalogue_items=("support_catalogue_items", "max"),
                support_train_interactions=("support_train_interactions", "max"),
                support_test_interactions=("support_test_interactions", "max"),
            ).reset_index()
            country_summary["visibility_target"] = country_summary[["history_share", "relevant_test_share"]].max(axis=1)
            country_summary["mean_gap_vs_target"] = country_summary["mean_exposure"] - country_summary["visibility_target"]

            best_model_by_country = (
                country_pass.sort_values(["group_name", "exposure_at_20"], ascending=[True, False])
                .drop_duplicates("group_name")
                [["group_name", "model_name", "exposure_at_20"]]
                .rename(columns={"model_name": "best_exposure_model", "exposure_at_20": "best_model_exposure_at_20"})
            )
            weakest_model_by_country = (
                country_pass.sort_values(["group_name", "PACPG_at_20"], ascending=[True, True])
                .drop_duplicates("group_name")
                [["group_name", "model_name", "PACPG_at_20"]]
                .rename(columns={"model_name": "weakest_pacpg_model", "PACPG_at_20": "weakest_model_pacpg_at_20"})
            )
            country_geo = (
                country_summary
                .merge(best_model_by_country, on="group_name", how="left")
                .merge(weakest_model_by_country, on="group_name", how="left")
                .merge(
                    ci_country.rename(columns={
                        "label": "group_name",
                        "share_catalogue": "catalogue_share",
                        "share_interaction": "train_interaction_share",
                    })[["group_name", "catalogue_share", "train_interaction_share"]],
                    on="group_name",
                    how="left",
                )
            )
            country_geo["european_region"] = country_geo["group_name"].map(REGION_MAP).fillna("Unknown / not mapped")
            country_geo["problem_type"] = np.select(
                [
                    (country_geo["mean_exposure"] < 0.001) & (country_geo["visibility_target"] >= 0.001),
                    country_geo["mean_pacpg"] <= -0.01,
                    country_geo["mean_pacpg"] >= 0.01,
                ],
                [
                    "near-invisible despite target signal",
                    "underexposed relative to target",
                    "overexposed relative to target",
                ],
                default="near target or low-support signal",
            )
            country_geo = country_geo.sort_values("mean_pacpg")
            save_table(country_geo, "country_geo_outcome_table.csv")
            display(country_geo[[
                "group_name", "european_region", "catalogue_share", "train_interaction_share",
                "visibility_target", "mean_exposure", "mean_pacpg", "problem_type",
                "best_exposure_model", "weakest_pacpg_model",
            ]].head(20))

            country_geo_eval = country_geo[~country_geo["group_name"].isin(["Unknown country"])].copy()
            most_visible_country = country_summary.sort_values("mean_exposure", ascending=False).iloc[0]
            most_under_country = country_summary.sort_values("mean_pacpg").iloc[0]
            most_over_country = country_summary.sort_values("mean_pacpg", ascending=False).iloc[0]

            if len(country_geo_eval):
                most_visible_country = country_geo_eval.sort_values("mean_exposure", ascending=False).iloc[0]
                most_under_country = country_geo_eval.sort_values("mean_pacpg").iloc[0]
                most_over_country = country_geo_eval.sort_values("mean_pacpg", ascending=False).iloc[0]

                fig, ax = plt.subplots(figsize=(10, max(5, 0.32 * len(country_geo_eval))))
                score_plot = country_geo_eval.sort_values("mean_pacpg")
                colors = np.where(score_plot["mean_pacpg"] < 0, "#d45a2f", "#167f86")
                ax.barh(score_plot["group_name"], score_plot["mean_pacpg"], color=colors)
                ax.axvline(0, color="#0b1f33", linewidth=1)
                ax.set_title("European country visibility scorecard: mean PACPG@20")
                ax.set_xlabel("Mean PACPG@20 across evaluated models")
                ax.set_ylabel("")
                save_current_figure(
                    "12_country_geo_scorecard.png",
                    "12. Which Europe",
                    "RQ2",
                    "country_geo_outcome_table.csv",
                    "Country-level scorecard: negative PACPG means lower exposure than the preference-adjusted target.",
                    "Support-passing countries only; country labels are Wikidata-derived proxy metadata.",
                )

                fig, ax = plt.subplots(figsize=(8.5, 6))
                scatter_vals = country_geo_eval["mean_pacpg"].to_numpy()
                max_abs = max(float(np.nanmax(np.abs(scatter_vals))) if len(scatter_vals) else 0, 0.001)
                size_base = np.sqrt(country_geo_eval["support_train_interactions"].clip(lower=1))
                sizes = 80 + 620 * (size_base - size_base.min()) / (size_base.max() - size_base.min() + 1e-9)
                sc = ax.scatter(
                    country_geo_eval["visibility_target"],
                    country_geo_eval["mean_exposure"],
                    s=sizes,
                    c=country_geo_eval["mean_pacpg"],
                    cmap="coolwarm",
                    vmin=-max_abs,
                    vmax=max_abs,
                    alpha=0.82,
                    edgecolor="white",
                    linewidth=0.8,
                )
                lim = max(country_geo_eval["visibility_target"].max(), country_geo_eval["mean_exposure"].max()) * 1.08
                ax.plot([0, lim], [0, lim], color="#0b1f33", linestyle="--", linewidth=1, label="equal exposure and target")
                ax.set_xlim(0, lim)
                ax.set_ylim(0, lim)
                ax.set_title("Country exposure versus preference-adjusted target")
                ax.set_xlabel("Target = max(history share, relevant test share)")
                ax.set_ylabel("Mean Exposure@20")
                ax.legend(loc="upper left")
                fig.colorbar(sc, ax=ax, label="Mean PACPG@20")
                label_source = pd.concat([
                    country_geo_eval.nsmallest(5, "mean_pacpg"),
                    country_geo_eval.nlargest(4, "visibility_target"),
                    country_geo_eval.nlargest(3, "mean_pacpg"),
                ]).drop_duplicates("group_name")
                label_points(ax, label_source, "visibility_target", "mean_exposure", "group_name", max_labels=10)
                save_current_figure(
                    "12_country_target_vs_exposure.png",
                    "12. Which Europe",
                    "RQ2",
                    "country_geo_outcome_table.csv",
                    "Country diagnostic plot: countries below the diagonal receive less exposure than their history/test target.",
                    "Bubble size reflects train-interaction support; this is not population-normalized geography.",
                )

                europe_country_centroids = {
                    "Albania": (20.0, 41.0), "Austria": (14.6, 47.6), "Belgium": (4.5, 50.6),
                    "Bosnia and Herzegovina": (17.7, 44.2), "Bulgaria": (25.5, 42.7),
                    "Croatia": (15.2, 45.1), "Czech Republic": (15.5, 49.8), "Czechia": (15.5, 49.8),
                    "Denmark": (10.0, 56.0), "Finland": (26.0, 64.5), "France": (2.2, 46.2),
                    "Germany": (10.4, 51.2), "Greece": (22.9, 39.1), "Hungary": (19.5, 47.1),
                    "Iceland": (-19.0, 65.0), "Ireland": (-8.2, 53.4), "Italy": (12.6, 42.8),
                    "Netherlands": (5.3, 52.1), "Norway": (8.5, 61.5), "Poland": (19.1, 52.1),
                    "Portugal": (-8.0, 39.6), "Romania": (24.9, 45.9), "Russia": (37.6, 55.8),
                    "Serbia": (20.9, 44.0), "Slovakia": (19.7, 48.7), "Slovenia": (14.9, 46.1),
                    "Spain": (-3.7, 40.4), "Sweden": (15.0, 62.0), "Switzerland": (8.2, 46.8),
                    "Turkey": (35.2, 39.0), "Ukraine": (31.2, 49.0), "United Kingdom": (-2.5, 54.0),
                }
                coords = pd.DataFrame([
                    {"group_name": country, "longitude": lon, "latitude": lat}
                    for country, (lon, lat) in europe_country_centroids.items()
                ])
                country_geo_map = country_geo_eval.merge(coords, on="group_name", how="left")
                save_table(country_geo_map, "country_geo_map_points.csv")
                map_data = country_geo_map.dropna(subset=["longitude", "latitude"]).copy()
                if len(map_data):
                    fig, ax = plt.subplots(figsize=(10, 6))
                    map_vals = map_data["mean_pacpg"].to_numpy()
                    max_abs = max(float(np.nanmax(np.abs(map_vals))) if len(map_vals) else 0, 0.001)
                    map_size_base = np.sqrt(map_data["support_train_interactions"].clip(lower=1))
                    map_sizes = 90 + 760 * (map_size_base - map_size_base.min()) / (map_size_base.max() - map_size_base.min() + 1e-9)
                    sc = ax.scatter(
                        map_data["longitude"],
                        map_data["latitude"],
                        s=map_sizes,
                        c=map_data["mean_pacpg"],
                        cmap="coolwarm",
                        vmin=-max_abs,
                        vmax=max_abs,
                        alpha=0.86,
                        edgecolor="white",
                        linewidth=0.8,
                    )
                    ax.set_title("Approximate European visibility risk map")
                    ax.set_xlabel("Approximate longitude")
                    ax.set_ylabel("Approximate latitude")
                    ax.set_xlim(-13, 42)
                    ax.set_ylim(35, 66)
                    ax.grid(True, color="#d9e2ea", linewidth=0.7)
                    fig.colorbar(sc, ax=ax, label="Mean PACPG@20")
                    label_source = pd.concat([
                        map_data.nsmallest(8, "mean_pacpg"),
                        map_data.nlargest(5, "mean_exposure"),
                    ]).drop_duplicates("group_name")
                    label_points(ax, label_source, "longitude", "latitude", "group_name", max_labels=15)
                    save_current_figure(
                        "12_country_geo_visibility_map.png",
                        "12. Which Europe",
                        "RQ2",
                        "country_geo_map_points.csv",
                        "Approximate geo view of country-level PACPG and support.",
                        "Centroids are approximate and used only for visual orientation, not geospatial measurement.",
                    )
        else:
            most_visible_country = most_under_country = most_over_country = pd.Series({"group_name": "n/a", "mean_exposure": np.nan, "mean_pacpg": np.nan})

        region_pass_for_answer = region_pass[~region_pass["group_name"].isin(["Unknown Europe"])].copy()
        if len(region_pass_for_answer):
            region_summary = region_pass_for_answer.groupby("group_name").agg(mean_exposure=("exposure_at_20", "mean"), mean_pacpg=("PACPG_at_20", "mean")).reset_index()
            most_visible_region = region_summary.sort_values("mean_exposure", ascending=False).iloc[0]
            most_under_region = region_summary.sort_values("mean_pacpg").iloc[0]
        else:
            most_visible_region = most_under_region = pd.Series({"group_name": "n/a", "mean_exposure": np.nan, "mean_pacpg": np.nan})

        if "country_geo_eval" in locals() and len(country_geo_eval):
            geo_problem_counts = country_geo_eval["problem_type"].value_counts()
            most_common_geo_problem = geo_problem_counts.index[0]
            most_common_geo_problem_n = int(geo_problem_counts.iloc[0])
            geo_problem_sentence = f"The geo scorecard flags {most_common_geo_problem_n} support-passing countries as '{most_common_geo_problem}', making the country problem visible instead of only aggregate."
        else:
            geo_problem_sentence = "The geo scorecard could not be computed because no support-passing country rows were available."

        result_box(
            "Chapter 12 result: Europe is not one visibility object",
            [
                f"Most visible support-passing European country on average: {most_visible_country['group_name']} with mean Exposure@20={fmt_pct(most_visible_country['mean_exposure'])}.",
                f"Most underexposed support-passing European country by mean PACPG@20: {most_under_country['group_name']} ({fmt_pp(most_under_country['mean_pacpg'])}).",
                f"Most overexposed support-passing European country by mean PACPG@20: {most_over_country['group_name']} ({fmt_pp(most_over_country['mean_pacpg'])}).",
                f"Most visible region: {most_visible_region['group_name']} ({fmt_pct(most_visible_region['mean_exposure'])}); weakest region by PACPG: {most_under_region['group_name']} ({fmt_pp(most_under_region['mean_pacpg'])}).",
                geo_problem_sentence,
                "The country and region heatmaps answer where European visibility concentrates, not only whether Europe as a block appears.",
            ],
            "Countries/regions below item and interaction thresholds remain documented but are not used for strong claims."
        )
        """
    ),
    md(
        """
        ## 13. Which languages get recommended?

        **Question.** Does non-English content remain visible, and which original languages receive exposure?
        """
    ),
    code(
        """
        top_languages_for_audit = catalogue_language[catalogue_language["value"] >= MIN_LANGUAGE_ITEMS].head(20)["label"].tolist()
        language_visibility = label_exposure_table(top_languages_for_audit + ["Unknown language"], "original_languages", "language", MIN_LANGUAGE_ITEMS, MIN_GROUP_INTERACTIONS, "Unknown language")
        save_table(language_visibility, "language_visibility_metrics.csv")
        display(language_visibility.query("threshold_passed").sort_values("PACPG_at_20").head(20))

        lang_heat = language_visibility.query("threshold_passed").pivot_table(index="group_name", columns="model_name", values="PACPG_at_20", aggfunc="mean").fillna(0)
        if not lang_heat.empty:
            fig, ax = plt.subplots(figsize=(10, max(4, 0.35 * len(lang_heat))))
            max_abs = np.nanmax(abs(lang_heat.values)) or 0.01
            im = ax.imshow(lang_heat.values, aspect="auto", cmap="coolwarm", vmin=-max_abs, vmax=max_abs)
            ax.set_xticks(range(len(lang_heat.columns)))
            ax.set_xticklabels(lang_heat.columns, rotation=35, ha="right")
            ax.set_yticks(range(len(lang_heat.index)))
            ax.set_yticklabels(lang_heat.index)
            ax.set_title("Language PACPG@20 by model")
            fig.colorbar(im, ax=ax, label="PACPG@20")
            save_current_figure("13_language_pacpg_heatmap.png", "13. Languages", "RQ3", "language_visibility_metrics.csv", "Language-level PACPG heatmap for support-passing languages.")

        four_way = pd.DataFrame([
            {"group": "European English", "count": int((movie_db["is_european_wide"] & movie_db["is_english_language"]).sum())},
            {"group": "European non-English", "count": int((movie_db["is_european_wide"] & movie_db["is_non_english"]).sum())},
            {"group": "Non-European English", "count": int((~movie_db["is_european_wide"] & movie_db["is_english_language"]).sum())},
            {"group": "Non-European non-English", "count": int((~movie_db["is_european_wide"] & movie_db["is_non_english"]).sum())},
        ])
        save_table(four_way, "europe_language_cross_counts.csv")
        display(four_way)

        lang_pass = language_visibility.query("threshold_passed").copy()
        if len(lang_pass):
            language_summary = lang_pass.groupby("group_name").agg(mean_exposure=("exposure_at_20", "mean"), mean_pacpg=("PACPG_at_20", "mean")).reset_index()
            most_visible_language = language_summary.sort_values("mean_exposure", ascending=False).iloc[0]
            most_under_language = language_summary.sort_values("mean_pacpg").iloc[0]
            non_english_lang_summary = language_summary[~language_summary["group_name"].isin(["English", "Unknown language"])]
            strongest_non_english_language = non_english_lang_summary.sort_values("mean_exposure", ascending=False).iloc[0] if len(non_english_lang_summary) else pd.Series({"group_name": "n/a", "mean_exposure": np.nan})
        else:
            most_visible_language = most_under_language = strongest_non_english_language = pd.Series({"group_name": "n/a", "mean_exposure": np.nan, "mean_pacpg": np.nan})

        europe_english = four_way.loc[four_way["group"].eq("European English"), "count"].iloc[0]
        europe_non_english = four_way.loc[four_way["group"].eq("European non-English"), "count"].iloc[0]
        result_box(
            "Chapter 13 result: language is a separate visibility layer, not a footnote",
            [
                f"Most visible support-passing language on average: {most_visible_language['group_name']} with mean Exposure@20={fmt_pct(most_visible_language['mean_exposure'])}.",
                f"Most underexposed support-passing language by mean PACPG@20: {most_under_language['group_name']} ({fmt_pp(most_under_language['mean_pacpg'])}).",
                f"Strongest non-English language by mean exposure: {strongest_non_english_language['group_name']} ({fmt_pct(strongest_non_english_language['mean_exposure'])}).",
                f"European catalogue content is split into {europe_english:,} English-language European films and {europe_non_english:,} non-English European films.",
                "The language heatmap therefore answers whether European visibility is culturally broad or carried mainly by a small set of languages.",
            ],
            "Original language is a proxy. It does not capture subtitles, dubbing, multilingual viewing or platform localisation."
        )
        """
    ),
    md(
        """
        **Interpretation.** Aggregate non-English exposure is only the first layer. The language heatmap shows whether visibility is spread across several languages or concentrated in a few dominant ones. Low-support languages remain in the saved CSV and should not be overclaimed.
        """
    ),
    md(
        """
        ## 14. Spain case study

        **Question.** Are Spain-origin films and Spanish-language films proportionally visible, and are they the same group?
        """
    ),
    code(
        """
        spain_groups = {
            "Spain-origin": lambda item: item_flag(item, "is_spain_origin"),
            "Spanish-language": lambda item: item_flag(item, "is_spanish_language"),
            "European Spanish-language": lambda item: item_flag(item, "is_european_spanish_language"),
            "France-origin": lambda item: "France" in item_labels(item, "origin_countries", "Unknown country"),
            "Germany-origin": lambda item: "Germany" in item_labels(item, "origin_countries", "Unknown country"),
            "Italy-origin": lambda item: "Italy" in item_labels(item, "origin_countries", "Unknown country"),
            "UK-origin": lambda item: "United Kingdom" in item_labels(item, "origin_countries", "Unknown country"),
        }
        spain_group_masks = {
            "Spain-origin": movie_db["is_spain_origin"],
            "Spanish-language": movie_db["is_spanish_language"],
            "European Spanish-language": movie_db["is_european_spanish_language"],
            "France-origin": movie_db["origin_countries"].apply(lambda xs: "France" in xs if isinstance(xs, list) else False),
            "Germany-origin": movie_db["origin_countries"].apply(lambda xs: "Germany" in xs if isinstance(xs, list) else False),
            "Italy-origin": movie_db["origin_countries"].apply(lambda xs: "Italy" in xs if isinstance(xs, list) else False),
            "UK-origin": movie_db["origin_countries"].apply(lambda xs: "United Kingdom" in xs if isinstance(xs, list) else False),
        }
        spain_support = {
            group: {
                "catalogue": int(mask.sum()),
                "train": int(movie_db.loc[mask, "rating_count_train"].sum()),
            }
            for group, mask in spain_group_masks.items()
        }
        spain_rows = []
        for model, recs in recs_by_model.items():
            for group, pred in spain_groups.items():
                exp = discounted_group_exposure(recs, pred, MAIN_K)
                hist = user_group_share(train_items_by_user, pred)
                rel = user_group_share(test_items_by_user, pred)
                spain_rows.append({
                    "group_name": group,
                    "model_name": model,
                    "catalogue_share": safe_div(spain_support[group]["catalogue"], len(catalogue_items)),
                    "train_interaction_share": safe_div(spain_support[group]["train"], len(train_items_all)),
                    "relevant_test_share": rel,
                    "exposure_at_20": exp,
                    "PACPG_at_20": pacpg(exp, hist, rel),
                    "support_catalogue_items": spain_support[group]["catalogue"],
                    "support_train_interactions": spain_support[group]["train"],
                    "threshold_passed": spain_support[group]["catalogue"] >= MIN_COUNTRY_ITEMS,
                })
        spain_case = pd.DataFrame(spain_rows)
        save_table(spain_case, "spain_case_study_metrics.csv")
        display(spain_case)

        spain_plot = spain_case[spain_case["group_name"].isin(["Spain-origin", "Spanish-language", "European Spanish-language"])].pivot_table(index="model_name", columns="group_name", values="exposure_at_20", aggfunc="mean")
        fig, ax = plt.subplots(figsize=(10, 5))
        spain_plot.plot(kind="bar", ax=ax)
        ax.set_title("Spain-origin vs Spanish-language Exposure@20 by model")
        ax.set_ylabel("Discounted Exposure@20")
        ax.tick_params(axis="x", rotation=25)
        save_current_figure("14_spain_case_exposure.png", "14. Spain case study", "RQ4", "spain_case_study_metrics.csv", "Spain-origin and Spanish-language exposure by model.")

        overlap = int((movie_db["is_spain_origin"] & movie_db["is_spanish_language"]).sum())
        only_spain_origin = int((movie_db["is_spain_origin"] & ~movie_db["is_spanish_language"]).sum())
        only_spanish_language = int((~movie_db["is_spain_origin"] & movie_db["is_spanish_language"]).sum())
        display(Markdown(f"**Spain distinction.** Spain-origin and Spanish-language are not equivalent: {overlap} films have both labels, {only_spain_origin} are Spain-origin without Spanish-language metadata, and {only_spanish_language} are Spanish-language without Spain-origin metadata."))

        spain_focus = spain_case[spain_case["group_name"].isin(["Spain-origin", "Spanish-language", "European Spanish-language"])].copy()
        best_spain_origin = spain_focus[spain_focus["group_name"].eq("Spain-origin")].sort_values("exposure_at_20", ascending=False).iloc[0]
        weakest_spain_origin = spain_focus[spain_focus["group_name"].eq("Spain-origin")].sort_values("PACPG_at_20").iloc[0]
        best_spanish_language = spain_focus[spain_focus["group_name"].eq("Spanish-language")].sort_values("exposure_at_20", ascending=False).iloc[0]
        result_box(
            "Chapter 14 result: Spain-origin and Spanish-language lead to different audit answers",
            [
                f"Best Spain-origin exposure model: {best_spain_origin['model_name']} with Exposure@20={fmt_pct(best_spain_origin['exposure_at_20'])}.",
                f"Weakest Spain-origin PACPG model: {weakest_spain_origin['model_name']} with PACPG@20={fmt_pp(weakest_spain_origin['PACPG_at_20'])}.",
                f"Best Spanish-language exposure model: {best_spanish_language['model_name']} with Exposure@20={fmt_pct(best_spanish_language['exposure_at_20'])}.",
                f"The overlap table is the central result: {overlap:,} films have both labels, {only_spain_origin:,} are Spain-origin without Spanish-language metadata, and {only_spanish_language:,} are Spanish-language without Spain-origin.",
                "The Spain chart therefore tests two different governance questions: visibility of a production country and visibility of a language space.",
            ],
            "Spain-specific claims are support-limited and should be interpreted as a case study, not as a full national film-market audit."
        )
        """
    ),
    md(
        """
        ## 15. Mechanism: why does visibility differ?

        **Question.** Are visibility gaps associated with popularity, language, US involvement, co-production, genre or release year?
        """
    ),
    code(
        """
        mechanism_rows = []
        for group, pred in {
            "European non-English": lambda item: item_flag(item, "is_european_wide") and item_flag(item, "is_non_english"),
            "Spain-origin": lambda item: item_flag(item, "is_spain_origin"),
            "Spanish-language European": lambda item: item_flag(item, "is_european_spanish_language"),
            "Long-tail Europe": lambda item: item_flag(item, "is_european_wide") and item_flag(item, "is_long_tail_20"),
            "European with US company": lambda item: item_flag(item, "is_european_wide") and item_flag(item, "has_us_company_involvement"),
        }.items():
            item_ids = [i for i in catalogue_items if pred(i)]
            db_rows = movie_db[movie_db["item_id"].isin(item_ids)]
            mechanism_rows.append({
                "group": group,
                "items": len(db_rows),
                "mean_train_interactions": db_rows["rating_count_train"].mean(),
                "median_train_interactions": db_rows["rating_count_train"].median(),
                "mean_release_year": db_rows["release_year"].mean(),
                "long_tail_20_share": db_rows["is_long_tail_20"].mean() if len(db_rows) else np.nan,
                "english_language_share": db_rows["is_english_language"].mean() if len(db_rows) else np.nan,
                "us_origin_share": db_rows["has_us_origin"].mean() if len(db_rows) else np.nan,
                "us_company_share": db_rows["has_us_company_involvement"].mean() if len(db_rows) else np.nan,
                "coproduction_share": db_rows["is_coproduction"].mean() if len(db_rows) else np.nan,
            })
        mechanism_metrics = pd.DataFrame(mechanism_rows)
        save_table(mechanism_metrics, "mechanism_analysis_metrics.csv")
        display(mechanism_metrics)

        fig, ax = plt.subplots(figsize=(9, 5))
        mechanism_metrics.set_index("group")[["long_tail_20_share", "english_language_share", "us_company_share", "coproduction_share"]].plot(kind="bar", ax=ax)
        ax.set_title("Mechanism descriptors by target group")
        ax.set_ylabel("Share")
        ax.tick_params(axis="x", rotation=30)
        save_current_figure("15_mechanism_descriptors.png", "15. Mechanism", "RQ7", "mechanism_analysis_metrics.csv", "Popularity/language/US-involvement descriptors for target groups.")

        # Diagnostic model: this predicts whether an item appears at least once in Hybrid recommendations.
        recommended_hybrid = set(chain.from_iterable([r[:MAIN_K] for r in recs_by_model.get("Hybrid", next(iter(recs_by_model.values()))).values()]))
        diag = movie_db[movie_db["item_id"].isin(candidate_items)].copy()
        diag["was_recommended"] = diag["item_id"].isin(recommended_hybrid).astype(int)
        diag["log_train_interactions"] = np.log1p(diag["rating_count_train"])
        genre_lists = diag["genres"].fillna("").str.split("|")
        mlb = MultiLabelBinarizer()
        genre_mat = pd.DataFrame(mlb.fit_transform(genre_lists), columns=[f"genre_{g}" for g in mlb.classes_], index=diag.index)
        feature_cols = ["log_train_interactions", "release_year", "is_european_wide", "is_non_english", "is_spain_origin", "is_spanish_language", "has_us_origin", "has_us_company_involvement", "is_coproduction", "has_mpnet_matrix", "has_clip_matrix"]
        X = pd.concat([diag[feature_cols].fillna(0).astype(float), genre_mat], axis=1)
        y = diag["was_recommended"].astype(int)
        if y.nunique() == 2 and y.sum() >= 10:
            clf = LogisticRegression(max_iter=1000, class_weight="balanced")
            clf.fit(StandardScaler(with_mean=False).fit_transform(X), y)
            coef = pd.DataFrame({"feature": X.columns, "coefficient": clf.coef_[0]}).sort_values("coefficient", ascending=False)
            save_table(coef, "diagnostic_visibility_model_coefficients.csv")
            display(coef.head(20))
        else:
            coef = pd.DataFrame(columns=["feature", "coefficient"])
            display(Markdown("**Diagnostic model skipped.** Recommendation positives are too sparse or single-class for logistic regression."))

        # Matched comparison: European non-English target vs non-target controls matched on popularity/year/rating.
        target = diag[diag["is_european_wide"] & diag["is_non_english"]].copy()
        control = diag[~(diag["is_european_wide"] & diag["is_non_english"])].copy()
        matched_rows = []
        if len(target) >= 30 and len(control) >= 30:
            match_features = ["log_train_interactions", "release_year", "mean_rating"]
            scaler = StandardScaler()
            control_x = scaler.fit_transform(control[match_features].fillna(0))
            target_x = scaler.transform(target[match_features].fillna(0))
            nn = NearestNeighbors(n_neighbors=1).fit(control_x)
            dist, idx = nn.kneighbors(target_x)
            controls = control.iloc[idx.ravel()].copy()
            matched_rows.append({
                "target_group": "European non-English",
                "target_items": len(target),
                "control_items": len(controls),
                "target_recommended_share": target["was_recommended"].mean(),
                "matched_control_recommended_share": controls["was_recommended"].mean(),
                "difference": target["was_recommended"].mean() - controls["was_recommended"].mean(),
                "matching_features": ", ".join(match_features),
            })
        matched_comparison = pd.DataFrame(matched_rows)
        save_table(matched_comparison, "matched_pair_visibility_comparison.csv")
        display(matched_comparison)

        most_long_tail_mechanism = mechanism_metrics.sort_values("long_tail_20_share", ascending=False).iloc[0]
        lowest_interaction_mechanism = mechanism_metrics.sort_values("median_train_interactions").iloc[0]
        top_diag_feature = coef.iloc[0] if len(coef) else pd.Series({"feature": "diagnostic model skipped", "coefficient": np.nan})
        matched_sentence = "Matched comparison was not feasible with the configured support." if matched_comparison.empty else (
            f"Matched European non-English items have recommendation share {fmt_pct(matched_comparison.iloc[0]['target_recommended_share'])} versus matched controls {fmt_pct(matched_comparison.iloc[0]['matched_control_recommended_share'])}, a difference of {fmt_pp(matched_comparison.iloc[0]['difference'])}."
        )
        result_box(
            "Chapter 15 result: visibility gaps are connected to mechanisms, but this is not causal proof",
            [
                f"Highest long-tail concentration among target groups: {most_long_tail_mechanism['group']} with {fmt_pct(most_long_tail_mechanism['long_tail_20_share'])}.",
                f"Lowest median train interactions among target groups: {lowest_interaction_mechanism['group']} with median {lowest_interaction_mechanism['median_train_interactions']:.1f}.",
                f"Top diagnostic visibility feature: {top_diag_feature['feature']} (coefficient={top_diag_feature['coefficient']:.3f}) if the diagnostic model ran.",
                matched_sentence,
                "The mechanism figure tells us which group differences are plausible explanations to discuss: popularity, language, US involvement and co-production rather than an undefined cultural-bias claim.",
            ],
            "The diagnostic model predicts recommendation appearance inside this offline setup. It is not causal evidence and not a user-behaviour model."
        )
        """
    ),
    md(
        """
        **Interpretation.** This section is diagnostic, not causal. It asks whether visible items differ systematically by popularity, language, US involvement, co-production, release year or genre. Any coefficient or matched difference is a lead for interpretation, not proof of why a user would click.
        """
    ),
    md(
        """
        ## 16. Multimodal audit

        **Question.** Do MPNet plot embeddings and CLIP poster embeddings help, hurt or shift cultural visibility?
        """
    ),
    code(
        """
        multimodal_rows = []
        for name in ["SVD", "MPNet-content", "CLIP-image-content", "Hybrid"]:
            if name in model_metrics["Model"].values:
                row = model_metrics[model_metrics["Model"].eq(name)].iloc[0]
                multimodal_rows.append({
                    "Model": name,
                    "NDCG@20": row["NDCG@20"],
                    "Coverage@20": row["Coverage@20"],
                    "Europe wide Exposure@20": row["Europe wide Exposure@20"],
                    "Non-English Exposure@20": row["Non-English Exposure@20"],
                    "Spain-origin Exposure@20": row.get("Spain-origin Exposure@20", np.nan),
                    "Mean recommendation popularity@20": row["Mean recommendation popularity@20"],
                })
        multimodal_audit = pd.DataFrame(multimodal_rows)
        save_table(multimodal_audit, "multimodal_audit_metrics.csv")
        display(multimodal_audit)

        if len(multimodal_audit):
            fig, ax = plt.subplots(figsize=(9, 5))
            multimodal_audit.set_index("Model")[["Europe wide Exposure@20", "Non-English Exposure@20", "Coverage@20"]].plot(kind="bar", ax=ax)
            ax.set_title("Multimodal models: coverage and cultural exposure")
            ax.set_ylabel("Metric value")
            ax.tick_params(axis="x", rotation=25)
            save_current_figure("16_multimodal_comparison.png", "16. Multimodal audit", "RQ6", "multimodal_audit_metrics.csv", "Comparison of CF, text, image and hybrid models.")

        def embedding_homophily(emb, label_func, label_name, n_items=1000):
            if emb is None:
                return {"label": label_name, "status": "skipped", "same_label_rate": np.nan, "reason": "embedding unavailable"}
            rng = np.random.default_rng(RANDOM_SEED)
            ids = np.array(candidate_items)
            if len(ids) > n_items:
                sample_pos = rng.choice(np.arange(len(ids)), n_items, replace=False)
            else:
                sample_pos = np.arange(len(ids))
            e = emb[sample_pos]
            labels = [label_func(int(ids[i])) for i in sample_pos]
            sim = cosine_similarity(e)
            np.fill_diagonal(sim, -np.inf)
            nn_idx = sim.argmax(axis=1)
            same = np.mean([labels[i] == labels[j] for i, j in enumerate(nn_idx)])
            return {"label": label_name, "status": "computed", "same_label_rate": same, "reason": ""}

        homophily = pd.DataFrame([
            embedding_homophily(mpnet_emb, lambda i: item_flag(i, "is_european_wide"), "MPNet European homophily"),
            embedding_homophily(mpnet_emb, lambda i: item_labels(i, "original_languages", "Unknown language")[0], "MPNet language homophily"),
            embedding_homophily(clip_emb, lambda i: item_flag(i, "is_european_wide"), "CLIP European homophily"),
            embedding_homophily(clip_emb, lambda i: item_labels(i, "original_languages", "Unknown language")[0], "CLIP language homophily"),
        ])
        save_table(homophily, "embedding_homophily_audit.csv")
        display(homophily)

        if len(multimodal_audit):
            best_mm_ndcg = multimodal_audit.sort_values("NDCG@20", ascending=False).iloc[0]
            best_mm_europe = multimodal_audit.sort_values("Europe wide Exposure@20", ascending=False).iloc[0]
            best_mm_non_english = multimodal_audit.sort_values("Non-English Exposure@20", ascending=False).iloc[0]
            best_mm_coverage = multimodal_audit.sort_values("Coverage@20", ascending=False).iloc[0]
            result_box(
                "Chapter 16 result: multimodal features change the visibility mix; they are not automatically a fairness fix",
                [
                    f"Best multimodal-section utility: {best_mm_ndcg['Model']} with NDCG@20={best_mm_ndcg['NDCG@20']:.4f}.",
                    f"Highest European exposure in this comparison: {best_mm_europe['Model']} with {fmt_pct(best_mm_europe['Europe wide Exposure@20'])}.",
                    f"Highest non-English exposure in this comparison: {best_mm_non_english['Model']} with {fmt_pct(best_mm_non_english['Non-English Exposure@20'])}.",
                    f"Highest coverage in this comparison: {best_mm_coverage['Model']} with Coverage@20={fmt_pct(best_mm_coverage['Coverage@20'])}.",
                    "The homophily table checks whether embeddings contain country/language structure before ranking; if same-label rates are high, content vectors can carry cultural clustering into recommendation.",
                ],
                "Embedding effects depend on feature coverage and candidate sampling. The notebook reports shifts, not universal model behaviour."
            )
        """
    ),
    md(
        """
        ## 17. User-fold robustness check, not full retraining cross-validation

        **Question.** Are the main utility and visibility conclusions stable across evaluated user subsets?

        This is a bounded user-fold robustness check, not a full retraining cross-validation. It reuses the already trained score matrices and evaluates metric stability across user folds. That is sufficient for a notebook-scale robustness check, but it should not be presented as full model retraining per fold.
        """
    ),
    code(
        """
        cv_rows = []
        if RUN_CROSS_VALIDATION and len(eval_users) >= CV_FOLDS * 50:
            cv_users = eval_users[:min(len(eval_users), CV_SAMPLE_USERS)]
            kf = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_SEED)
            for fold_id, (_, fold_idx) in enumerate(kf.split(cv_users), start=1):
                fold_users = [cv_users[i] for i in fold_idx]
                # Reuse trained score matrices but evaluate separate users. This is a bounded robustness check, not a full retraining CV.
                for model, recs in recs_by_model.items():
                    fold_recs = {u: recs[u] for u in fold_users if u in recs}
                    fold_test = {u: test_items_by_user[u] for u in fold_users if u in fold_recs}
                    if not fold_recs:
                        continue
                    row = {"fold": fold_id, "Model": model}
                    row["NDCG@20"] = np.nanmean([ndcg_at_k(fold_recs[u], fold_test[u], MAIN_K) for u in fold_recs])
                    row["Recall@20"] = np.nanmean([recall_at_k(fold_recs[u], fold_test[u], MAIN_K) for u in fold_recs])
                    row["Europe wide Exposure@20"] = discounted_group_exposure(fold_recs, base_groups["Europe wide"], MAIN_K)
                    row["Non-English Exposure@20"] = discounted_group_exposure(fold_recs, base_groups["Non-English"], MAIN_K)
                    row["Long-tail 20 Exposure@20"] = discounted_group_exposure(fold_recs, base_groups["Long-tail 20"], MAIN_K)
                    cv_rows.append(row)
        else:
            filter_threshold_ledger.append({"step": "user_fold_robustness_check", "rule": "skipped", "reason": "RUN_CROSS_VALIDATION disabled or too few users"})

        cv_fold_results = pd.DataFrame(cv_rows)
        if len(cv_fold_results):
            cv_summary = cv_fold_results.groupby("Model").agg(["mean", "std"]).reset_index()
            cv_summary.columns = [" ".join([str(x) for x in col if x]).strip() for col in cv_summary.columns]
        else:
            cv_summary = pd.DataFrame()
        save_table(cv_fold_results, "cross_validation_fold_results.csv")
        save_table(cv_summary, "cross_validation_summary.csv")
        save_table(cv_fold_results, "user_fold_robustness_fold_results.csv")
        save_table(cv_summary, "user_fold_robustness_summary.csv")
        display(cv_summary)

        if len(cv_fold_results):
            fig, ax = plt.subplots(figsize=(10, 5))
            cv_fold_results.groupby("Model")["NDCG@20"].mean().sort_values().plot(kind="barh", ax=ax, color="#167f86", xerr=cv_fold_results.groupby("Model")["NDCG@20"].std().reindex(cv_fold_results.groupby("Model")["NDCG@20"].mean().sort_values().index))
            ax.set_title("User-fold robustness check: NDCG@20 mean +/- std")
            ax.set_xlabel("NDCG@20")
            save_current_figure("17_cv_ndcg_stability.png", "17. User-fold robustness", "Robustness", "user_fold_robustness_summary.csv", "Fold-level stability for NDCG@20 across evaluated user subsets.", "This bounded check reuses trained scores; it is not full retraining cross-validation.")

            cv_mean = cv_fold_results.groupby("Model").agg(
                ndcg_mean=("NDCG@20", "mean"),
                ndcg_std=("NDCG@20", "std"),
                europe_mean=("Europe wide Exposure@20", "mean"),
                non_english_mean=("Non-English Exposure@20", "mean"),
            ).reset_index()
            stable_utility = cv_mean.sort_values("ndcg_mean", ascending=False).iloc[0]
            stable_europe = cv_mean.sort_values("europe_mean", ascending=False).iloc[0]
            result_box(
                "Chapter 17 result: the main comparison is directionally robust in the bounded user-fold check",
                [
                    f"Highest mean fold NDCG@20: {stable_utility['Model']} ({stable_utility['ndcg_mean']:.4f}, std={stable_utility['ndcg_std']:.4f}).",
                    f"Highest mean fold European Exposure@20: {stable_europe['Model']} ({fmt_pct(stable_europe['europe_mean'])}).",
                    "The robustness chart asks whether model ranking is a one-split accident. Large error bars would weaken the claim.",
                ],
                "This is a bounded user-fold check that reuses trained scores. It is suitable for notebook-scale robustness, not full retraining cross-validation."
            )
        else:
            result_box(
                "Chapter 17 result: robustness check not run",
                ["Cross-validation was skipped because the run configuration or sample size did not meet the threshold."],
                "Switch RUN_CROSS_VALIDATION=True and increase sample size for a heavier run."
            )
        """
    ),
    md(
        """
        **Caveat.** This section should be described as a **user-fold robustness check**, not full cross-validation. A full retraining CV would rebuild the train/test matrices and refit Popularity, ItemKNN, SVD and Hybrid inside every fold. The current notebook tests whether the observed metric patterns are stable across evaluated user subsets.
        """
    ),
    md(
        """
        ## 18. Schedl-inspired feedback-loop stress test

        **Question.** Do visibility gaps compound under repeated recommendation and simulated consumption?

        This section is inspired by Lesota, Geiger, Walder, Kowald and Schedl (2024), but it is not a full reproduction of their feedback-loop protocol. The bounded notebook run updates user profiles and seen-item masks over iterations; it does not retrain every model after every simulated interaction. Therefore, the result is a **dynamic stress test** rather than full dynamic evidence.
        """
    ),
    code(
        """
        feedback_methodology_comparison = pd.DataFrame([
            {"feature": "Repeated recommendations", "Full Schedl-style design": "yes", "Our bounded notebook run": "yes"},
            {"feature": "Simulated accepted item", "Full Schedl-style design": "yes", "Our bounded notebook run": "yes, rank-biased default"},
            {"feature": "Profile update", "Full Schedl-style design": "yes", "Our bounded notebook run": "yes"},
            {"feature": "Retraining after each iteration", "Full Schedl-style design": "yes", "Our bounded notebook run": "no; scores are reused and seen-item masks/profile sets are updated"},
            {"feature": "Multiple choice models", "Full Schedl-style design": "paper-dependent", "Our bounded notebook run": "rank-biased default"},
            {"feature": "Interpretation", "Full Schedl-style design": "dynamic feedback-loop evidence", "Our bounded notebook run": "dynamic stress test for representation drift"},
        ])
        save_table(feedback_methodology_comparison, "feedback_loop_methodology_comparison.csv")
        display(feedback_methodology_comparison)

        feedback_rows = []
        if RUN_FEEDBACK_LOOP:
            fb_users = eval_users[:min(len(eval_users), FEEDBACK_SAMPLE_USERS)]
            fb_profiles = {u: set(train_seen_by_user.get(u, set())) & set(candidate_items) for u in fb_users}
            fb_models = [m for m in ["Popularity", "SVD", "Hybrid"] if m in score_matrices]
            for model in fb_models:
                profiles = {u: set(items) for u, items in fb_profiles.items()}
                for iteration in range(1, FEEDBACK_ITERATIONS + 1):
                    # Re-score with a lightweight profile update. Popularity is static; SVD/Hybrid scores are reused and seen masks change.
                    base_scores = score_matrices[model]
                    recs_iter = {}
                    for u in fb_users:
                        ui = user_to_idx[u]
                        scores = np.asarray(base_scores[ui]).ravel().copy()
                        seen_idx = {item_to_idx[i] for i in profiles[u] if i in item_to_idx}
                        mask_seen(scores, seen_idx)
                        n = min(MAIN_K, np.isfinite(scores).sum())
                        if n <= 0:
                            recs_iter[u] = []
                            continue
                        top_idx = np.argpartition(-scores, range(n))[:n]
                        top_idx = top_idx[np.argsort(-scores[top_idx])]
                        recs_iter[u] = [idx_to_item[int(i)] for i in top_idx]
                    for metric_name, pred in base_groups.items():
                        if metric_name in ["Europe wide", "Non-English", "Long-tail 20", "Spain-origin", "Spanish-language"]:
                            feedback_rows.append({
                                "iteration": iteration,
                                "Model": model,
                                "group_name": metric_name,
                                "recommendation_exposure": discounted_group_exposure(recs_iter, pred, MAIN_K),
                                "profile_share": user_group_share(profiles, pred),
                            })
                    # Rank-biased choice: accept top item with decaying rank probability.
                    for u, recs in recs_iter.items():
                        if not recs:
                            continue
                        alpha = -0.2
                        ranks = np.arange(len(recs))
                        probs = np.exp(alpha * ranks)
                        probs = probs / probs.sum()
                        accepted = np.random.default_rng(RANDOM_SEED + iteration + u).choice(recs, p=probs)
                        profiles[u].add(int(accepted))
        else:
            filter_threshold_ledger.append({"step": "feedback_loop", "rule": "skipped", "reason": "RUN_FEEDBACK_LOOP disabled"})

        feedback_loop_summary = pd.DataFrame(feedback_rows)
        save_table(feedback_loop_summary, "feedback_loop_summary.csv")
        display(feedback_loop_summary.head())

        if len(feedback_loop_summary):
            fig, ax = plt.subplots(figsize=(10, 5))
            plot_fb = feedback_loop_summary[feedback_loop_summary["group_name"].isin(["Europe wide", "Non-English", "Spain-origin"])]
            for (model, group), grp in plot_fb.groupby(["Model", "group_name"]):
                ax.plot(grp["iteration"], grp["recommendation_exposure"], marker="o", label=f"{model}: {group}")
            ax.set_title("Feedback-loop exposure over iterations")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Exposure@20")
            ax.legend(fontsize=8)
            save_current_figure("18_feedback_loop_exposure.png", "18. Feedback loop", "RQ8", "feedback_loop_summary.csv", "Exposure drift under repeated recommendation.", "Lightweight loop reuses score matrices and updates seen profiles for runtime.")

            fb_first_last = feedback_loop_summary.pivot_table(index=["Model", "group_name"], columns="iteration", values="recommendation_exposure", aggfunc="mean")
            fb_drift = (fb_first_last[FEEDBACK_ITERATIONS] - fb_first_last[1]).reset_index(name="exposure_drift")
            largest_positive_drift = fb_drift.sort_values("exposure_drift", ascending=False).iloc[0]
            largest_negative_drift = fb_drift.sort_values("exposure_drift").iloc[0]
            result_box(
                "Chapter 18 result: repeated recommendation can create visibility drift in the stress test",
                [
                    f"Largest positive exposure drift: {largest_positive_drift['Model']} / {largest_positive_drift['group_name']} ({fmt_pp(largest_positive_drift['exposure_drift'])}).",
                    f"Largest negative exposure drift: {largest_negative_drift['Model']} / {largest_negative_drift['group_name']} ({fmt_pp(largest_negative_drift['exposure_drift'])}).",
                    "The feedback-loop plot answers whether under- or over-exposure is stable, amplified, or dampened when recommendations become new profile history.",
                ],
                "This is a Schedl-inspired lightweight stress test. It updates profiles and seen masks, but does not fully retrain models after each simulated interaction."
            )
        else:
            result_box(
                "Chapter 18 result: feedback loop not run",
                ["The feedback-loop section was skipped by configuration."],
                "Enable RUN_FEEDBACK_LOOP for the dynamic drift audit."
            )
        """
    ),
    md(
        """
        ## 19. Governance-aware re-ranking

        **Question.** Can transparent re-ranking improve visibility, and what utility cost does it create?
        """
    ),
    code(
        """
        def governance_bonus(item, variant="combined"):
            bonus = 0.0
            if variant in ["combined", "europe"] and item_flag(item, "is_european_wide"):
                bonus += 1.0
            if variant in ["combined", "non_english"] and item_flag(item, "is_non_english"):
                bonus += 1.0
            if variant in ["combined", "long_tail"] and item_flag(item, "is_long_tail_20"):
                bonus += 1.0
            if variant == "spain" and item_flag(item, "is_spain_origin"):
                bonus += 1.0
            return bonus

        rerank_rows = []
        rerank_recs = {}
        if "Hybrid" in score_matrices:
            base_scores = minmax_rows(score_matrices["Hybrid"])
            for variant in ["combined", "europe", "non_english", "long_tail", "spain"]:
                for lam in LAMBDAS:
                    recs = {}
                    for u in eval_users:
                        ui = user_to_idx[u]
                        scores = base_scores[ui].copy()
                        bonuses = np.array([governance_bonus(idx_to_item[i], variant) for i in range(len(candidate_items))])
                        final_scores = scores + lam * bonuses
                        mask_seen(final_scores, user_seen_idx.get(u, set()))
                        n = min(max(K_VALUES), np.isfinite(final_scores).sum())
                        top_idx = np.argpartition(-final_scores, range(n))[:n]
                        top_idx = top_idx[np.argsort(-final_scores[top_idx])]
                        recs[u] = [idx_to_item[int(i)] for i in top_idx]
                    row = evaluate_model(f"Hybrid rerank {variant} lambda={lam}", recs)
                    row["variant"] = variant
                    row["lambda"] = lam
                    row["Spain-origin Exposure@20"] = discounted_group_exposure(recs, base_groups["Spain-origin"], MAIN_K)
                    row["Spanish-language Exposure@20"] = discounted_group_exposure(recs, base_groups["Spanish-language"], MAIN_K)
                    rerank_rows.append(row)
                    if variant == "combined" and lam in [0.0, 0.3, 0.7, 1.0]:
                        rerank_recs[row["Model"]] = recs
        else:
            filter_threshold_ledger.append({"step": "reranking", "rule": "skipped", "reason": "Hybrid base model unavailable"})

        reranking_frontier = pd.DataFrame(rerank_rows)
        save_table(reranking_frontier, "reranking_frontier.csv")
        display(reranking_frontier[["Model", "variant", "lambda", "NDCG@20", "Recall@20", "Europe wide Exposure@20", "Non-English Exposure@20", "Long-tail 20 Exposure@20", "Spain-origin Exposure@20", "Spanish-language Exposure@20"]].head(20))

        if len(reranking_frontier):
            combined = reranking_frontier[reranking_frontier["variant"].eq("combined")]
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(combined["lambda"], combined["NDCG@20"], marker="o", label="NDCG@20")
            ax.plot(combined["lambda"], combined["Europe wide Exposure@20"], marker="o", label="European exposure")
            ax.plot(combined["lambda"], combined["Non-English Exposure@20"], marker="o", label="Non-English exposure")
            ax.set_title("Transparent re-ranking frontier")
            ax.set_xlabel("lambda")
            ax.set_ylabel("Metric")
            ax.legend()
            save_current_figure("19_reranking_frontier.png", "19. Re-ranking", "RQ9", "reranking_frontier.csv", "Utility and visibility response to lambda.")

            base_combined = combined[combined["lambda"].eq(0.0)].iloc[0]
            combined = combined.copy()
            combined["ndcg_retention"] = combined["NDCG@20"] / base_combined["NDCG@20"] if base_combined["NDCG@20"] else np.nan
            defensible = combined[combined["ndcg_retention"] >= 0.8].sort_values("Europe wide Exposure@20", ascending=False)
            best_defensible = defensible.iloc[0] if len(defensible) else combined.sort_values("NDCG@20", ascending=False).iloc[0]
            max_visibility = combined.sort_values("Europe wide Exposure@20", ascending=False).iloc[0]
            result_box(
                "Chapter 19 result: re-ranking is a governance lever because it makes the cost visible",
                [
                    f"Base Hybrid NDCG@20={base_combined['NDCG@20']:.4f} and European Exposure@20={fmt_pct(base_combined['Europe wide Exposure@20'])}.",
                    f"Best combined variant with at least 80% NDCG retention: lambda={best_defensible['lambda']} with NDCG@20={best_defensible['NDCG@20']:.4f} and European Exposure@20={fmt_pct(best_defensible['Europe wide Exposure@20'])}.",
                    f"Maximum European exposure in the sweep occurs at lambda={max_visibility['lambda']} with Exposure@20={fmt_pct(max_visibility['Europe wide Exposure@20'])}, but NDCG@20={max_visibility['NDCG@20']:.4f}.",
                    "The frontier plot is the answer: mitigation is possible, but the notebook makes the utility cost and overcorrection risk explicit.",
                ],
                "This does not claim fairness is solved. It shows an auditable post-processing trade-off."
            )
        else:
            result_box(
                "Chapter 19 result: re-ranking not run",
                ["The Hybrid base model was unavailable, so the lambda frontier could not be computed."],
                "Run the Hybrid model first to evaluate governance-aware re-ranking."
            )
        """
    ),
    md(
        """
        ## 20. Religion/theme exploratory check

        **Question.** Is religion-related film-theme visibility reliable enough to analyse as a main metric?
        """
    ),
    code(
        """
        religion_terms = ["christian", "catholic", "judaism", "jewish", "islam", "muslim", "buddh", "hindu", "religion", "church", "priest", "monastery", "faith", "rabbi", "mosque", "temple"]
        theme_source = movie_db[["movieId", "title", "top_tags", "top_genome_tags"]].copy()
        theme_source["theme_text"] = (theme_source["top_tags"].fillna("") + " | " + theme_source["top_genome_tags"].fillna("")).str.lower()
        theme_source["religion_theme_proxy"] = theme_source["theme_text"].apply(lambda txt: any(term in txt for term in religion_terms))
        religion_theme = theme_source[theme_source["religion_theme_proxy"]].copy()
        religion_support = pd.DataFrame([
            {"field": "user tags or genome tags", "covered_movies": int(theme_source["theme_text"].str.len().gt(3).sum()), "religion_proxy_movies": len(religion_theme), "share_of_catalogue": len(religion_theme) / len(movie_db)}
        ])
        save_table(religion_support, "religion_theme_exploratory_support.csv")
        save_table(religion_theme.head(200), "religion_theme_exploratory_examples.csv")
        display(religion_support)
        display(religion_theme[["movieId", "title", "top_tags", "top_genome_tags"]].head(20))

        if len(religion_theme) < MIN_COUNTRY_ITEMS:
            display(Markdown("**Conclusion.** Religion-related film-theme visibility is not used as a main fairness metric because the available metadata is sparse and proxy risk is high. The project therefore focuses on country, language and long-tail visibility."))
        else:
            display(Markdown("**Exploratory caveat.** Religion-related tags exist, but they describe film themes only. They must not be used to infer user religion or community fairness."))
        """
    ),
    md(
        """
        ## 21. Research extension: European Film Visibility DNA

        **Question.** Does the recommender show Europe as local culture, or mostly the globally compatible version of Europe?

        This extension does not add more models. It adds a sharper interpretive layer. We build two transparent DNA scores from the existing MovieLens/M3L/Wikidata pipeline and the new limited deeper-Wikidata cache:

        - **Global Compatibility Score:** English-language, US-origin or US-company involvement, blockbuster-head status, high MovieLens interaction signal and broad industrial compatibility.
        - **Local Europe Score:** European origin, non-English, no US-origin/company involvement, non-blockbuster status and single-origin/local production signal.

        TMDb details/watch providers and LUMIERE admissions are documented as next data layers, but they are not faked or scraped here.
        """
    ),
    code(
        """
        dna_cache_path = PROJECT_ROOT / "data" / "interim" / "wikidata_visibility_dna_extra.csv"
        dna_status_path = TABLE_DIR / "visibility_dna_enrichment_status.csv"

        if dna_cache_path.exists():
            dna_extra = pd.read_csv(dna_cache_path)
        else:
            dna_extra = pd.DataFrame(columns=[
                "wikidata_uri", "directors", "director_citizenship",
                "filming_location_country", "award_count", "award_examples",
            ])

        if dna_status_path.exists():
            dna_status = pd.read_csv(dna_status_path)
        else:
            dna_status = pd.DataFrame([
                {
                    "source": "Wikidata deeper DNA cache",
                    "status": "missing_optional_cache",
                    "rows": len(dna_extra),
                    "priority_movies_requested": 0,
                    "coverage_of_priority_movies": np.nan,
                    "notes": "Run scripts/build_visibility_dna_enrichment.py to fetch the optional deep-Wikidata cache.",
                },
                {
                    "source": "TMDb details/watch providers",
                    "status": "not_queried_missing_api_key",
                    "rows": 0,
                    "priority_movies_requested": 0,
                    "coverage_of_priority_movies": 0,
                    "notes": "TMDb IDs exist locally, but TMDb API requires credentials. No scraping or synthetic provider data was used.",
                },
                {
                    "source": "LUMIERE admissions",
                    "status": "not_integrated_validation_layer",
                    "rows": 0,
                    "priority_movies_requested": 0,
                    "coverage_of_priority_movies": 0,
                    "notes": "Recommended as a later confident-match validation subset, not as blind bulk data.",
                },
            ])
        save_table(dna_status, "visibility_dna_enrichment_status.csv")
        display(dna_status)

        dna_movies = movie_db.merge(dna_extra, on="wikidata_uri", how="left")
        for col in ["directors", "director_citizenship", "filming_location_country", "award_examples"]:
            if col not in dna_movies.columns:
                dna_movies[col] = ""
            dna_movies[col + "_list"] = dna_movies[col].apply(split_pipe)

        dna_movies["award_count"] = pd.to_numeric(dna_movies.get("award_count", 0), errors="coerce").fillna(0)
        dna_movies["has_tmdb_id"] = dna_movies["tmdbId"].notna()
        dna_movies["director_is_european"] = dna_movies["director_citizenship_list"].apply(lambda xs: any(x in WIDE_EUROPE for x in xs))
        dna_movies["filmed_in_europe"] = dna_movies["filming_location_country_list"].apply(lambda xs: any(x in WIDE_EUROPE for x in xs))
        dna_movies["has_award_signal"] = dna_movies["award_count"].gt(0)
        dna_movies["us_industrial_involvement"] = dna_movies["has_us_origin"] | dna_movies["has_us_company_involvement"]
        popularity_threshold = dna_movies["rating_count_train"].quantile(0.80)
        high_interaction = dna_movies["rating_count_train"].fillna(0) >= popularity_threshold

        # The scores are deliberately transparent. Each component is binary and auditable.
        dna_movies["global_compatibility_score"] = (
            dna_movies["is_english_language"].astype(int)
            + dna_movies["us_industrial_involvement"].astype(int)
            + dna_movies["is_blockbuster_head"].astype(int)
            + high_interaction.astype(int)
            + dna_movies["is_coproduction"].astype(int)
        ) / 5
        dna_movies["local_europe_score"] = (
            dna_movies["is_european_wide"].astype(int)
            + dna_movies["is_non_english"].astype(int)
            + (~dna_movies["us_industrial_involvement"]).astype(int)
            + (~dna_movies["is_blockbuster_head"]).astype(int)
            + (dna_movies["num_origin_countries"].fillna(0) <= 1).astype(int)
        ) / 5

        dna_movies["platform_compatible_europe"] = dna_movies["is_european_wide"] & dna_movies["global_compatibility_score"].ge(0.60)
        dna_movies["local_europe"] = dna_movies["is_european_wide"] & dna_movies["local_europe_score"].ge(0.80)
        dna_movies["local_non_english_no_us_europe"] = (
            dna_movies["is_european_wide"]
            & dna_movies["is_non_english"]
            & (~dna_movies["us_industrial_involvement"])
        )
        dna_movies["award_signal_europe"] = dna_movies["is_european_wide"] & dna_movies["has_award_signal"]
        dna_movies["director_europe_european_film"] = dna_movies["is_european_wide"] & dna_movies["director_is_european"]

        dna_cols = [
            "movieId", "item_id", "title", "is_european_wide", "is_non_english",
            "has_us_origin", "has_us_company_involvement", "is_coproduction",
            "is_blockbuster_head", "num_origin_countries", "rating_count_train",
            "directors", "director_citizenship", "filming_location_country",
            "award_count", "global_compatibility_score", "local_europe_score",
            "platform_compatible_europe", "local_europe",
            "local_non_english_no_us_europe", "award_signal_europe",
            "director_europe_european_film",
        ]
        save_table(dna_movies[[c for c in dna_cols if c in dna_movies.columns]], "visibility_dna_movie_scores.csv")

        dna_coverage = pd.DataFrame([
            {"field": "movies with TMDb ID bridge", "share": dna_movies["has_tmdb_id"].mean(), "count": int(dna_movies["has_tmdb_id"].sum())},
            {"field": "movies with deep Wikidata DNA row", "share": dna_movies["wikidata_uri"].isin(dna_extra["wikidata_uri"]).mean() if len(dna_extra) else 0, "count": int(dna_movies["wikidata_uri"].isin(dna_extra["wikidata_uri"]).sum()) if len(dna_extra) else 0},
            {"field": "DNA rows with director citizenship", "share": dna_extra["director_citizenship"].notna().mean() if len(dna_extra) and "director_citizenship" in dna_extra else 0, "count": int(dna_extra["director_citizenship"].notna().sum()) if len(dna_extra) and "director_citizenship" in dna_extra else 0},
            {"field": "DNA rows with filming-location country", "share": dna_extra["filming_location_country"].notna().mean() if len(dna_extra) and "filming_location_country" in dna_extra else 0, "count": int(dna_extra["filming_location_country"].notna().sum()) if len(dna_extra) and "filming_location_country" in dna_extra else 0},
            {"field": "DNA rows with award count", "share": dna_extra["award_count"].notna().mean() if len(dna_extra) and "award_count" in dna_extra else 0, "count": int(dna_extra["award_count"].notna().sum()) if len(dna_extra) and "award_count" in dna_extra else 0},
        ])
        save_table(dna_coverage, "visibility_dna_coverage.csv")
        display(dna_coverage)

        dna_group_masks = {
            "Platform-compatible Europe": dna_movies["platform_compatible_europe"],
            "Local Europe": dna_movies["local_europe"],
            "Local non-English no-US Europe": dna_movies["local_non_english_no_us_europe"],
            "European award-signal subset": dna_movies["award_signal_europe"],
            "European film with European director citizenship": dna_movies["director_europe_european_film"],
        }

        dna_rows = []
        for group_name, mask in dna_group_masks.items():
            group_items = set(dna_movies.loc[mask & dna_movies["item_id"].notna(), "item_id"].astype(int))
            pred = lambda item, items=group_items: item in items
            hist = user_group_share(train_items_by_user, pred)
            rel = user_group_share(test_items_by_user, pred)
            for model_name, recs in recs_by_model.items():
                exposure = discounted_group_exposure(recs, pred, MAIN_K)
                dna_rows.append({
                    "group_name": group_name,
                    "model_name": model_name,
                    "exposure_at_20": exposure,
                    "history_share": hist,
                    "relevant_test_share": rel,
                    "PACPG_at_20": pacpg(exposure, hist, rel),
                    "support_catalogue_items": int(mask.sum()),
                    "support_train_interactions": int(dna_movies.loc[mask, "rating_count_train"].sum()),
                    "support_test_interactions": int(dna_movies.loc[mask, "rating_count_test"].sum()),
                    "support_recommended_slots": int(sum(pred(i) for i in chain.from_iterable([r[:MAIN_K] for r in recs.values()]))),
                })
        visibility_dna_group_metrics = pd.DataFrame(dna_rows)
        save_table(visibility_dna_group_metrics, "visibility_dna_group_metrics.csv")
        display(visibility_dna_group_metrics.sort_values("PACPG_at_20").head(15))

        visibility_dna_group_summary = visibility_dna_group_metrics.groupby("group_name").agg(
            mean_exposure=("exposure_at_20", "mean"),
            mean_pacpg=("PACPG_at_20", "mean"),
            min_pacpg=("PACPG_at_20", "min"),
            max_exposure=("exposure_at_20", "max"),
            support_catalogue_items=("support_catalogue_items", "max"),
            support_train_interactions=("support_train_interactions", "max"),
        ).reset_index()
        save_table(visibility_dna_group_summary, "visibility_dna_group_summary.csv")
        display(visibility_dna_group_summary.sort_values("mean_pacpg"))

        europe_dna = dna_movies[dna_movies["is_european_wide"]].copy()
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.hist(europe_dna["global_compatibility_score"], bins=np.linspace(0, 1, 11), alpha=0.70, label="Global Compatibility Score", color="#315f9c")
        ax.hist(europe_dna["local_europe_score"], bins=np.linspace(0, 1, 11), alpha=0.62, label="Local Europe Score", color="#d65f3a")
        ax.set_title("European Film Visibility DNA score distribution")
        ax.set_xlabel("Transparent score")
        ax.set_ylabel("European films")
        ax.legend()
        save_current_figure("21_visibility_dna_score_distribution.png", "21. Visibility DNA", "RQ-A/RQ-B", "visibility_dna_movie_scores.csv", "Distribution of transparent global-compatibility and local-Europe scores for European films.")

        dna_plot_groups = ["Platform-compatible Europe", "Local Europe", "Local non-English no-US Europe"]
        exposure_pivot = visibility_dna_group_metrics[visibility_dna_group_metrics["group_name"].isin(dna_plot_groups)].pivot_table(
            index="model_name", columns="group_name", values="exposure_at_20", aggfunc="mean"
        )
        if not exposure_pivot.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            exposure_pivot.plot(kind="bar", ax=ax)
            ax.set_title("Global-compatible Europe vs local Europe exposure")
            ax.set_xlabel("Model")
            ax.set_ylabel("Exposure@20")
            ax.legend(title="")
            save_current_figure("21_visibility_dna_group_exposure.png", "21. Visibility DNA", "RQ-A/RQ-B", "visibility_dna_group_metrics.csv", "Model exposure for platform-compatible and local-Europe DNA groups.")

        pacpg_pivot = visibility_dna_group_metrics[visibility_dna_group_metrics["group_name"].isin(dna_plot_groups)].pivot_table(
            index="model_name", columns="group_name", values="PACPG_at_20", aggfunc="mean"
        )
        if not pacpg_pivot.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            pacpg_pivot.plot(kind="bar", ax=ax)
            ax.axhline(0, color="#0c1b2a", linewidth=1)
            ax.set_title("Visibility DNA PACPG by model")
            ax.set_xlabel("Model")
            ax.set_ylabel("PACPG@20")
            ax.legend(title="")
            save_current_figure("21_visibility_dna_pacpg.png", "21. Visibility DNA", "RQ-A/RQ-B", "visibility_dna_group_metrics.csv", "Preference-adjusted visibility gaps for global-compatible and local-Europe groups.")

        platform_row = visibility_dna_group_summary[visibility_dna_group_summary["group_name"].eq("Platform-compatible Europe")].iloc[0]
        local_row = visibility_dna_group_summary[visibility_dna_group_summary["group_name"].eq("Local Europe")].iloc[0]
        local_no_us_row = visibility_dna_group_summary[visibility_dna_group_summary["group_name"].eq("Local non-English no-US Europe")].iloc[0]
        dna_answer_sentence = (
            f"Platform-compatible Europe has mean Exposure@20={fmt_pct(platform_row['mean_exposure'])} and mean PACPG@20={fmt_pp(platform_row['mean_pacpg'])}; "
            f"Local Europe has mean Exposure@20={fmt_pct(local_row['mean_exposure'])} and mean PACPG@20={fmt_pp(local_row['mean_pacpg'])}; "
            f"the stricter local non-English/no-US subset has mean Exposure@20={fmt_pct(local_no_us_row['mean_exposure'])}."
        )
        result_box(
            "Chapter 21 result: the extension turns Europe into a visibility-DNA question",
            [
                dna_answer_sentence,
                f"Deep-Wikidata cache coverage in this run: {len(dna_extra):,} priority rows; director citizenship coverage {fmt_pct(dna_coverage.loc[dna_coverage['field'].eq('DNA rows with director citizenship'), 'share'].iloc[0])}; filming-location country coverage {fmt_pct(dna_coverage.loc[dna_coverage['field'].eq('DNA rows with filming-location country'), 'share'].iloc[0])}.",
                "TMDb and LUMIERE are documented as next data layers, but not faked: TMDb needs an API key; LUMIERE should be used as a confident-match validation subset.",
                "The new question is stronger than Europe/non-Europe: do rankings preserve local European cultural signals, or mainly surface globally compatible European films?",
            ],
            "This is an extension layer. The score is transparent and useful for interpretation, but it is not a legal definition of cultural identity."
        )
        """
    ),
    md(
        """
        ## 22. Final answers to all questions

        **Question.** What are the direct answers, with confidence and caveats?
        """
    ),
    code(
        """
        def short_answer_table():
            best_utility_row = model_metrics.sort_values("NDCG@20", ascending=False).iloc[0] if len(model_metrics) else pd.Series({"Model": "not available", "NDCG@20": np.nan})
            best_europe_row = model_metrics.sort_values("Europe wide Exposure@20", ascending=False).iloc[0] if len(model_metrics) else pd.Series({"Model": "not available", "Europe wide Exposure@20": np.nan})
            worst_europe_row = model_metrics.sort_values("Europe wide PACPG@20").iloc[0] if len(model_metrics) else pd.Series({"Model": "not available", "Europe wide PACPG@20": np.nan})
            best_non_english_row = model_metrics.sort_values("Non-English Exposure@20", ascending=False).iloc[0] if len(model_metrics) else pd.Series({"Model": "not available", "Non-English Exposure@20": np.nan})
            best_utility_model = best_utility_row["Model"]
            best_europe_model = best_europe_row["Model"]

            strongest_country = country_summary.sort_values("mean_exposure", ascending=False).iloc[0] if "country_summary" in globals() and len(country_summary) else pd.Series({"group_name": "not available", "mean_exposure": np.nan})
            weakest_country = country_summary.sort_values("mean_pacpg").iloc[0] if "country_summary" in globals() and len(country_summary) else pd.Series({"group_name": "not available", "mean_pacpg": np.nan})
            strongest_region = region_summary.sort_values("mean_exposure", ascending=False).iloc[0] if "region_summary" in globals() and len(region_summary) else pd.Series({"group_name": "not available", "mean_exposure": np.nan})
            weakest_region = region_summary.sort_values("mean_pacpg").iloc[0] if "region_summary" in globals() and len(region_summary) else pd.Series({"group_name": "not available", "mean_pacpg": np.nan})
            strongest_language = language_summary.sort_values("mean_exposure", ascending=False).iloc[0] if "language_summary" in globals() and len(language_summary) else pd.Series({"group_name": "not available", "mean_exposure": np.nan})
            weakest_language = language_summary.sort_values("mean_pacpg").iloc[0] if "language_summary" in globals() and len(language_summary) else pd.Series({"group_name": "not available", "mean_pacpg": np.nan})

            spain_best = spain_focus[spain_focus["group_name"].eq("Spain-origin")].sort_values("exposure_at_20", ascending=False).iloc[0] if "spain_focus" in globals() and len(spain_focus) else pd.Series({"model_name": "not available", "exposure_at_20": np.nan})
            multimodal_best = multimodal_audit.sort_values("Europe wide Exposure@20", ascending=False).iloc[0] if "multimodal_audit" in globals() and len(multimodal_audit) else pd.Series({"Model": "not available", "Europe wide Exposure@20": np.nan})
            feedback_answer = "not run"
            if "fb_drift" in globals() and len(fb_drift):
                fb_loss = fb_drift.sort_values("exposure_drift").iloc[0]
                fb_gain = fb_drift.sort_values("exposure_drift", ascending=False).iloc[0]
                feedback_answer = f"largest loss {fb_loss['Model']} / {fb_loss['group_name']} ({fmt_pp(fb_loss['exposure_drift'])}); largest gain {fb_gain['Model']} / {fb_gain['group_name']} ({fmt_pp(fb_gain['exposure_drift'])})"
            rerank_answer = "not run"
            if "best_defensible" in globals() and "base_combined" in globals():
                if "max_visibility" in globals():
                    rerank_answer = f"strict 80% NDCG retention selects lambda={best_defensible['lambda']} ({fmt_pct(best_defensible['Europe wide Exposure@20'])} Europe exposure); maximum visibility occurs at lambda={max_visibility['lambda']} ({fmt_pct(max_visibility['Europe wide Exposure@20'])}, NDCG@20={max_visibility['NDCG@20']:.4f})"
                else:
                    rerank_answer = f"lambda={best_defensible['lambda']} keeps at least 80% NDCG retention and changes Europe exposure from {fmt_pct(base_combined['Europe wide Exposure@20'])} to {fmt_pct(best_defensible['Europe wide Exposure@20'])}"
            visibility_dna_answer = dna_answer_sentence if "dna_answer_sentence" in globals() else "not run; build the optional visibility-DNA cache and rerun section 21"

            answers = [
                ("Does the algorithm hide Europe?", f"Model-dependent. Worst Europe PACPG@20 is {worst_europe_row['Model']} ({fmt_pp(worst_europe_row['Europe wide PACPG@20'])}); highest Europe exposure is {best_europe_model} ({fmt_pct(best_europe_row['Europe wide Exposure@20'])}).", "11, 12", "moderate", "Offline sampled audit with proxy metadata."),
                ("Is catalogue diversity enough?", "No. The visibility funnel shows catalogue share, train interactions, user history, relevant test share and ranked exposure are different layers.", "6, 8, 11", "strong", "Catalogue and metadata coverage still matter."),
                ("Which Europe gets recommended?", f"Visibility concentrates unevenly: strongest region is {strongest_region['group_name']} ({fmt_pct(strongest_region['mean_exposure'])}); weakest region by PACPG is {weakest_region['group_name']} ({fmt_pp(weakest_region['mean_pacpg'])}).", "12", "moderate", "Low-support regions are not overclaimed."),
                ("Does the recommender show local Europe or globally compatible Europe?", visibility_dna_answer, "21", "exploratory/moderate", "DNA scores are transparent proxies; TMDb provider data and LUMIERE admissions are documented next layers, not included as fake data."),
                ("Which European countries are most visible?", f"{strongest_country['group_name']} has the highest mean Exposure@20 among support-passing European countries ({fmt_pct(strongest_country['mean_exposure'])}).", "12", "moderate", "Fractional/binary country counting can differ."),
                ("Which European countries are least visible?", f"{weakest_country['group_name']} has the lowest mean PACPG@20 among support-passing European countries ({fmt_pp(weakest_country['mean_pacpg'])}).", "12", "moderate", "Low support limits claims."),
                ("Which languages are most visible?", f"{strongest_language['group_name']} has the highest mean language Exposure@20 ({fmt_pct(strongest_language['mean_exposure'])}).", "13", "moderate", "Original-language metadata is a proxy."),
                ("Is non-English content underexposed?", f"Non-English visibility is model-dependent; the highest Non-English Exposure@20 model is {best_non_english_row['Model']} ({fmt_pct(best_non_english_row['Non-English Exposure@20'])}), while language PACPG identifies weaker languages.", "11, 13", "moderate", "Multilingual films complicate binary labels."),
                ("Are Spanish films visible?", f"Best Spain-origin exposure model is {spain_best['model_name']} with Exposure@20={fmt_pct(spain_best['exposure_at_20'])}.", "14", "moderate", "Support and metadata coverage limit fine claims."),
                ("Are Spanish-language films the same as Spain-origin films?", f"No. The data show {overlap:,} films with both labels, {only_spain_origin:,} Spain-origin without Spanish-language metadata, and {only_spanish_language:,} Spanish-language without Spain-origin.", "6, 14", "strong", "Depends on Wikidata language metadata."),
                ("Do models differ?", f"Yes. Best utility model is {best_utility_model} (NDCG@20={best_utility_row['NDCG@20']:.4f}); best Europe-exposure model is {best_europe_model}.", "11", "strong", "Bounded candidate/user sample."),
                ("Does the best utility model also have best cultural visibility?", "Not necessarily; the notebook reports utility and exposure separately and the best utility/exposure models can diverge.", "11", "strong", "Metric choice matters."),
                ("Do MPNet/CLIP help?", f"They shift visibility. In the multimodal comparison, highest Europe exposure is {multimodal_best['Model']} ({fmt_pct(multimodal_best['Europe wide Exposure@20'])}).", "16", "moderate", "Embedding availability and candidate sampling matter."),
                ("Is the gap just popularity bias?", f"No single mechanism is enough. The mechanism table highlights long-tail concentration, language, US involvement and co-production; top diagnostic feature is {top_diag_feature['feature']}.", "15", "weak/moderate", "Diagnostic, not causal."),
                ("Does the gap compound in feedback loops?", f"The Schedl-inspired stress test reports dynamic drift: {feedback_answer}.", "18", "moderate", "Lightweight offline stress test; no per-iteration model retraining."),
                ("Can re-ranking mitigate it?", f"Yes, as a trade-off frontier: {rerank_answer}.", "19", "moderate", "Does not solve fairness; can overcorrect."),
                ("What should platforms report?", "Catalogue share, ranked exposure, country/language exposure, missingness, feedback-loop drift, visibility-DNA slices and mitigation frontiers.", "23", "strong", "Requires reliable metadata and audit access."),
                ("What can we not conclude?", "No legal compliance, discrimination, causal manipulation, or user identity claim.", "24", "strong", "Offline proxy-label audit."),
            ]
            return pd.DataFrame(answers, columns=["Question", "Short answer", "Evidence in notebook section", "Confidence", "Main caveat"])

        final_answer_table = short_answer_table()
        save_table(final_answer_table, "final_answer_table.csv")
        display(final_answer_table)
        """
    ),
    md(
        """
        ## 23. Governance audit card

        **Question.** What should a platform report from a governance perspective?
        """
    ),
    code(
        """
        audit_card = pd.DataFrame([
            {"field": "dataset scope", "value": f"{len(movie_db):,} MovieLens movies; {len(m3l_interactions):,} M3L interactions"},
            {"field": "sample restrictions", "value": f"FULL_RUN={FULL_RUN}; evaluated users={len(eval_users):,}; candidate items={len(candidate_items):,}"},
            {"field": "raw data sources", "value": "MovieLens metadata/ratings/tags/genome; M3L interactions/features; cached Wikidata metadata"},
            {"field": "join coverage", "value": f"country={movie_db['has_country'].mean():.1%}; language={movie_db['has_language'].mean():.1%}; MPNet matrix={movie_db['has_mpnet_matrix'].mean():.1%}; CLIP matrix={movie_db['has_clip_matrix'].mean():.1%}"},
            {"field": "proxy definitions", "value": "country of origin, original language, production company country, wider Europe/EU27 sensitivity"},
            {"field": "models tested", "value": ', '.join(model_metrics['Model']) if len(model_metrics) else "none"},
            {"field": "utility metrics", "value": "NDCG@5/10/20/50, Recall@5/10/20/50, MAP@20, Coverage@20"},
            {"field": "exposure metrics", "value": "discounted exposure, catalogue/interaction/relevance gaps, PACPG"},
            {"field": "dynamic drift metrics", "value": "Schedl-inspired lightweight stress test: exposure/profile shares over iterations; no per-iteration retraining in bounded run"},
            {"field": "visibility DNA extension", "value": "transparent Global Compatibility and Local Europe scores; optional deeper Wikidata cache; TMDb/LUMIERE documented as next layers"},
            {"field": "mitigation tested", "value": "transparent lambda-based post-processing re-ranking"},
            {"field": "strongest findings", "value": "catalogue, interaction and ranked exposure differ; model choice changes visibility"},
            {"field": "weakest findings", "value": "low-support countries, granular Spain subgroups and religion/theme themes require caution"},
            {"field": "platform reporting recommendation", "value": "report catalogue share, ranked exposure, country/language distribution, missingness, feedback-loop drift and mitigation trade-off frontier"},
        ])
        save_table(audit_card, "governance_audit_card.csv")
        display(audit_card)
        """
    ),
    md(
        """
        ## 24. Limitations

        This is an offline recommender audit. It does not observe a real streaming interface, a real platform algorithm or actual user choices after recommendation.

        Main limitations:

        - MovieLens users are not representative of all streaming users.
        - Country and language labels are proxies from Wikidata and may be missing or ambiguous.
        - Country of origin does not fully capture cultural identity, financing, distribution or audience context.
        - Original-language metadata is not the same as subtitle/dubbing availability or actual viewer language.
        - Co-productions and US-company involvement complicate simple Europe/non-Europe labels.
        - Missing country/language metadata may be concentrated in long-tail items.
        - Feedback loops are offline simulations, not causal evidence of user manipulation.
        - Re-ranking is transparent and auditable, but it does not solve fairness.
        - Religion/theme analysis is exploratory and not used as a main fairness metric because proxy risk is high.
        - The European Film Visibility DNA extension is a transparent proxy layer, not a legal or complete cultural identity model.
        - TMDb watch-provider and LUMIERE admissions data are documented as next data layers; they are not scraped, guessed or used without proper access/matching.
        """
    ),
    md(
        """
        ## 25. Conclusion

        The relevant governance object is not only whether European films exist in a catalogue, but whether countries and languages receive ranked and repeated visibility. This notebook shows how to audit that transformation: start with data coverage, preserve unknowns, define proxy labels transparently, compare catalogue and interaction baselines, evaluate recommendation utility and exposure together, inspect intra-European and language visibility, simulate feedback-loop drift, test transparent re-ranking and extend the interpretation through a European Film Visibility DNA layer.

        The practical governance recommendation is precise: platforms should report catalogue share, ranked exposure share, country/language distributions, missing metadata share, local-vs-global compatibility slices, feedback-loop drift and mitigation trade-off frontiers. Catalogue diversity alone is not enough.
        """
    ),
    code(
        """
        # Final output ledgers and sanity checks.
        figure_ledger_df = pd.DataFrame(figure_ledger)
        save_table(figure_ledger_df, "figure_ledger.csv")

        sanity_df = pd.DataFrame(sanity_checks)
        save_table(sanity_df, "sanity_checks.csv")
        filter_threshold_df = pd.DataFrame(filter_threshold_ledger)
        save_table(filter_threshold_df, "filter_and_threshold_ledger.csv")

        run_manifest = {
            "config": config,
            "runtime_info": runtime_info.to_dict("records"),
            "tables_dir": str(TABLE_DIR.relative_to(PROJECT_ROOT)),
            "figures_dir": str(FIG_DIR.relative_to(PROJECT_ROOT)),
            "model_run_ledger": model_run_ledger,
            "filter_threshold_ledger": filter_threshold_ledger,
            "sanity_checks": sanity_checks,
        }
        save_json(run_manifest, "notebook_run_manifest.json")

        display(Markdown("**Notebook completed.** All notebook-derived tables, figures, figure ledger, sanity checks and run manifest were saved under `cultural_prominence_audit/outputs/final_notebook_*`."))
        display(sanity_df)
        display(figure_ledger_df.tail(10))
        """
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    nb["cells"] = cells
    nb["metadata"] = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "pygments_lexer": "ipython3"},
    }
    target = NOTEBOOKS / "does_algorithm_hide_europe_final_research_story.ipynb"
    nbf.write(nb, target)
    print(f"Wrote {target}")


if __name__ == "__main__":
    main()
