from __future__ import annotations

from pathlib import Path
import json
import math
import time
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
from tqdm.auto import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
for path in [DATA_INTERIM, DATA_PROCESSED, OUTPUTS]:
    path.mkdir(parents=True, exist_ok=True)

SEED = 42
TOP_K = 20
CANDIDATE_K = 100
FULL_MAX_USERS = 5_000
TOP_POPULAR_ITEMS = 5_000

M3L_INTERACTIONS = PROJECT_ROOT / "m3l-20m" / "m3l-20m.inter"
MOVIELENS_MOVIES = PROJECT_ROOT / "MovieLens 20M Dataset" / "movie.csv"
MOVIELENS_LINKS = PROJECT_ROOT / "MovieLens 20M Dataset" / "link.csv"
MPNET_MATRIX = PROJECT_ROOT / "m3l-20m" / "text" / "mpnet.npy"
CLIP_MATRIX = PROJECT_ROOT / "m3l-20m" / "image" / "clip_image.npy"
TEXT_JSON_DIR = PROJECT_ROOT / "TEXT_mpnet"

COLORS = {
    "ink": "#243447",
    "blue": "#2F6B9A",
    "green": "#2E8B57",
    "gold": "#C99700",
    "red": "#B84A4A",
    "teal": "#1F8A8A",
    "violet": "#6E5AA8",
    "gray": "#7A869A",
}

EUROPE_COUNTRIES = {
    "Albania", "Austria", "Belgium", "Bosnia and Herzegovina", "Bulgaria",
    "Croatia", "Cyprus", "Czech Republic", "Czechia", "Denmark", "Estonia",
    "Finland", "France", "Germany", "Greece", "Hungary", "Iceland", "Ireland",
    "Italy", "Latvia", "Lithuania", "Luxembourg", "Malta", "Moldova",
    "Montenegro", "Netherlands", "North Macedonia", "Norway", "Poland",
    "Portugal", "Romania", "Serbia", "Slovakia", "Slovenia", "Spain",
    "Sweden", "Switzerland", "Ukraine", "United Kingdom",
}
ENGLISH_LABELS = {"English"}
US_LABELS = {"United States", "United States of America"}


def savefig(name: str):
    path = OUTPUTS / name
    plt.savefig(path, dpi=220, bbox_inches="tight")
    print(f"Saved {path}")


def sorted_unique(values):
    clean = {str(x) for x in values if pd.notna(x) and str(x).strip() and str(x).lower() != "nan"}
    return sorted(clean)


def split_pipe(value):
    if pd.isna(value) or value == "":
        return []
    return sorted({part.strip() for part in str(value).split("|") if part.strip()})


def load_core_data():
    print("Loading M3L interactions...")
    interactions = pd.read_csv(
        M3L_INTERACTIONS,
        sep="\t",
        dtype={"userID": "int32", "itemID": "int32", "rating": "float32", "x_label": "int8"},
    ).rename(columns={"userID": "user_id", "itemID": "item_id", "rating": "rating_or_weight", "x_label": "split"})
    interactions["split"] = interactions["split"].map({0: "train", 1: "valid", 2: "test"})

    movies = pd.read_csv(MOVIELENS_MOVIES)
    links = pd.read_csv(MOVIELENS_LINKS)
    links["imdb_id_str"] = links["imdbId"].apply(lambda x: f"tt{int(x):07d}" if pd.notna(x) else None)
    item_id_map = pd.read_csv(DATA_INTERIM / "m3l_internal_to_movielens.csv")
    return interactions, movies, links, item_id_map


def query_wikidata_extended(imdb_ids, batch_size=200, sleep=0.08):
    endpoint = "https://query.wikidata.org/sparql"
    rows = []
    imdb_ids = [x for x in imdb_ids if isinstance(x, str) and x.startswith("tt")]

    for i in tqdm(range(0, len(imdb_ids), batch_size), desc="Querying extended Wikidata"):
        batch = imdb_ids[i:i + batch_size]
        values = " ".join(f'"{x}"' for x in batch)
        query = f"""
        SELECT ?imdb ?film ?filmLabel
          (GROUP_CONCAT(DISTINCT ?countryLabel; separator="|") AS ?countries)
          (GROUP_CONCAT(DISTINCT ?originalLanguageLabel; separator="|") AS ?originalLanguages)
          (GROUP_CONCAT(DISTINCT ?workLanguageLabel; separator="|") AS ?workLanguages)
          (SAMPLE(?publicationDate) AS ?publicationDate)
          (GROUP_CONCAT(DISTINCT ?productionCompanyLabel; separator="|") AS ?productionCompanies)
          (GROUP_CONCAT(DISTINCT ?companyCountryLabel; separator="|") AS ?productionCompanyCountries)
          (GROUP_CONCAT(DISTINCT ?companyHqCountryLabel; separator="|") AS ?productionCompanyHqCountries)
        WHERE {{
          VALUES ?imdb {{ {values} }}
          ?film wdt:P345 ?imdb .
          OPTIONAL {{ ?film wdt:P495 ?country . }}
          OPTIONAL {{ ?film wdt:P364 ?originalLanguage . }}
          OPTIONAL {{ ?film wdt:P407 ?workLanguage . }}
          OPTIONAL {{ ?film wdt:P577 ?publicationDate . }}
          OPTIONAL {{
            ?film wdt:P272 ?productionCompany .
            OPTIONAL {{ ?productionCompany wdt:P17 ?companyCountry . }}
            OPTIONAL {{ ?productionCompany wdt:P159/wdt:P17 ?companyHqCountry . }}
          }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en".
            ?film rdfs:label ?filmLabel .
            ?country rdfs:label ?countryLabel .
            ?originalLanguage rdfs:label ?originalLanguageLabel .
            ?workLanguage rdfs:label ?workLanguageLabel .
            ?productionCompany rdfs:label ?productionCompanyLabel .
            ?companyCountry rdfs:label ?companyCountryLabel .
            ?companyHqCountry rdfs:label ?companyHqCountryLabel .
          }}
        }}
        GROUP BY ?imdb ?film ?filmLabel
        """
        try:
            response = requests.get(
                endpoint,
                params={"query": query, "format": "json"},
                headers={"User-Agent": "WU-Data-Algorithmic-Governance-Student-Project/1.0"},
                timeout=90,
            )
            response.raise_for_status()
            bindings = response.json()["results"]["bindings"]
        except Exception as exc:
            print(f"Wikidata batch failed at {i}: {exc}")
            time.sleep(3)
            continue

        for row in bindings:
            rows.append({
                "imdb_id_str": row.get("imdb", {}).get("value"),
                "wikidata_uri": row.get("film", {}).get("value"),
                "title_wikidata": row.get("filmLabel", {}).get("value"),
                "country": row.get("countries", {}).get("value"),
                "original_language": row.get("originalLanguages", {}).get("value"),
                "language_of_work": row.get("workLanguages", {}).get("value"),
                "publication_date": row.get("publicationDate", {}).get("value"),
                "production_company": row.get("productionCompanies", {}).get("value"),
                "production_company_country": row.get("productionCompanyCountries", {}).get("value"),
                "production_company_hq_country": row.get("productionCompanyHqCountries", {}).get("value"),
            })
        time.sleep(sleep)
    return pd.DataFrame(rows)


def load_or_query_extended_wikidata(movie_meta):
    cache_path = DATA_INTERIM / "wikidata_movie_metadata_extended.csv"
    imdb_ids = movie_meta["imdb_id_str"].dropna().drop_duplicates().tolist()
    if cache_path.exists():
        wikidata = pd.read_csv(cache_path)
    else:
        wikidata = pd.DataFrame(columns=[
            "imdb_id_str", "wikidata_uri", "title_wikidata", "country",
            "original_language", "language_of_work", "publication_date",
            "production_company", "production_company_country", "production_company_hq_country",
        ])

    cached = set(wikidata["imdb_id_str"].dropna()) if len(wikidata) else set()
    missing = [x for x in imdb_ids if x not in cached]
    print(f"Extended Wikidata cache rows before query: {len(wikidata):,}")
    print(f"Movie IMDb IDs: {len(imdb_ids):,}; missing from extended cache: {len(missing):,}")
    if missing:
        new_rows = query_wikidata_extended(missing)
        if len(new_rows):
            wikidata = pd.concat([wikidata, new_rows], ignore_index=True).drop_duplicates("imdb_id_str")
            wikidata.to_csv(cache_path, index=False)
    elif not cache_path.exists():
        wikidata.to_csv(cache_path, index=False)
    print(f"Extended Wikidata cache rows after query: {len(wikidata):,}")
    return wikidata


def build_metadata(interactions, movies, links, item_id_map):
    train = interactions[interactions["split"].eq("train")][["user_id", "item_id", "rating_or_weight"]].copy()
    item_counts = train.groupby("item_id").size().rename("train_interaction_count").reset_index()
    movie_meta = (
        item_id_map
        .merge(movies, on="movieId", how="left")
        .merge(links, on="movieId", how="left")
        .merge(item_counts, on="item_id", how="left")
    )
    movie_meta["train_interaction_count"] = movie_meta["train_interaction_count"].fillna(0)
    wikidata = load_or_query_extended_wikidata(movie_meta)
    wd = wikidata.copy()
    for col in ["country", "original_language", "language_of_work", "production_company", "production_company_country", "production_company_hq_country"]:
        wd[col] = wd[col].apply(split_pipe)

    movie_meta = movie_meta.merge(wd, on="imdb_id_str", how="left")
    list_cols = ["country", "original_language", "language_of_work", "production_company", "production_company_country", "production_company_hq_country"]
    for col in list_cols:
        movie_meta[col] = movie_meta[col].apply(lambda x: x if isinstance(x, list) else [])

    movie_meta["has_wikidata_match"] = movie_meta["wikidata_uri"].notna()
    movie_meta["has_country"] = movie_meta["country"].apply(bool)
    movie_meta["has_language"] = movie_meta["original_language"].apply(bool)
    movie_meta["has_language_of_work"] = movie_meta["language_of_work"].apply(bool)
    movie_meta["has_production_company"] = movie_meta["production_company"].apply(bool)
    movie_meta["has_production_company_country"] = movie_meta.apply(
        lambda row: bool(row["production_company_country"]) or bool(row["production_company_hq_country"]),
        axis=1,
    )

    movie_meta["origin_country_count"] = movie_meta["country"].apply(len)
    movie_meta["original_language_count"] = movie_meta["original_language"].apply(len)
    movie_meta["is_coproduction_by_country"] = movie_meta["origin_country_count"] > 1
    movie_meta["is_multilingual_original"] = movie_meta["original_language_count"] > 1
    movie_meta["is_european"] = movie_meta["country"].apply(lambda xs: any(c in EUROPE_COUNTRIES for c in xs))
    movie_meta["has_us_origin_country"] = movie_meta["country"].apply(lambda xs: any(c in US_LABELS for c in xs))
    movie_meta["is_non_english"] = movie_meta["original_language"].apply(lambda xs: bool(xs) and not any(lang in ENGLISH_LABELS for lang in xs))
    movie_meta["has_english_original_language"] = movie_meta["original_language"].apply(lambda xs: any(lang in ENGLISH_LABELS for lang in xs))
    movie_meta["production_company_country_all"] = movie_meta.apply(
        lambda row: sorted(set(row["production_company_country"]) | set(row["production_company_hq_country"])),
        axis=1,
    )
    movie_meta["has_us_company_country"] = movie_meta["production_company_country_all"].apply(lambda xs: any(c in US_LABELS for c in xs))
    movie_meta["is_european_with_us_company"] = movie_meta["is_european"] & movie_meta["has_us_company_country"]
    movie_meta["is_non_us_origin_with_us_company"] = (~movie_meta["has_us_origin_country"]) & movie_meta["has_us_company_country"]

    q80 = movie_meta["train_interaction_count"].quantile(0.80)
    q20 = movie_meta["train_interaction_count"].quantile(0.20)
    movie_meta["is_blockbuster_head"] = movie_meta["train_interaction_count"] >= q80
    movie_meta["is_long_tail"] = movie_meta["train_interaction_count"] <= q20

    movie_meta.to_csv(DATA_PROCESSED / "movie_audit_metadata_extended.csv", index=False)
    return movie_meta, train, q20, q80


def plot_extra_caveats(movie_meta):
    rows = [
        {"check": "Has original language", "share": movie_meta["has_language"].mean()},
        {"check": "Has language of work/name", "share": movie_meta["has_language_of_work"].mean()},
        {"check": "Multiple original languages", "share": movie_meta["is_multilingual_original"].mean()},
        {"check": "Multiple origin countries", "share": movie_meta["is_coproduction_by_country"].mean()},
        {"check": "Has production company", "share": movie_meta["has_production_company"].mean()},
        {"check": "Has company country", "share": movie_meta["has_production_company_country"].mean()},
        {"check": "European origin + US company", "share": movie_meta["is_european_with_us_company"].mean()},
        {"check": "Non-US origin + US company", "share": movie_meta["is_non_us_origin_with_us_company"].mean()},
    ]
    proxy_risk = pd.DataFrame(rows)
    proxy_risk.to_csv(OUTPUTS / "11_metadata_proxy_risk_table.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.barh(proxy_risk["check"][::-1], proxy_risk["share"][::-1], color=COLORS["teal"])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of M3L items")
    ax.set_title("Metadata proxy-risk checks: language, country and production-company caveats", weight="bold")
    for i, value in enumerate(proxy_risk["share"][::-1]):
        ax.text(value + 0.015, i, f"{value:.1%}", va="center", fontsize=9)
    ax.text(
        0,
        -0.16,
        "Interpretation: country of origin and original language are useful proxies, but co-productions and US-company involvement require caveats.",
        transform=ax.transAxes,
        fontsize=9,
        color=COLORS["gray"],
    )
    fig.tight_layout()
    savefig("11_language_country_company_caveats.png")
    plt.close(fig)
    return proxy_risk


def rank_discount(rank):
    return 1 / math.log2(rank + 1)


def recall_at_k(items, relevant, k=TOP_K):
    if not relevant:
        return np.nan
    return len(set(items[:k]) & relevant) / len(relevant)


def ndcg_at_k(items, relevant, k=TOP_K):
    if not relevant:
        return np.nan
    dcg = sum(rank_discount(rank) for rank, item in enumerate(items[:k], start=1) if item in relevant)
    ideal = sum(rank_discount(rank) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else np.nan


def map_at_k(items, relevant, k=TOP_K):
    if not relevant:
        return np.nan
    hits = 0
    precisions = []
    for rank, item in enumerate(items[:k], start=1):
        if item in relevant:
            hits += 1
            precisions.append(hits / rank)
    return float(np.sum(precisions) / min(len(relevant), k)) if precisions else 0.0


def run_model_audit(interactions, train, movie_meta):
    rng = np.random.default_rng(SEED)
    test = interactions[interactions["split"].eq("test")][["user_id", "item_id", "rating_or_weight"]].copy()
    item_counts = train.groupby("item_id").size().rename("train_interaction_count").reset_index()

    train_user_counts = train.groupby("user_id").size()
    test_user_counts = test.groupby("user_id").size()
    eligible_users = sorted(set(train_user_counts[train_user_counts >= 5].index) & set(test_user_counts[test_user_counts >= 1].index))
    sample_users = np.array(eligible_users, dtype=np.int32)
    rng.shuffle(sample_users)
    sample_users = np.sort(sample_users[:min(FULL_MAX_USERS, len(sample_users))])

    sample_train = train[np.isin(train["user_id"].to_numpy(), sample_users)].copy()
    sample_test = test[np.isin(test["user_id"].to_numpy(), sample_users)].copy()

    top_items = item_counts.sort_values("train_interaction_count", ascending=False)["item_id"].head(TOP_POPULAR_ITEMS).astype(int).tolist()
    candidate_items = sorted(set(top_items) | set(sample_train["item_id"].astype(int)) | set(sample_test["item_id"].astype(int)))
    candidate_items = np.array(candidate_items, dtype=np.int32)

    user_to_idx = {int(u): i for i, u in enumerate(sample_users)}
    item_to_idx = {int(it): j for j, it in enumerate(candidate_items)}
    sample_train = sample_train[sample_train["item_id"].isin(item_to_idx)]
    sample_test = sample_test[sample_test["item_id"].isin(item_to_idx)]

    rows = sample_train["user_id"].map(user_to_idx).to_numpy()
    cols = sample_train["item_id"].map(item_to_idx).to_numpy()
    data = (sample_train["rating_or_weight"].to_numpy() > 0).astype(np.float32)
    r_train = sparse.csr_matrix((data, (rows, cols)), shape=(len(sample_users), len(candidate_items)))

    seen_by_user = {
        int(uid): set(group["item_id"].map(item_to_idx).dropna().astype(int))
        for uid, group in sample_train.groupby("user_id")
    }
    test_relevant_by_user = {}
    for uid, group in sample_test[sample_test["rating_or_weight"] > 0].groupby("user_id"):
        items = set(group["item_id"].astype(int))
        if items:
            test_relevant_by_user[int(uid)] = items

    candidate_meta = (
        movie_meta[["item_id", "is_european", "is_non_english", "is_long_tail", "is_blockbuster_head"]]
        .set_index("item_id")
        .reindex(candidate_items)
        .fillna(False)
    )
    group_columns = {"European": "is_european", "Non-English": "is_non_english", "Long-tail": "is_long_tail"}
    group_flags = {
        group: candidate_meta[col].astype(bool).to_dict()
        for group, col in group_columns.items()
    }
    train_groups = sample_train.merge(movie_meta[["item_id", *group_columns.values()]], on="item_id", how="left")
    test_groups = sample_test[sample_test["rating_or_weight"] > 0].merge(movie_meta[["item_id", *group_columns.values()]], on="item_id", how="left")
    history_share = {group: train_groups.groupby("user_id")[col].mean().fillna(0).to_dict() for group, col in group_columns.items()}
    relevant_share = {group: test_groups.groupby("user_id")[col].mean().fillna(0).to_dict() for group, col in group_columns.items()}
    user_targets = {
        int(uid): {
            group: max(float(history_share[group].get(int(uid), 0.0)), float(relevant_share[group].get(int(uid), 0.0)))
            for group in group_columns
        }
        for uid in sample_users
    }

    def mask_seen(scores):
        scores = np.asarray(scores, dtype=np.float32).copy()
        for uid in sample_users:
            ui = user_to_idx[int(uid)]
            seen = seen_by_user.get(int(uid), set())
            if seen:
                scores[ui, list(seen)] = -np.inf
        return scores

    def recommend_from_scores(scores, k=TOP_K):
        masked = mask_seen(scores)
        recs = {}
        for uid in sample_users:
            ui = user_to_idx[int(uid)]
            s = masked[ui]
            finite = np.isfinite(s)
            if finite.sum() == 0:
                recs[int(uid)] = []
                continue
            k_eff = min(k, int(finite.sum()))
            idx = np.argpartition(-s, k_eff - 1)[:k_eff]
            idx = idx[np.argsort(-s[idx])]
            recs[int(uid)] = [int(candidate_items[j]) for j in idx]
        return recs

    def rowwise_standardise(scores):
        arr = np.asarray(scores, dtype=np.float32)
        out = np.zeros_like(arr, dtype=np.float32)
        finite = np.isfinite(arr)
        for i in range(arr.shape[0]):
            mask = finite[i]
            if not mask.any():
                continue
            vals = arr[i, mask]
            std = vals.std()
            out[i, mask] = (vals - vals.mean()) / std if std > 1e-8 else 0.0
            out[i, ~mask] = -np.inf
        return out

    def exposure(items, group_name, k=TOP_K):
        numerator = 0.0
        denominator = 0.0
        flags = group_flags[group_name]
        for rank, item in enumerate(items[:k], start=1):
            w = rank_discount(rank)
            denominator += w
            numerator += w * float(flags.get(item, False))
        return numerator / denominator if denominator else np.nan

    def evaluate(recs_dict, k=TOP_K):
        rows = []
        for model_name, recs in recs_dict.items():
            recalls, ndcgs, maps = [], [], []
            exposure_values = {group: [] for group in group_columns}
            prominence_gaps = {group: [] for group in group_columns}
            pacpg_values = {group: [] for group in group_columns}
            covered = set()
            for uid, items in recs.items():
                relevant = test_relevant_by_user.get(uid, set())
                if not relevant:
                    continue
                covered.update(items[:k])
                recalls.append(recall_at_k(items, relevant, k))
                ndcgs.append(ndcg_at_k(items, relevant, k))
                maps.append(map_at_k(items, relevant, k))
                for group in group_columns:
                    exp = exposure(items, group, k)
                    rel_share = float(relevant_share[group].get(uid, 0.0))
                    target = float(user_targets.get(uid, {}).get(group, 0.0))
                    exposure_values[group].append(exp)
                    prominence_gaps[group].append(exp - rel_share)
                    pacpg_values[group].append(exp - target)
            row = {
                "Model": model_name,
                f"NDCG@{k}": np.nanmean(ndcgs),
                f"Recall@{k}": np.nanmean(recalls),
                f"MAP@{k}": np.nanmean(maps),
                f"Coverage@{k}": len(covered) / len(candidate_items),
            }
            for group in group_columns:
                row[f"{group} Exposure@{k}"] = np.nanmean(exposure_values[group])
                row[f"ProminenceGap {group}"] = np.nanmean(prominence_gaps[group])
                row[f"PACPG {group}"] = np.nanmean(pacpg_values[group])
            rows.append(row)
        return pd.DataFrame(rows)

    print(f"Full audit sample: {len(sample_users):,} users, {len(candidate_items):,} candidate items.")

    candidate_pop = item_counts.set_index("item_id").reindex(candidate_items)["train_interaction_count"].fillna(0).to_numpy(np.float32)
    popularity_scores = np.tile(np.log1p(candidate_pop), (len(sample_users), 1)).astype(np.float32)
    popularity_recs = recommend_from_scores(popularity_scores)

    item_user = r_train.T.astype(np.float32)
    item_norms = np.sqrt(item_user.multiply(item_user).sum(axis=1)).A1
    item_norms[item_norms == 0] = 1.0
    item_user_norm = item_user.multiply(1 / item_norms[:, None])
    item_similarity = (item_user_norm @ item_user_norm.T).toarray().astype(np.float32)
    np.fill_diagonal(item_similarity, 0.0)
    keep = min(50, item_similarity.shape[1])
    if keep < item_similarity.shape[1]:
        threshold_idx = np.argpartition(-item_similarity, keep, axis=1)[:, keep:]
        row_idx = np.arange(item_similarity.shape[0])[:, None]
        item_similarity[row_idx, threshold_idx] = 0.0
    itemknn_scores = (r_train @ item_similarity).astype(np.float32)
    itemknn_recs = recommend_from_scores(itemknn_scores)
    del item_similarity, itemknn_scores

    n_components = min(96, max(2, min(r_train.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=SEED)
    user_factors = svd.fit_transform(r_train).astype(np.float32)
    svd_scores = (user_factors @ svd.components_.astype(np.float32)).astype(np.float32)
    svd_recs = recommend_from_scores(svd_scores)

    def content_scores(feature_path):
        features = np.load(feature_path, mmap_mode="r")[candidate_items].astype(np.float32)
        features = normalize(features, norm="l2", axis=1)
        profile_sum = r_train @ features
        counts_per_user = np.maximum(r_train.sum(axis=1).A1, 1.0).astype(np.float32)
        profiles = profile_sum / counts_per_user[:, None]
        profiles = normalize(profiles, norm="l2", axis=1)
        return (profiles @ features.T).astype(np.float32)

    mpnet_scores = content_scores(MPNET_MATRIX)
    clip_scores = content_scores(CLIP_MATRIX)
    mpnet_recs = recommend_from_scores(mpnet_scores)
    clip_recs = recommend_from_scores(clip_scores)

    hybrid_scores = (
        0.50 * rowwise_standardise(svd_scores)
        + 0.25 * rowwise_standardise(mpnet_scores)
        + 0.25 * rowwise_standardise(clip_scores)
    ).astype(np.float32)
    hybrid_recs = recommend_from_scores(hybrid_scores)
    del mpnet_scores, clip_scores, svd_scores

    def rerank_hybrid(lambda_value=0.3, k=TOP_K, candidate_k=CANDIDATE_K):
        masked = mask_seen(hybrid_scores)
        recs = {}
        for uid in sample_users:
            uid = int(uid)
            ui = user_to_idx[uid]
            s = masked[ui]
            finite = np.isfinite(s)
            if finite.sum() == 0:
                recs[uid] = []
                continue
            pool_size = min(candidate_k, int(finite.sum()))
            pool_idx = np.argpartition(-s, pool_size - 1)[:pool_size]
            pool_idx = pool_idx[np.argsort(-s[pool_idx])]
            pool_items = [int(candidate_items[j]) for j in pool_idx]
            rel_scores = s[pool_idx].astype(np.float32)
            rel_std = rel_scores.std()
            rel_scores = (rel_scores - rel_scores.mean()) / rel_std if rel_std > 1e-8 else rel_scores * 0
            rel_lookup = dict(zip(pool_items, rel_scores))
            selected = []
            target_by_group = user_targets.get(uid, {group: 0.0 for group in group_columns})
            for _ in range(k):
                best_item = None
                best_score = -np.inf
                current_counts = {group: sum(bool(group_flags[group].get(x, False)) for x in selected) for group in group_columns}
                denom = len(selected) + 1
                for item in pool_items:
                    if item in selected:
                        continue
                    bonus = 0.0
                    for group in group_columns:
                        target = float(target_by_group.get(group, 0.0))
                        flag = bool(group_flags[group].get(item, False))
                        current_share = (current_counts[group] + int(flag)) / denom
                        if current_share < target and flag:
                            bonus += target - current_share
                    final_score = float(rel_lookup[item]) + lambda_value * bonus
                    if final_score > best_score:
                        best_score = final_score
                        best_item = item
                if best_item is None:
                    break
                selected.append(best_item)
            recs[uid] = selected
        return recs

    base_recs = {
        "Popularity": popularity_recs,
        "ItemKNN": itemknn_recs,
        "SVD": svd_recs,
        "MPNet-content": mpnet_recs,
        "CLIP-image-content": clip_recs,
        "Hybrid": hybrid_recs,
    }
    lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
    rerank_recs = {f"Hybrid + reranking lambda={lam:.1f}": rerank_hybrid(lam) for lam in lambdas}
    all_recs = dict(base_recs)
    all_recs["Hybrid + reranking lambda=0.3"] = rerank_recs["Hybrid + reranking lambda=0.3"]
    all_recs["Hybrid + reranking lambda=0.7"] = rerank_recs["Hybrid + reranking lambda=0.7"]

    model_comparison = evaluate(all_recs)
    order = ["Popularity", "ItemKNN", "SVD", "MPNet-content", "CLIP-image-content", "Hybrid", "Hybrid + reranking lambda=0.3", "Hybrid + reranking lambda=0.7"]
    model_comparison["Model"] = pd.Categorical(model_comparison["Model"], categories=order, ordered=True)
    model_comparison = model_comparison.sort_values("Model").reset_index(drop=True)
    model_comparison.to_csv(OUTPUTS / "model_comparison.csv", index=False)
    model_comparison.to_csv(OUTPUTS / "12_full_model_comparison.csv", index=False)

    rerank_tradeoff = evaluate(rerank_recs)
    rerank_tradeoff.to_csv(OUTPUTS / "12_full_rerank_tradeoff.csv", index=False)
    run_report = pd.DataFrame([
        {"object": "full analysis users", "count": len(sample_users)},
        {"object": "candidate items", "count": len(candidate_items)},
        {"object": "sample train rows", "count": len(sample_train)},
        {"object": "sample test rows", "count": len(sample_test)},
        {"object": "users with relevant test items", "count": len(test_relevant_by_user)},
    ])
    run_report.to_csv(OUTPUTS / "12_full_run_report.csv", index=False)

    plot_model_outputs(model_comparison, rerank_tradeoff)
    return model_comparison, rerank_tradeoff, run_report


def plot_model_outputs(model_comparison, rerank_tradeoff):
    exposure_cols = [f"European Exposure@{TOP_K}", f"Non-English Exposure@{TOP_K}", f"Long-tail Exposure@{TOP_K}"]
    exposure_long = model_comparison.melt(id_vars="Model", value_vars=exposure_cols, var_name="group", value_name="exposure")
    exposure_long["group"] = exposure_long["group"].str.replace(f" Exposure@{TOP_K}", "", regex=False)
    fig, ax = plt.subplots(figsize=(11, 5.4))
    groups = exposure_long["group"].unique()
    x = np.arange(len(model_comparison))
    width = 0.24
    palette = [COLORS["green"], COLORS["gold"], COLORS["blue"]]
    for i, group in enumerate(groups):
        values = exposure_long[exposure_long["group"].eq(group)]["exposure"].to_numpy()
        ax.bar(x + (i - 1) * width, values, width, label=group, color=palette[i])
    ax.set_xticks(x, model_comparison["Model"], rotation=30, ha="right")
    ax.set_ylabel(f"Discounted group exposure@{TOP_K}")
    ax.set_title("Full-run group exposure by recommender model", weight="bold")
    ax.legend()
    fig.tight_layout()
    savefig("08_group_exposure_by_model.png")
    savefig("12_full_group_exposure_by_model.png")
    plt.close(fig)

    summary = model_comparison.copy()
    hybrid_row = summary[summary["Model"].astype(str).eq("Hybrid")].iloc[0]
    summary["NDCG retention vs Hybrid"] = summary[f"NDCG@{TOP_K}"] / hybrid_row[f"NDCG@{TOP_K}"] if hybrid_row[f"NDCG@{TOP_K}"] else np.nan
    summary["Absolute PACPG"] = summary[["PACPG European", "PACPG Non-English", "PACPG Long-tail"]].abs().sum(axis=1)
    summary["Prominence improvement vs Hybrid"] = hybrid_row[["PACPG European", "PACPG Non-English", "PACPG Long-tail"]].abs().sum() - summary["Absolute PACPG"]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.scatter(summary["NDCG retention vs Hybrid"], summary["Prominence improvement vs Hybrid"], s=80, color=COLORS["violet"])
    for _, row in summary.iterrows():
        ax.annotate(str(row["Model"]), (row["NDCG retention vs Hybrid"], row["Prominence improvement vs Hybrid"]), textcoords="offset points", xytext=(6, 4), fontsize=8)
    ax.axhline(0, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.axvline(1, color=COLORS["gray"], linestyle="--", linewidth=1)
    ax.set_xlabel("NDCG retention relative to Hybrid")
    ax.set_ylabel("Reduction in absolute PACPG relative to Hybrid")
    ax.set_title("Full-run accuracy and cultural prominence summary", weight="bold")
    fig.tight_layout()
    savefig("09_accuracy_fairness_summary.png")
    savefig("13_full_accuracy_fairness_summary.png")
    plt.close(fig)

    base_hybrid = rerank_tradeoff[rerank_tradeoff["Model"].eq("Hybrid + reranking lambda=0.0")].iloc[0]
    tradeoff = rerank_tradeoff.copy()
    tradeoff["lambda"] = tradeoff["Model"].str.extract(r"lambda=([0-9.]+)").astype(float)
    tradeoff["PACPG improvement"] = (
        abs(base_hybrid["PACPG European"]) + abs(base_hybrid["PACPG Non-English"]) + abs(base_hybrid["PACPG Long-tail"])
        - (tradeoff["PACPG European"].abs() + tradeoff["PACPG Non-English"].abs() + tradeoff["PACPG Long-tail"].abs())
    )
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(tradeoff[f"NDCG@{TOP_K}"], tradeoff["PACPG improvement"], s=90, color=COLORS["teal"])
    for _, row in tradeoff.iterrows():
        ax.annotate(f"lambda={row['lambda']:.1f}", (row[f"NDCG@{TOP_K}"], row["PACPG improvement"]), textcoords="offset points", xytext=(6, 6), fontsize=9)
    ax.axhline(0, color=COLORS["gray"], linewidth=1, linestyle="--")
    ax.set_title("Full-run utility vs cultural prominence frontier", weight="bold")
    ax.set_xlabel(f"NDCG@{TOP_K}")
    ax.set_ylabel("Reduction in absolute PACPG across audit groups")
    fig.tight_layout()
    savefig("07_utility_vs_prominence_frontier.png")
    savefig("14_full_utility_vs_prominence_frontier.png")
    plt.close(fig)


def update_readme():
    outputs_manifest = [
        "00_research_gap_matrix.png",
        "01_dataset_scale.png",
        "02_data_pipeline.png",
        "03_join_funnel.png",
        "04_popularity_long_tail.png",
        "05_catalogue_vs_interaction_share.png",
        "06_metadata_feature_coverage.png",
        "07_utility_vs_prominence_frontier.png",
        "08_group_exposure_by_model.png",
        "09_accuracy_fairness_summary.png",
        "10_workplan_gantt.png",
        "11_language_country_company_caveats.png",
        "11_metadata_proxy_risk_table.csv",
        "12_full_model_comparison.csv",
        "12_full_rerank_tradeoff.csv",
        "12_full_run_report.csv",
        "12_full_group_exposure_by_model.png",
        "13_full_accuracy_fairness_summary.png",
        "14_full_utility_vs_prominence_frontier.png",
        "15_cv_fold_results.csv",
        "15_cv_model_summary.csv",
        "15_cv_model_summary_long.csv",
        "15_cv_run_report.csv",
        "15_cv_metric_stability.png",
        "27_movies_db_file_inventory.csv",
        "27_movies_db_core_metadata_summary.csv",
        "27_movies_db_movieid_overlap.csv",
        "27_movies_db_summary_stats.csv",
        "27_movies_db_coverage_report.csv",
        "27_movies_db_coverage.png",
        "28_movies_db_rating_distribution.png",
        "29_movies_db_genre_interest_visibility.png",
        "30_movies_db_user_concentration.png",
        "data_source_ledger.csv",
        "library_ledger.csv",
        "model_comparison.csv",
    ]
    lines = [
        "# Proposal Outputs",
        "",
        "Generated by `notebooks/does_algorithm_hide_europe_realdata.ipynb` and `scripts/run_full_cultural_prominence_audit.py`.",
        "",
        "## Important scope note",
        "",
        "These outputs use real M3L/MovieLens/Wikidata data only. The full audit run loads all interaction rows but evaluates recommender models on a 5,000-user real-data sample with an expanded candidate universe for local-compute feasibility.",
        "",
        "## Files",
    ]
    for name in outputs_manifest:
        lines.append(f"- `{name}` - {'present' if (OUTPUTS / name).exists() else 'missing'}")
    lines.extend([
        "",
        "## Data governance",
        "",
        "Raw MovieLens and M3L files are not redistributed. Keep `data/raw/`, `old/`, `m3l-20m/`, `TEXT_mpnet/`, `IMG_clip-image/` and `MovieLens 20M Dataset/` out of any public submission archive unless the course explicitly permits local-only data sharing.",
        "",
        "## Additional caveats added in the full run",
        "",
        "- Language labels distinguish original language (Wikidata P364) from language of work/name (P407) where available.",
        "- Production country uses country of origin (P495), but the audit also checks production company (P272) and company/headquarters country as a proxy for US-firm involvement.",
        "- Co-productions and American-company involvement are reported as proxy risks rather than overwritten in the main labels.",
        "",
        "## Cross-validation robustness check",
        "",
        "`scripts/run_recommender_cross_validation.py` rebuilds the recommender stack on three non-overlapping user folds. The outputs report fold-level utility, exposure and PACPG stability for Popularity, ItemKNN, SVD, MPNet, CLIP-image, Hybrid and Hybrid re-ranking models.",
        "",
        "## Movies DB pipeline",
        "",
        "`scripts/build_movies_db.py` builds the combined movie-level database from MovieLens metadata, ratings, tags, genome tags, available M3L feature coverage and cached Wikidata enrichment. This is the catalogue foundation for the audit and intentionally reports missing Nico-specific raw M3L TSV files instead of fabricating plot/poster/trailer values.",
    ])
    (OUTPUTS / "README_outputs.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    interactions, movies, links, item_id_map = load_core_data()
    movie_meta, train, q20, q80 = build_metadata(interactions, movies, links, item_id_map)
    proxy_risk = plot_extra_caveats(movie_meta)
    model_comparison, rerank_tradeoff, run_report = run_model_audit(interactions, train, movie_meta)
    update_readme()

    print("\nFull audit complete.")
    print(run_report.to_string(index=False))
    print("\nMetadata proxy risk checks:")
    print(proxy_risk.to_string(index=False))
    print("\nModel comparison:")
    print(model_comparison.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
