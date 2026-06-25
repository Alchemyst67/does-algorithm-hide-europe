from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import KFold
from sklearn.preprocessing import normalize


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

SEED = 42
TOP_K = 20
CV_FOLDS = 3
CV_MAX_USERS = 3_000
CV_TOP_POPULAR_ITEMS = 4_000
CV_CANDIDATE_K = 100
CV_KNN_NEIGHBOURS = 50
CV_SVD_COMPONENTS = 64
CV_LAMBDAS = (0.3, 0.7)

M3L_INTERACTIONS = PROJECT_ROOT / "m3l-20m" / "m3l-20m.inter"
MOVIE_META = DATA_PROCESSED / "movie_audit_metadata_extended.csv"
MPNET_MATRIX = PROJECT_ROOT / "m3l-20m" / "text" / "mpnet.npy"
CLIP_MATRIX = PROJECT_ROOT / "m3l-20m" / "image" / "clip_image.npy"

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


@dataclass
class FoldContext:
    fold: int
    sample_users: np.ndarray
    candidate_items: np.ndarray
    train: pd.DataFrame
    test: pd.DataFrame
    matrix: sparse.csr_matrix
    user_to_idx: dict[int, int]
    item_to_idx: dict[int, int]
    seen_by_user_idx: dict[int, set[int]]
    test_relevant_by_user: dict[int, set[int]]
    group_flags: dict[str, dict[int, bool]]
    relevant_group_share: dict[str, dict[int, float]]
    user_group_targets: dict[int, dict[str, float]]


GROUP_COLUMNS = {
    "European": "is_european",
    "Non-English": "is_non_english",
    "Long-tail": "is_long_tail",
}


def rank_discount(rank: int) -> float:
    return 1 / math.log2(rank + 1)


def recall_at_k(items: list[int], relevant: set[int], k: int = TOP_K) -> float:
    if not relevant:
        return np.nan
    return len(set(items[:k]) & relevant) / len(relevant)


def ndcg_at_k(items: list[int], relevant: set[int], k: int = TOP_K) -> float:
    if not relevant:
        return np.nan
    dcg = sum(rank_discount(rank) for rank, item in enumerate(items[:k], start=1) if item in relevant)
    ideal = sum(rank_discount(rank) for rank in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal > 0 else np.nan


def map_at_k(items: list[int], relevant: set[int], k: int = TOP_K) -> float:
    if not relevant:
        return np.nan
    hits = 0
    precisions = []
    for rank, item in enumerate(items[:k], start=1):
        if item in relevant:
            hits += 1
            precisions.append(hits / rank)
    return float(np.sum(precisions) / min(len(relevant), k)) if precisions else 0.0


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    interactions = pd.read_csv(
        M3L_INTERACTIONS,
        sep="\t",
        dtype={"userID": "int32", "itemID": "int32", "rating": "float32", "x_label": "int8"},
    ).rename(columns={"userID": "user_id", "itemID": "item_id", "rating": "rating_or_weight", "x_label": "split"})
    interactions["split"] = interactions["split"].map({0: "train", 1: "valid", 2: "test"})

    movie_meta = pd.read_csv(MOVIE_META)
    for col in GROUP_COLUMNS.values():
        movie_meta[col] = movie_meta[col].fillna(False).astype(bool)
    return interactions, movie_meta


def eligible_users(interactions: pd.DataFrame) -> np.ndarray:
    train_counts = interactions[interactions["split"].eq("train")].groupby("user_id").size()
    test_counts = interactions[interactions["split"].eq("test")].groupby("user_id").size()
    users = sorted(set(train_counts[train_counts >= 5].index) & set(test_counts[test_counts >= 1].index))
    return np.array(users, dtype=np.int32)


def make_user_folds(users: np.ndarray) -> list[np.ndarray]:
    rng = np.random.default_rng(SEED)
    users = users.copy()
    rng.shuffle(users)
    users = users[: min(CV_MAX_USERS, len(users))]
    splitter = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=SEED)
    return [np.sort(users[test_idx]).astype(np.int32) for _, test_idx in splitter.split(users)]


def prepare_fold_context(
    fold: int,
    fold_users: np.ndarray,
    interactions: pd.DataFrame,
    movie_meta: pd.DataFrame,
    global_item_counts: pd.DataFrame,
) -> FoldContext:
    train = interactions[interactions["split"].eq("train") & interactions["user_id"].isin(fold_users)].copy()
    test = interactions[interactions["split"].eq("test") & interactions["user_id"].isin(fold_users)].copy()

    top_items = (
        global_item_counts.sort_values("train_interaction_count", ascending=False)["item_id"]
        .head(CV_TOP_POPULAR_ITEMS)
        .astype(int)
        .tolist()
    )
    candidate_items = sorted(set(top_items) | set(train["item_id"].astype(int)) | set(test["item_id"].astype(int)))
    candidate_items = np.array(candidate_items, dtype=np.int32)

    user_to_idx = {int(u): i for i, u in enumerate(fold_users)}
    item_to_idx = {int(it): j for j, it in enumerate(candidate_items)}
    train = train[train["item_id"].isin(item_to_idx)]
    test = test[test["item_id"].isin(item_to_idx)]

    rows = train["user_id"].map(user_to_idx).to_numpy()
    cols = train["item_id"].map(item_to_idx).to_numpy()
    values = (train["rating_or_weight"].to_numpy() > 0).astype(np.float32)
    matrix = sparse.csr_matrix((values, (rows, cols)), shape=(len(fold_users), len(candidate_items)))

    seen_by_user_idx = {
        int(uid): set(group["item_id"].map(item_to_idx).dropna().astype(int))
        for uid, group in train.groupby("user_id")
    }
    test_relevant_by_user = {
        int(uid): set(group["item_id"].astype(int))
        for uid, group in test[test["rating_or_weight"] > 0].groupby("user_id")
    }

    candidate_meta = (
        movie_meta[["item_id", *GROUP_COLUMNS.values()]]
        .set_index("item_id")
        .reindex(candidate_items)
        .fillna(False)
    )
    group_flags = {
        group: candidate_meta[col].astype(bool).to_dict()
        for group, col in GROUP_COLUMNS.items()
    }

    train_groups = train.merge(movie_meta[["item_id", *GROUP_COLUMNS.values()]], on="item_id", how="left")
    test_groups = test[test["rating_or_weight"] > 0].merge(movie_meta[["item_id", *GROUP_COLUMNS.values()]], on="item_id", how="left")
    history_share = {
        group: train_groups.groupby("user_id")[col].mean().fillna(0).to_dict()
        for group, col in GROUP_COLUMNS.items()
    }
    relevant_share = {
        group: test_groups.groupby("user_id")[col].mean().fillna(0).to_dict()
        for group, col in GROUP_COLUMNS.items()
    }
    targets = {
        int(uid): {
            group: max(float(history_share[group].get(int(uid), 0.0)), float(relevant_share[group].get(int(uid), 0.0)))
            for group in GROUP_COLUMNS
        }
        for uid in fold_users
    }

    return FoldContext(
        fold=fold,
        sample_users=fold_users,
        candidate_items=candidate_items,
        train=train,
        test=test,
        matrix=matrix,
        user_to_idx=user_to_idx,
        item_to_idx=item_to_idx,
        seen_by_user_idx=seen_by_user_idx,
        test_relevant_by_user=test_relevant_by_user,
        group_flags=group_flags,
        relevant_group_share=relevant_share,
        user_group_targets=targets,
    )


def mask_seen(scores: np.ndarray, context: FoldContext) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32).copy()
    for uid in context.sample_users:
        ui = context.user_to_idx[int(uid)]
        seen = context.seen_by_user_idx.get(int(uid), set())
        if seen:
            scores[ui, list(seen)] = -np.inf
    return scores


def recommend_from_scores(scores: np.ndarray, context: FoldContext, k: int = TOP_K) -> dict[int, list[int]]:
    masked = mask_seen(scores, context)
    recs: dict[int, list[int]] = {}
    for uid in context.sample_users:
        uid = int(uid)
        ui = context.user_to_idx[uid]
        user_scores = masked[ui]
        finite = np.isfinite(user_scores)
        if finite.sum() == 0:
            recs[uid] = []
            continue
        k_eff = min(k, int(finite.sum()))
        top_idx = np.argpartition(-user_scores, k_eff - 1)[:k_eff]
        top_idx = top_idx[np.argsort(-user_scores[top_idx])]
        recs[uid] = [int(context.candidate_items[j]) for j in top_idx]
    return recs


def rowwise_standardise(scores: np.ndarray) -> np.ndarray:
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


def content_scores(feature_path: Path, context: FoldContext) -> np.ndarray:
    features = np.load(feature_path, mmap_mode="r")[context.candidate_items].astype(np.float32)
    features = normalize(features, norm="l2", axis=1)
    profile_sum = context.matrix @ features
    counts = np.maximum(context.matrix.sum(axis=1).A1, 1.0).astype(np.float32)
    profiles = normalize(profile_sum / counts[:, None], norm="l2", axis=1)
    return (profiles @ features.T).astype(np.float32)


def model_scores(context: FoldContext, global_item_counts: pd.DataFrame) -> dict[str, np.ndarray]:
    candidate_pop = (
        global_item_counts.set_index("item_id")
        .reindex(context.candidate_items)["train_interaction_count"]
        .fillna(0)
        .to_numpy(np.float32)
    )
    popularity = np.tile(np.log1p(candidate_pop), (len(context.sample_users), 1)).astype(np.float32)

    item_user = context.matrix.T.astype(np.float32)
    item_norms = np.sqrt(item_user.multiply(item_user).sum(axis=1)).A1
    item_norms[item_norms == 0] = 1.0
    item_user_norm = item_user.multiply(1 / item_norms[:, None])
    similarity = (item_user_norm @ item_user_norm.T).toarray().astype(np.float32)
    np.fill_diagonal(similarity, 0.0)
    keep = min(CV_KNN_NEIGHBOURS, similarity.shape[1])
    if keep < similarity.shape[1]:
        threshold_idx = np.argpartition(-similarity, keep, axis=1)[:, keep:]
        row_idx = np.arange(similarity.shape[0])[:, None]
        similarity[row_idx, threshold_idx] = 0.0
    itemknn = (context.matrix @ similarity).astype(np.float32)
    del similarity

    n_components = min(CV_SVD_COMPONENTS, max(2, min(context.matrix.shape) - 1))
    svd = TruncatedSVD(n_components=n_components, random_state=SEED + context.fold)
    user_factors = svd.fit_transform(context.matrix).astype(np.float32)
    svd_scores = (user_factors @ svd.components_.astype(np.float32)).astype(np.float32)

    mpnet = content_scores(MPNET_MATRIX, context)
    clip = content_scores(CLIP_MATRIX, context)
    hybrid = (
        0.50 * rowwise_standardise(svd_scores)
        + 0.25 * rowwise_standardise(mpnet)
        + 0.25 * rowwise_standardise(clip)
    ).astype(np.float32)

    return {
        "Popularity": popularity,
        "ItemKNN": itemknn,
        "SVD": svd_scores,
        "MPNet-content": mpnet,
        "CLIP-image-content": clip,
        "Hybrid": hybrid,
    }


def rerank_hybrid(context: FoldContext, hybrid_scores: np.ndarray, lambda_value: float) -> dict[int, list[int]]:
    masked = mask_seen(hybrid_scores, context)
    recs: dict[int, list[int]] = {}
    for uid in context.sample_users:
        uid = int(uid)
        ui = context.user_to_idx[uid]
        scores = masked[ui]
        finite = np.isfinite(scores)
        if finite.sum() == 0:
            recs[uid] = []
            continue
        pool_size = min(CV_CANDIDATE_K, int(finite.sum()))
        pool_idx = np.argpartition(-scores, pool_size - 1)[:pool_size]
        pool_idx = pool_idx[np.argsort(-scores[pool_idx])]
        pool_items = [int(context.candidate_items[j]) for j in pool_idx]
        rel_scores = scores[pool_idx].astype(np.float32)
        rel_std = rel_scores.std()
        rel_scores = (rel_scores - rel_scores.mean()) / rel_std if rel_std > 1e-8 else rel_scores * 0
        rel_lookup = dict(zip(pool_items, rel_scores))

        selected: list[int] = []
        targets = context.user_group_targets.get(uid, {group: 0.0 for group in GROUP_COLUMNS})
        for _ in range(TOP_K):
            best_item = None
            best_score = -np.inf
            current_counts = {
                group: sum(bool(context.group_flags[group].get(item, False)) for item in selected)
                for group in GROUP_COLUMNS
            }
            denom = len(selected) + 1
            for item in pool_items:
                if item in selected:
                    continue
                bonus = 0.0
                for group in GROUP_COLUMNS:
                    flag = bool(context.group_flags[group].get(item, False))
                    current_share = (current_counts[group] + int(flag)) / denom
                    target = float(targets.get(group, 0.0))
                    if flag and current_share < target:
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


def discounted_group_exposure(items: list[int], context: FoldContext, group_name: str, k: int = TOP_K) -> float:
    numerator = 0.0
    denominator = 0.0
    flags = context.group_flags[group_name]
    for rank, item in enumerate(items[:k], start=1):
        weight = rank_discount(rank)
        denominator += weight
        numerator += weight * float(flags.get(item, False))
    return numerator / denominator if denominator else np.nan


def evaluate_recommendations(context: FoldContext, recs_by_model: dict[str, dict[int, list[int]]]) -> pd.DataFrame:
    rows = []
    for model_name, recs in recs_by_model.items():
        recalls, ndcgs, maps = [], [], []
        exposure_values = {group: [] for group in GROUP_COLUMNS}
        prominence_gaps = {group: [] for group in GROUP_COLUMNS}
        pacpg_values = {group: [] for group in GROUP_COLUMNS}
        covered_items = set()

        for uid, items in recs.items():
            relevant = context.test_relevant_by_user.get(uid, set())
            if not relevant:
                continue
            covered_items.update(items[:TOP_K])
            recalls.append(recall_at_k(items, relevant))
            ndcgs.append(ndcg_at_k(items, relevant))
            maps.append(map_at_k(items, relevant))
            for group in GROUP_COLUMNS:
                exposure = discounted_group_exposure(items, context, group)
                relevant_share = float(context.relevant_group_share[group].get(uid, 0.0))
                target = float(context.user_group_targets.get(uid, {}).get(group, 0.0))
                exposure_values[group].append(exposure)
                prominence_gaps[group].append(exposure - relevant_share)
                pacpg_values[group].append(exposure - target)

        row = {
            "Fold": context.fold,
            "Model": model_name,
            "Users": len(context.sample_users),
            "Candidate items": len(context.candidate_items),
            "Train rows": len(context.train),
            "Test rows": len(context.test),
            f"NDCG@{TOP_K}": np.nanmean(ndcgs),
            f"Recall@{TOP_K}": np.nanmean(recalls),
            f"MAP@{TOP_K}": np.nanmean(maps),
            f"Coverage@{TOP_K}": len(covered_items) / len(context.candidate_items),
        }
        for group in GROUP_COLUMNS:
            row[f"{group} Exposure@{TOP_K}"] = np.nanmean(exposure_values[group])
            row[f"ProminenceGap {group}"] = np.nanmean(prominence_gaps[group])
            row[f"PACPG {group}"] = np.nanmean(pacpg_values[group])
        rows.append(row)

    return pd.DataFrame(rows)


def run_fold(
    fold: int,
    fold_users: np.ndarray,
    interactions: pd.DataFrame,
    movie_meta: pd.DataFrame,
    global_item_counts: pd.DataFrame,
) -> pd.DataFrame:
    context = prepare_fold_context(fold, fold_users, interactions, movie_meta, global_item_counts)
    print(
        f"Fold {fold}: {len(context.sample_users):,} users, "
        f"{len(context.candidate_items):,} candidate items, {len(context.train):,} train rows."
    )
    scores = model_scores(context, global_item_counts)
    recs = {model: recommend_from_scores(score_matrix, context) for model, score_matrix in scores.items()}
    for lam in CV_LAMBDAS:
        recs[f"Hybrid + reranking lambda={lam:.1f}"] = rerank_hybrid(context, scores["Hybrid"], lam)
    return evaluate_recommendations(context, recs)


def summarise_folds(fold_results: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_cols = [
        col for col in fold_results.columns
        if col not in {"Fold", "Model", "Users", "Candidate items", "Train rows", "Test rows"}
    ]
    summary = fold_results.groupby("Model")[metric_cols].agg(["mean", "std", "min", "max"])
    summary.columns = [f"{metric} {stat}" for metric, stat in summary.columns]
    summary = summary.reset_index()

    long_summary = []
    for model, group in fold_results.groupby("Model"):
        for metric in metric_cols:
            values = group[metric].astype(float)
            long_summary.append({
                "Model": model,
                "Metric": metric,
                "mean": values.mean(),
                "std": values.std(ddof=1),
                "min": values.min(),
                "max": values.max(),
            })
    return summary, pd.DataFrame(long_summary)


def plot_cv_summary(summary: pd.DataFrame) -> None:
    summary = summary.copy()
    summary["Absolute PACPG mean"] = (
        summary["PACPG European mean"].abs()
        + summary["PACPG Non-English mean"].abs()
        + summary["PACPG Long-tail mean"].abs()
    )
    summary["Absolute PACPG std proxy"] = (
        summary["PACPG European std"].fillna(0).abs()
        + summary["PACPG Non-English std"].fillna(0).abs()
        + summary["PACPG Long-tail std"].fillna(0).abs()
    )

    order = [
        "Popularity", "ItemKNN", "SVD", "MPNet-content", "CLIP-image-content",
        "Hybrid", "Hybrid + reranking lambda=0.3", "Hybrid + reranking lambda=0.7",
    ]
    summary["Model"] = pd.Categorical(summary["Model"], categories=order, ordered=True)
    summary = summary.sort_values("Model")
    x = np.arange(len(summary))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    axes[0].bar(x, summary[f"NDCG@{TOP_K} mean"], yerr=summary[f"NDCG@{TOP_K} std"], color=COLORS["teal"], capsize=4)
    axes[0].set_title("Cross-fold utility stability", weight="bold")
    axes[0].set_ylabel(f"Mean NDCG@{TOP_K} +/- fold std")
    axes[0].set_xticks(x, summary["Model"], rotation=35, ha="right")

    axes[1].bar(x, summary["Absolute PACPG mean"], yerr=summary["Absolute PACPG std proxy"], color=COLORS["violet"], capsize=4)
    axes[1].set_title("Cross-fold cultural-prominence stability", weight="bold")
    axes[1].set_ylabel("Mean absolute PACPG across groups +/- fold std proxy")
    axes[1].set_xticks(x, summary["Model"], rotation=35, ha="right")

    fig.suptitle("User-fold cross-validation robustness check", weight="bold", y=1.03)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "15_cv_metric_stability.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    interactions, movie_meta = load_inputs()
    train_all = interactions[interactions["split"].eq("train")][["user_id", "item_id", "rating_or_weight"]].copy()
    global_item_counts = train_all.groupby("item_id").size().rename("train_interaction_count").reset_index()

    users = eligible_users(interactions)
    folds = make_user_folds(users)
    print(f"Cross-validation protocol: {len(folds)} user folds, {sum(len(f) for f in folds):,} sampled users.")

    fold_results = []
    for fold, fold_users in enumerate(folds, start=1):
        fold_results.append(run_fold(fold, fold_users, interactions, movie_meta, global_item_counts))
    fold_results = pd.concat(fold_results, ignore_index=True)

    summary, long_summary = summarise_folds(fold_results)
    fold_results.to_csv(OUTPUTS / "15_cv_fold_results.csv", index=False)
    summary.to_csv(OUTPUTS / "15_cv_model_summary.csv", index=False)
    long_summary.to_csv(OUTPUTS / "15_cv_model_summary_long.csv", index=False)

    run_report = pd.DataFrame([
        {"setting": "folds", "value": CV_FOLDS},
        {"setting": "sampled users", "value": sum(len(fold) for fold in folds)},
        {"setting": "users per fold", "value": len(folds[0]) if folds else 0},
        {"setting": "top popular candidate items", "value": CV_TOP_POPULAR_ITEMS},
        {"setting": "candidate rerank pool", "value": CV_CANDIDATE_K},
        {"setting": "top k", "value": TOP_K},
    ])
    run_report.to_csv(OUTPUTS / "15_cv_run_report.csv", index=False)
    plot_cv_summary(summary)

    print("\nCross-validation complete.")
    print(run_report.to_string(index=False))
    print("\nSummary:")
    print(summary.round(4).to_string(index=False))


if __name__ == "__main__":
    main()
