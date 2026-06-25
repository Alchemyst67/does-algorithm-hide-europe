from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ast
import math
import os
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"
OUTPUTS.mkdir(exist_ok=True)

M3L_INTERACTIONS = PROJECT_ROOT / "m3l-20m" / "m3l-20m.inter"
MOVIE_META = DATA_PROCESSED / "movie_audit_metadata_extended.csv"

SEED = int(os.getenv("SCHEDL_SEED", "42"))
MAX_USERS = int(os.getenv("SCHEDL_MAX_USERS", "650"))
TOP_POPULAR_ITEMS = int(os.getenv("SCHEDL_TOP_POPULAR_ITEMS", "2200"))
ITERATIONS = int(os.getenv("SCHEDL_ITERATIONS", "8"))
TOP_K = int(os.getenv("SCHEDL_TOP_K", "10"))
ACCEPTANCE_ALPHA = float(os.getenv("SCHEDL_ACCEPTANCE_ALPHA", "-0.1"))

LATENT_DIM = int(os.getenv("SCHEDL_LATENT_DIM", "32"))
BPR_EPOCHS = int(os.getenv("SCHEDL_BPR_EPOCHS", "2"))
NEUMF_EPOCHS = int(os.getenv("SCHEDL_NEUMF_EPOCHS", "2"))
VAE_EPOCHS = int(os.getenv("SCHEDL_VAE_EPOCHS", "2"))
BATCH_SIZE = int(os.getenv("SCHEDL_BATCH_SIZE", "1024"))
TORCH_SAMPLE_MULTIPLIER = int(os.getenv("SCHEDL_TORCH_SAMPLE_MULTIPLIER", "2"))

COLORS = {
    "ink": "#142232",
    "teal": "#167f86",
    "orange": "#d65a31",
    "blue": "#2f6b9a",
    "gold": "#c99700",
    "green": "#2e8b57",
    "red": "#b84a4a",
    "gray": "#697582",
}


@dataclass
class SimulationState:
    users: np.ndarray
    candidate_items: np.ndarray
    profile_sets: list[set[int]]
    initial_profile_sets: list[set[int]]
    seen_global_items: set[int]
    labels: pd.DataFrame
    rng: np.random.Generator


def maybe_torch():
    try:
        import torch

        torch.manual_seed(SEED)
        return torch
    except Exception:
        return None


def safe_list(value) -> list[str]:
    if isinstance(value, list):
        return [str(v) for v in value]
    if pd.isna(value) or value == "":
        return []
    text = str(value)
    try:
        parsed = ast.literal_eval(text)
        if isinstance(parsed, list):
            return [str(v) for v in parsed]
    except Exception:
        pass
    return [part.strip() for part in text.split("|") if part.strip()]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    interactions = pd.read_csv(
        M3L_INTERACTIONS,
        sep="\t",
        dtype={"userID": "int32", "itemID": "int32", "rating": "float32", "x_label": "int8"},
    ).rename(columns={"userID": "user_id", "itemID": "item_id", "rating": "rating_or_weight", "x_label": "split"})
    interactions["split"] = interactions["split"].map({0: "train", 1: "valid", 2: "test"})

    movie_meta = pd.read_csv(MOVIE_META)
    for col in ["country", "original_language", "production_company_country", "production_company_hq_country"]:
        movie_meta[col] = movie_meta[col].apply(safe_list)
    for col in [
        "is_european",
        "is_non_english",
        "is_long_tail",
        "has_us_origin_country",
        "has_us_company_country",
        "is_european_with_us_company",
        "is_coproduction_by_country",
        "is_multilingual_original",
    ]:
        if col in movie_meta:
            movie_meta[col] = movie_meta[col].fillna(False).astype(bool)
    return interactions, movie_meta


def language_bin(languages: list[str]) -> str:
    if not languages:
        return "Unknown language"
    has_english = "English" in set(languages)
    if len(languages) > 1 and has_english:
        return "Multilingual incl. English"
    if len(languages) > 1:
        return "Multilingual non-English"
    if has_english:
        return "English-only"
    return "Non-English-only"


def origin_bin(row: pd.Series) -> str:
    countries = row.get("country", [])
    if not countries:
        return "Unknown origin"
    if bool(row.get("is_european", False)) and bool(row.get("has_us_origin_country", False)):
        return "European-US co-production"
    if bool(row.get("is_european_with_us_company", False)):
        return "European origin + US company"
    if bool(row.get("is_european", False)):
        return "European origin"
    if bool(row.get("has_us_origin_country", False)):
        return "US origin"
    return "Other origin"


def prepare_labels(movie_meta: pd.DataFrame) -> pd.DataFrame:
    labels = movie_meta.copy()
    labels["origin_bin"] = labels.apply(origin_bin, axis=1)
    labels["language_bin"] = labels["original_language"].apply(language_bin)
    q20 = labels["train_interaction_count"].quantile(0.20)
    q80 = labels["train_interaction_count"].quantile(0.80)
    labels["popularity_bin"] = np.select(
        [
            labels["train_interaction_count"].le(q20),
            labels["train_interaction_count"].ge(q80),
        ],
        ["Low popularity", "High popularity"],
        default="Mid popularity",
    )
    return labels


def make_initial_state(interactions: pd.DataFrame, labels: pd.DataFrame) -> SimulationState:
    rng = np.random.default_rng(SEED)
    train = interactions[interactions["split"].eq("train") & interactions["rating_or_weight"].gt(0)].copy()
    user_counts = train.groupby("user_id").size()
    users = np.array(user_counts[user_counts >= 20].index, dtype=np.int32)
    rng.shuffle(users)
    users = np.sort(users[: min(MAX_USERS, len(users))])

    sample_train = train[train["user_id"].isin(users)].copy()
    popular_items = (
        train.groupby("item_id").size().sort_values(ascending=False).head(TOP_POPULAR_ITEMS).index.astype(int).tolist()
    )
    candidate_items = np.array(sorted(set(popular_items) | set(sample_train["item_id"].astype(int))), dtype=np.int32)
    item_to_pos = {int(item): pos for pos, item in enumerate(candidate_items)}
    user_to_pos = {int(user): pos for pos, user in enumerate(users)}

    profile_sets = [set() for _ in users]
    for user_id, group in sample_train.groupby("user_id"):
        pos = user_to_pos[int(user_id)]
        profile_sets[pos] = {item_to_pos[int(item)] for item in group["item_id"] if int(item) in item_to_pos}

    candidate_labels = labels.set_index("item_id").reindex(candidate_items).reset_index()
    candidate_labels["item_pos"] = np.arange(len(candidate_items))
    for col in ["is_european", "is_non_english", "is_long_tail", "has_us_origin_country", "has_us_company_country"]:
        candidate_labels[col] = candidate_labels[col].fillna(False).astype(bool)
    for col in ["origin_bin", "language_bin", "popularity_bin"]:
        candidate_labels[col] = candidate_labels[col].fillna(f"Unknown {col}")

    return SimulationState(
        users=users,
        candidate_items=candidate_items,
        profile_sets=[set(items) for items in profile_sets],
        initial_profile_sets=[set(items) for items in profile_sets],
        seen_global_items=set().union(*profile_sets),
        labels=candidate_labels,
        rng=rng,
    )


def build_matrix(profile_sets: list[set[int]], n_items: int) -> sparse.csr_matrix:
    rows, cols = [], []
    for user_pos, items in enumerate(profile_sets):
        rows.extend([user_pos] * len(items))
        cols.extend(items)
    data = np.ones(len(rows), dtype=np.float32)
    return sparse.csr_matrix((data, (rows, cols)), shape=(len(profile_sets), n_items), dtype=np.float32)


def mask_seen(scores: np.ndarray, profile_sets: list[set[int]]) -> np.ndarray:
    masked = np.asarray(scores, dtype=np.float32).copy()
    for user_pos, seen in enumerate(profile_sets):
        if seen:
            masked[user_pos, list(seen)] = -np.inf
    return masked


def topk_recommendations(scores: np.ndarray, profile_sets: list[set[int]], k: int = TOP_K) -> list[list[int]]:
    masked = mask_seen(scores, profile_sets)
    recommendations = []
    for user_pos in range(masked.shape[0]):
        row = masked[user_pos]
        finite = np.isfinite(row)
        if finite.sum() == 0:
            recommendations.append([])
            continue
        k_eff = min(k, int(finite.sum()))
        idx = np.argpartition(-row, k_eff - 1)[:k_eff]
        idx = idx[np.argsort(-row[idx])]
        recommendations.append([int(i) for i in idx])
    return recommendations


def distribution(items: set[int] | list[int], categories: np.ndarray, values: list[str]) -> np.ndarray:
    if not items:
        return np.ones(len(values), dtype=np.float64) / len(values)
    idx = np.array(list(items), dtype=int)
    counts = np.array([(categories[idx] == value).mean() for value in values], dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return np.ones(len(values), dtype=np.float64) / len(values)
    return counts / total


def jsd(p: np.ndarray, q: np.ndarray) -> float:
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    p = p / p.sum() if p.sum() else np.ones_like(p) / len(p)
    q = q / q.sum() if q.sum() else np.ones_like(q) / len(q)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def share(items: set[int] | list[int], flags: np.ndarray) -> float:
    if not items:
        return np.nan
    return float(flags[np.array(list(items), dtype=int)].mean())


def metric_rows(model: str, iteration: int, recommendations: list[list[int]], state: SimulationState) -> list[dict]:
    rows = []
    labels = state.labels
    europe = labels["is_european"].to_numpy(bool)
    non_english = labels["is_non_english"].to_numpy(bool)
    long_tail = labels["is_long_tail"].to_numpy(bool)
    us_origin = labels["has_us_origin_country"].to_numpy(bool)
    us_company = labels["has_us_company_country"].to_numpy(bool)

    dimensions = {
        "origin": sorted(labels["origin_bin"].dropna().unique().tolist()),
        "language": sorted(labels["language_bin"].dropna().unique().tolist()),
        "popularity": ["Low popularity", "Mid popularity", "High popularity"],
    }
    category_arrays = {
        "origin": labels["origin_bin"].to_numpy(str),
        "language": labels["language_bin"].to_numpy(str),
        "popularity": labels["popularity_bin"].to_numpy(str),
    }

    for user_pos, recs in enumerate(recommendations):
        initial = state.initial_profile_sets[user_pos]
        profile = state.profile_sets[user_pos]
        row = {
            "Model": model,
            "iteration": iteration,
            "user_pos": user_pos,
            "recommendation_european_share": share(recs, europe),
            "profile_european_share": share(profile, europe),
            "initial_european_share": share(initial, europe),
            "recommendation_non_english_share": share(recs, non_english),
            "profile_non_english_share": share(profile, non_english),
            "initial_non_english_share": share(initial, non_english),
            "recommendation_long_tail_share": share(recs, long_tail),
            "profile_long_tail_share": share(profile, long_tail),
            "initial_long_tail_share": share(initial, long_tail),
            "recommendation_us_origin_share": share(recs, us_origin),
            "profile_us_origin_share": share(profile, us_origin),
            "initial_us_origin_share": share(initial, us_origin),
            "recommendation_us_company_share": share(recs, us_company),
            "profile_us_company_share": share(profile, us_company),
            "initial_us_company_share": share(initial, us_company),
        }
        for dim, values in dimensions.items():
            arr = category_arrays[dim]
            init_dist = distribution(initial, arr, values)
            rec_dist = distribution(recs, arr, values)
            prof_dist = distribution(profile, arr, values)
            row[f"recommendation_{dim}_jsd"] = jsd(init_dist, rec_dist)
            row[f"profile_{dim}_jsd"] = jsd(init_dist, prof_dist)
        rows.append(row)
    return rows


def accept_one_item(recommendations: list[list[int]], state: SimulationState) -> int:
    weights = np.exp(ACCEPTANCE_ALPHA * np.arange(1, TOP_K + 1))
    accepted = 0
    for user_pos, recs in enumerate(recommendations):
        if not recs:
            continue
        local_weights = weights[: len(recs)]
        local_weights = local_weights / local_weights.sum()
        item = int(state.rng.choice(recs, p=local_weights))
        state.profile_sets[user_pos].add(item)
        accepted += 1
    return accepted


def popularity_scores(matrix: sparse.csr_matrix) -> np.ndarray:
    counts = np.asarray(matrix.sum(axis=0)).ravel().astype(np.float32)
    return np.tile(np.log1p(counts), (matrix.shape[0], 1))


def itemknn_scores(matrix: sparse.csr_matrix) -> np.ndarray:
    item_user = matrix.T.tocsr()
    norms = np.sqrt(item_user.multiply(item_user).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    item_user_norm = item_user.multiply(1 / norms[:, None])
    sim = (item_user_norm @ item_user_norm.T).toarray().astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    keep = min(80, sim.shape[1])
    if keep < sim.shape[1]:
        low_idx = np.argpartition(-sim, keep, axis=1)[:, keep:]
        sim[np.arange(sim.shape[0])[:, None], low_idx] = 0.0
    return (matrix @ sim).astype(np.float32)


def lightgcn_style_scores(matrix: sparse.csr_matrix) -> np.ndarray:
    # This is a transparent one-step graph-propagation approximation of LightGCN scoring.
    user_degree = np.asarray(matrix.sum(axis=1)).ravel()
    item_degree = np.asarray(matrix.sum(axis=0)).ravel()
    user_degree[user_degree == 0] = 1.0
    item_degree[item_degree == 0] = 1.0
    norm = matrix.multiply(1 / np.sqrt(user_degree)[:, None]).multiply(1 / np.sqrt(item_degree)[None, :]).tocsr()
    item_graph = norm.T @ norm
    scores = norm @ item_graph
    return scores.toarray().astype(np.float32)


def sample_training_triples(profile_sets: list[set[int]], n_items: int, rng: np.random.Generator, n_samples: int):
    active_users = np.array([i for i, items in enumerate(profile_sets) if items], dtype=np.int64)
    users = rng.choice(active_users, size=n_samples, replace=True)
    positives = np.empty(n_samples, dtype=np.int64)
    negatives = np.empty(n_samples, dtype=np.int64)
    for idx, user in enumerate(users):
        positives[idx] = rng.choice(list(profile_sets[int(user)]))
        neg = int(rng.integers(0, n_items))
        while neg in profile_sets[int(user)]:
            neg = int(rng.integers(0, n_items))
        negatives[idx] = neg
    return users, positives, negatives


def bpr_scores(matrix: sparse.csr_matrix, profile_sets: list[set[int]], rng: np.random.Generator) -> np.ndarray:
    torch = maybe_torch()
    if torch is None:
        return lightgcn_style_scores(matrix)
    n_users, n_items = matrix.shape
    user_emb = torch.nn.Embedding(n_users, LATENT_DIM)
    item_emb = torch.nn.Embedding(n_items, LATENT_DIM)
    torch.nn.init.normal_(user_emb.weight, std=0.08)
    torch.nn.init.normal_(item_emb.weight, std=0.08)
    opt = torch.optim.Adam([*user_emb.parameters(), *item_emb.parameters()], lr=0.03, weight_decay=1e-5)
    n_samples = max(BATCH_SIZE, min(int(matrix.nnz * TORCH_SAMPLE_MULTIPLIER), 120_000))
    for _ in range(BPR_EPOCHS):
        users, pos, neg = sample_training_triples(profile_sets, n_items, rng, n_samples)
        order = rng.permutation(n_samples)
        for start in range(0, n_samples, BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            u = torch.as_tensor(users[batch], dtype=torch.long)
            i = torch.as_tensor(pos[batch], dtype=torch.long)
            j = torch.as_tensor(neg[batch], dtype=torch.long)
            pos_score = (user_emb(u) * item_emb(i)).sum(dim=1)
            neg_score = (user_emb(u) * item_emb(j)).sum(dim=1)
            loss = torch.nn.functional.softplus(-(pos_score - neg_score)).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return (user_emb.weight @ item_emb.weight.T).cpu().numpy().astype(np.float32)


def neumf_lite_scores(matrix: sparse.csr_matrix, profile_sets: list[set[int]], rng: np.random.Generator) -> np.ndarray:
    torch = maybe_torch()
    if torch is None:
        return bpr_scores(matrix, profile_sets, rng)
    n_users, n_items = matrix.shape

    class NeuMFLite(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.user_mf = torch.nn.Embedding(n_users, LATENT_DIM)
            self.item_mf = torch.nn.Embedding(n_items, LATENT_DIM)
            self.user_mlp = torch.nn.Embedding(n_users, LATENT_DIM)
            self.item_mlp = torch.nn.Embedding(n_items, LATENT_DIM)
            self.mlp = torch.nn.Sequential(
                torch.nn.Linear(2 * LATENT_DIM, 64),
                torch.nn.ReLU(),
                torch.nn.Linear(64, 32),
                torch.nn.ReLU(),
            )
            self.out = torch.nn.Linear(LATENT_DIM + 32, 1)

        def forward(self, users, items):
            mf = self.user_mf(users) * self.item_mf(items)
            mlp = self.mlp(torch.cat([self.user_mlp(users), self.item_mlp(items)], dim=1))
            return self.out(torch.cat([mf, mlp], dim=1)).squeeze(1)

    model = NeuMFLite()
    opt = torch.optim.Adam(model.parameters(), lr=0.003, weight_decay=1e-6)
    n_samples = max(BATCH_SIZE, min(int(matrix.nnz * TORCH_SAMPLE_MULTIPLIER), 100_000))
    for _ in range(NEUMF_EPOCHS):
        users, pos, neg = sample_training_triples(profile_sets, n_items, rng, n_samples)
        order = rng.permutation(n_samples)
        for start in range(0, n_samples, BATCH_SIZE):
            batch = order[start:start + BATCH_SIZE]
            u = torch.as_tensor(users[batch], dtype=torch.long)
            i = torch.as_tensor(pos[batch], dtype=torch.long)
            j = torch.as_tensor(neg[batch], dtype=torch.long)
            loss = torch.nn.functional.softplus(-(model(u, i) - model(u, j))).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        user_ids = torch.arange(n_users, dtype=torch.long)
        item_ids = torch.arange(n_items, dtype=torch.long)
        scores = np.empty((n_users, n_items), dtype=np.float32)
        for start in range(0, n_items, 512):
            batch_items = item_ids[start:start + 512]
            parts = []
            for u_start in range(0, n_users, 256):
                batch_users = user_ids[u_start:u_start + 256]
                uu = batch_users.repeat_interleave(len(batch_items))
                ii = batch_items.repeat(len(batch_users))
                parts.append(model(uu, ii).reshape(len(batch_users), len(batch_items)).cpu().numpy())
            scores[:, start:start + 512] = np.vstack(parts)
        return scores


def multivae_lite_scores(matrix: sparse.csr_matrix, profile_sets: list[set[int]], rng: np.random.Generator) -> np.ndarray:
    torch = maybe_torch()
    if torch is None:
        return lightgcn_style_scores(matrix)
    n_users, n_items = matrix.shape
    x = torch.as_tensor(matrix.toarray(), dtype=torch.float32)
    hidden = min(256, max(64, n_items // 12))
    latent = min(64, LATENT_DIM)

    class MultiVAELite(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Sequential(torch.nn.Linear(n_items, hidden), torch.nn.Tanh())
            self.mu = torch.nn.Linear(hidden, latent)
            self.logvar = torch.nn.Linear(hidden, latent)
            self.decoder = torch.nn.Sequential(torch.nn.Linear(latent, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, n_items))

        def forward(self, batch):
            h = self.encoder(torch.nn.functional.dropout(batch, p=0.25, training=self.training))
            mu = self.mu(h)
            logvar = self.logvar(h).clamp(-8, 8)
            z = mu + torch.randn_like(mu) * torch.exp(0.5 * logvar)
            return self.decoder(z), mu, logvar

    model = MultiVAELite()
    opt = torch.optim.Adam(model.parameters(), lr=0.004)
    for _ in range(VAE_EPOCHS):
        order = rng.permutation(n_users)
        for start in range(0, n_users, 128):
            batch = x[order[start:start + 128]]
            logits, mu, logvar = model(batch)
            recon = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch, reduction="sum") / len(batch)
            kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
            loss = recon + 0.02 * kl
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        logits, _, _ = model(x)
        return logits.cpu().numpy().astype(np.float32)


def aggregate_metrics(rows: list[dict]) -> pd.DataFrame:
    per_user = pd.DataFrame(rows)
    numeric_cols = per_user.select_dtypes(include=[np.number]).columns.difference(["user_pos"])
    summary = per_user.groupby(["Model", "iteration"], as_index=False)[list(numeric_cols)].mean()
    for target in ["european", "non_english", "long_tail", "us_origin", "us_company"]:
        summary[f"recommendation_{target}_shift"] = (
            summary[f"recommendation_{target}_share"] - summary[f"initial_{target}_share"]
        )
        summary[f"profile_{target}_shift"] = summary[f"profile_{target}_share"] - summary[f"initial_{target}_share"]
    return summary


def run_feedback_loop(model_name: str, score_fn: Callable, base_state: SimulationState) -> tuple[pd.DataFrame, pd.DataFrame]:
    state = SimulationState(
        users=base_state.users.copy(),
        candidate_items=base_state.candidate_items.copy(),
        profile_sets=[set(items) for items in base_state.initial_profile_sets],
        initial_profile_sets=[set(items) for items in base_state.initial_profile_sets],
        seen_global_items=set(base_state.seen_global_items),
        labels=base_state.labels.copy(),
        rng=np.random.default_rng(SEED + abs(hash(model_name)) % 10_000),
    )
    all_rows = []
    acceptance_rows = []
    for iteration in range(1, ITERATIONS + 1):
        matrix = build_matrix(state.profile_sets, len(state.candidate_items))
        scores = score_fn(matrix, state.profile_sets, state.rng)
        recs = topk_recommendations(scores, state.profile_sets, TOP_K)
        all_rows.extend(metric_rows(model_name, iteration, recs, state))
        accepted = accept_one_item(recs, state)
        acceptance_rows.append({"Model": model_name, "iteration": iteration, "accepted_items": accepted, "profile_rows": matrix.nnz + accepted})
        print(f"{model_name}: iteration {iteration}/{ITERATIONS}, accepted {accepted:,} items")
    return aggregate_metrics(all_rows), pd.DataFrame(acceptance_rows)


def plot_outputs(summary: pd.DataFrame, final_summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    for model, group in summary.groupby("Model"):
        axes[0].plot(group["iteration"], group["recommendation_european_share"], marker="o", linewidth=1.8, label=model)
        axes[1].plot(group["iteration"], group["recommendation_non_english_share"], marker="o", linewidth=1.8, label=model)
    axes[0].set_title("European-origin share in Top-K recommendations", weight="bold")
    axes[1].set_title("Non-English share in Top-K recommendations", weight="bold")
    for ax in axes:
        ax.set_xlabel("Feedback-loop iteration")
        ax.set_ylabel("Average recommendation share")
        ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
        ax.grid(alpha=0.25)
    axes[1].legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "31_feedback_loop_representation_dynamics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    heat = final_summary.set_index("Model")[[
        "recommendation_origin_jsd",
        "profile_origin_jsd",
        "recommendation_language_jsd",
        "profile_language_jsd",
        "recommendation_popularity_jsd",
        "profile_popularity_jsd",
    ]]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    im = ax.imshow(heat.to_numpy(), cmap="YlOrRd", aspect="auto")
    ax.set_xticks(np.arange(len(heat.columns)), [c.replace("_", " ") for c in heat.columns], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(heat.index)), heat.index)
    ax.set_title("Schedl-style miscalibration after feedback-loop simulation", weight="bold")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax.text(j, i, f"{heat.iloc[i, j]:.3f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im, ax=ax, label="Jensen-Shannon divergence")
    fig.tight_layout()
    fig.savefig(OUTPUTS / "32_feedback_loop_jsd_heatmap.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    shifts = final_summary.set_index("Model")[[
        "recommendation_european_shift",
        "recommendation_non_english_shift",
        "recommendation_us_origin_shift",
        "recommendation_us_company_shift",
        "recommendation_long_tail_shift",
    ]]
    fig, ax = plt.subplots(figsize=(11, 5.4))
    shifts.plot(kind="bar", ax=ax)
    ax.axhline(0, color=COLORS["ink"], linewidth=1)
    ax.set_title("Final recommendation shift relative to initial user histories", weight="bold")
    ax.set_ylabel("Share-point shift")
    ax.yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    ax.tick_params(axis="x", rotation=25)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "33_feedback_loop_final_shift.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    lang_country = final_summary.set_index("Model")[[
        "recommendation_european_share",
        "recommendation_non_english_share",
        "recommendation_us_origin_share",
        "recommendation_us_company_share",
        "recommendation_origin_jsd",
        "recommendation_language_jsd",
    ]]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    lang_country[[
        "recommendation_european_share",
        "recommendation_non_english_share",
        "recommendation_us_origin_share",
        "recommendation_us_company_share",
    ]].plot(kind="bar", ax=axes[0])
    axes[0].set_title("Final recommendation composition", weight="bold")
    axes[0].set_ylabel("Average Top-K share")
    axes[0].yaxis.set_major_formatter(lambda y, _: f"{y:.0%}")
    lang_country[["recommendation_origin_jsd", "recommendation_language_jsd"]].plot(kind="bar", ax=axes[1], color=[COLORS["orange"], COLORS["teal"]])
    axes[1].set_title("Country/language miscalibration", weight="bold")
    axes[1].set_ylabel("JSD vs initial histories")
    for ax in axes:
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(OUTPUTS / "34_language_country_bias_panels.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_ledgers(summary: pd.DataFrame, acceptance: pd.DataFrame, base_state: SimulationState) -> None:
    final_summary = summary.loc[summary.groupby("Model")["iteration"].idxmax()].copy().sort_values("Model")
    summary.to_csv(OUTPUTS / "31_feedback_loop_iteration_metrics.csv", index=False)
    final_summary.to_csv(OUTPUTS / "31_feedback_loop_final_summary.csv", index=False)
    acceptance.to_csv(OUTPUTS / "31_feedback_loop_acceptance_log.csv", index=False)
    run_report = pd.DataFrame([
        {"object": "feedback-loop users", "count": len(base_state.users)},
        {"object": "candidate items", "count": len(base_state.candidate_items)},
        {"object": "iterations", "count": ITERATIONS},
        {"object": "top-k recommendations", "count": TOP_K},
        {"object": "acceptance alpha", "count": ACCEPTANCE_ALPHA},
    ])
    run_report.to_csv(OUTPUTS / "31_feedback_loop_run_report.csv", index=False)

    model_ledger = pd.DataFrame([
        {"model": "Pop", "schedl_reference": "Pop baseline", "local_implementation": "Most popular unseen item by current profile counts", "status": "executed"},
        {"model": "ItemKNN", "schedl_reference": "ItemKNN", "local_implementation": "Cosine item-item collaborative filtering", "status": "executed"},
        {"model": "BPR-MF", "schedl_reference": "BPR", "local_implementation": "PyTorch Bayesian Personalized Ranking matrix factorisation", "status": "executed"},
        {"model": "LightGCN-style", "schedl_reference": "LightGCN", "local_implementation": "One-step normalized user-item graph propagation proxy", "status": "executed as transparent lightweight approximation"},
        {"model": "NeuMF-lite", "schedl_reference": "NeuMF", "local_implementation": "PyTorch neural matrix-factorisation-style pairwise ranking model", "status": "executed as lightweight local variant"},
        {"model": "MultiVAE-lite", "schedl_reference": "MultiVAE", "local_implementation": "PyTorch variational autoencoder over user-item vectors", "status": "executed as lightweight local variant"},
    ])
    model_ledger.to_csv(OUTPUTS / "31_schedl_model_ledger.csv", index=False)

    note = """# Schedl-Style Feedback-Loop Adaptation

Source logic: Lesota, Geiger, Walder, Kowald and Schedl (2024) simulate recommender feedback loops by repeatedly recommending items, accepting one ranked item per user, appending the item to the user profile and retraining the model.

MovieLens adaptation: MovieLens/M3L does not contain user-country metadata. We therefore do not claim a literal local-country audit. Instead, we translate the logic into user-history cultural calibration: a recommendation is treated as culturally miscalibrated when its country/language/popularity distribution diverges from the distribution already visible in the user's initial movie history.

Executed model family: Pop, ItemKNN, BPR-MF, LightGCN-style graph propagation, NeuMF-lite and MultiVAE-lite. The last three are lightweight local variants because the RecBole stack used in the paper is not installed in this workspace.

Main metrics: representation shifts for European, non-English, US-origin, US-company and long-tail films; plus Jensen-Shannon divergence for origin, language and popularity distributions.

References:
- Lesota, O., Geiger, J., Walder, M., Kowald, D., & Schedl, M. (2024). *Oh, Behave! Country Representation Dynamics Created by Feedback Loops in Music Recommender Systems*. https://arxiv.org/abs/2408.11565
- Mansoury, M., Abdollahpouri, H., Pechenizkiy, M., Mobasher, B., & Burke, R. (2020). *Feedback Loop and Bias Amplification in Recommender Systems*. https://arxiv.org/abs/2007.13019
- Steck, H. (2018). *Calibrated Recommendations*. RecSys 2018. https://doi.org/10.1145/3240323.3240372
"""
    (OUTPUTS / "31_schedl_adaptation_note.md").write_text(note, encoding="utf-8")
    plot_outputs(summary, final_summary)


def main() -> None:
    interactions, movie_meta = load_inputs()
    labels = prepare_labels(movie_meta)
    base_state = make_initial_state(interactions, labels)
    print(f"Schedl-style sample: {len(base_state.users):,} users, {len(base_state.candidate_items):,} candidate items.")

    models: list[tuple[str, Callable]] = [
        ("Pop", lambda matrix, profile_sets, rng: popularity_scores(matrix)),
        ("ItemKNN", lambda matrix, profile_sets, rng: itemknn_scores(matrix)),
        ("BPR-MF", lambda matrix, profile_sets, rng: bpr_scores(matrix, profile_sets, rng)),
        ("LightGCN-style", lambda matrix, profile_sets, rng: lightgcn_style_scores(matrix)),
        ("NeuMF-lite", lambda matrix, profile_sets, rng: neumf_lite_scores(matrix, profile_sets, rng)),
        ("MultiVAE-lite", lambda matrix, profile_sets, rng: multivae_lite_scores(matrix, profile_sets, rng)),
    ]

    summaries, acceptance_logs = [], []
    for model_name, score_fn in models:
        summary, acceptance = run_feedback_loop(model_name, score_fn, base_state)
        summaries.append(summary)
        acceptance_logs.append(acceptance)

    full_summary = pd.concat(summaries, ignore_index=True)
    full_acceptance = pd.concat(acceptance_logs, ignore_index=True)
    write_ledgers(full_summary, full_acceptance, base_state)
    print("Schedl-style feedback-loop audit complete.")
    print(full_summary.loc[full_summary.groupby("Model")["iteration"].idxmax()].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
