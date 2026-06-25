from pathlib import Path

import nbformat as nbf


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK_PATH = PROJECT_ROOT / "notebooks" / "does_algorithm_hide_europe_realdata.ipynb"


def md(source: str):
    return nbf.v4.new_markdown_cell(source.strip())


def code(source: str):
    return nbf.v4.new_code_cell(source.strip())


cells = []

cells.append(md("""
# Does the Algorithm Hide Europe?
## A Multimodal Audit of Cultural Prominence Bias in Movie Recommender Systems

**Team:** Max Priessnitz & Nico [surname]  
**Course:** Data Science and Artificial Intelligence II: Data and Algorithmic Governance  
**Institution:** WU Vienna, 2026  
**Proposal notebook:** Research proposal plus executable real-data audit skeleton  

**Core idea:** Streaming catalogues can contain culturally diverse films while recommendation rankings still concentrate user attention on mainstream content. This project audits cultural prominence at the ranking level.
"""))

cells.append(code("""
from pathlib import Path
import os
import zipfile
import json
import math
import time
import textwrap
import platform
from collections import defaultdict, Counter
from importlib import metadata

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy import sparse
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

import requests
from tqdm.auto import tqdm
from IPython.display import display, Markdown

plt.style.use("seaborn-v0_8-whitegrid")

PROJECT_ROOT = Path(".").resolve()
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_INTERIM = PROJECT_ROOT / "data" / "interim"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
OUTPUTS = PROJECT_ROOT / "outputs"

for p in [DATA_RAW, DATA_INTERIM, DATA_PROCESSED, OUTPUTS]:
    p.mkdir(parents=True, exist_ok=True)

SEED = 42
TOP_K = 20
CANDIDATE_K = 100
MAX_USERS_FOR_PROPOSAL_RUN = 500
MAX_ITEMS_FOR_PROPOSAL_RUN = 3000
POPULAR_ITEMS_FOR_CANDIDATES = 1500
MAX_WIKIDATA_IDS_FOR_PROPOSAL_RUN = 3000
RUN_FULL = False
NO_SYNTHETIC_DATA = True

rng = np.random.default_rng(SEED)

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

print(f"Project root: {PROJECT_ROOT}")
"""))

cells.append(md("""
## Required Files Check

The notebook uses real local data only. It accepts the canonical `data/raw` layout and also detects the folder layout currently used in this project directory. If the required raw files are not available, it stops and prints exact download commands.
"""))

cells.append(code("""
required_files = {
    "MMRec_M3L-20M.zip": "M3L-20M interaction data in MMRec format",
    "TEXT_mpnet.zip": "MPNet text embeddings for movie plots",
    "IMG_clip-image.zip": "CLIP image embeddings for posters",
    "ml-20m.zip": "MovieLens 20M bridge files: movies.csv and links.csv",
}

known_alternatives = {
    "MMRec_M3L-20M.zip": [DATA_RAW / "MMRec_M3L-20M.zip", PROJECT_ROOT / "old" / "MMRec_M3L-20M.zip", PROJECT_ROOT / "m3l-20m"],
    "TEXT_mpnet.zip": [DATA_RAW / "TEXT_mpnet.zip", PROJECT_ROOT / "old" / "TEXT_mpnet.zip", PROJECT_ROOT / "TEXT_mpnet"],
    "IMG_clip-image.zip": [DATA_RAW / "IMG_clip-image.zip", PROJECT_ROOT / "old" / "IMG_clip-image.zip", PROJECT_ROOT / "IMG_clip-image"],
    "ml-20m.zip": [DATA_RAW / "ml-20m.zip", PROJECT_ROOT / "old" / "MovieLens 20M Dataset.zip", PROJECT_ROOT / "MovieLens 20M Dataset"],
}

def first_existing(paths):
    for path in paths:
        if path.exists():
            return path
    return None

resolved_inputs = {name: first_existing(paths) for name, paths in known_alternatives.items()}

rows = []
for fname, desc in required_files.items():
    canonical = DATA_RAW / fname
    found = resolved_inputs[fname]
    rows.append({
        "file": fname,
        "purpose": desc,
        "canonical_path": str(canonical),
        "found_path": str(found) if found else None,
        "exists": found is not None,
        "size_mb": round(found.stat().st_size / 1024 / 1024, 2) if found and found.is_file() else None,
        "input_type": "directory" if found and found.is_dir() else ("zip" if found else None),
    })

file_check = pd.DataFrame(rows)
display(file_check)

if not file_check["exists"].all():
    print("Missing required files. Download them with:")
    print(r'''
mkdir -p data/raw

curl -L -o data/raw/MMRec_M3L-20M.zip \\
"https://zenodo.org/records/18499145/files/MMRec_M3L-20M.zip?download=1"

curl -L -o data/raw/TEXT_mpnet.zip \\
"https://zenodo.org/records/18499145/files/TEXT_mpnet.zip?download=1"

curl -L -o data/raw/IMG_clip-image.zip \\
"https://zenodo.org/records/18499145/files/IMG_clip-image.zip?download=1"

curl -L -o data/raw/ml-20m.zip \\
"https://files.grouplens.org/datasets/movielens/ml-20m.zip"
''')
    raise FileNotFoundError("Required real data files are missing. No synthetic fallback is allowed.")
"""))

cells.append(md("""
## 1. Title

**Does the Algorithm Hide Europe? A Multimodal Audit of Cultural Prominence Bias in Movie Recommender Systems**

This title signals both the empirical domain and the governance angle: the project audits whether recommendation rankings reduce the practical visibility of European, non-English and long-tail films.
"""))

cells.append(md("""
## 2. Background and Rationale

Video-on-demand platforms do not only provide catalogues; they organise attention. Even if a catalogue contains European, non-English or long-tail films, recommender systems decide whether these works become visible in practice.

This creates a governance-relevant problem: catalogue diversity may look compliant, while ranking-level prominence may still concentrate attention on US, English-language and already popular content.

The project is relevant from three perspectives:

- **Societal perspective:** cultural diversity, media pluralism and access to local or non-mainstream works.
- **Industrial perspective:** platforms must balance relevance, engagement, catalogue value and governance expectations.
- **Algorithmic governance perspective:** recommender systems should be auditable not only for accuracy, but also for exposure and prominence effects.

The EU Audiovisual Media Services Directive context is relevant because EU coordination explicitly includes cultural diversity and the promotion and distribution of European works.
"""))

cells.append(md("""
## Research Gap Visual

The project builds on mature popularity-bias research but shifts the object of analysis from generic item popularity to cultural prominence in multimodal movie recommendation.
"""))

cells.append(code("""
gap_rows = [
    "Popularity bias",
    "Provider fairness",
    "Multimodal recommendation",
    "European/cultural media governance",
    "Ranking-level prominence audit",
]
gap_cols = ["Well studied", "Partly studied", "Our contribution"]
gap_matrix = np.array([
    [1.00, 0.55, 0.20],
    [0.80, 0.65, 0.35],
    [0.65, 0.75, 0.45],
    [0.35, 0.70, 0.55],
    [0.20, 0.55, 1.00],
])

fig, ax = plt.subplots(figsize=(9, 3.8))
im = ax.imshow(gap_matrix, cmap="YlGnBu", vmin=0, vmax=1)
ax.set_xticks(range(len(gap_cols)), gap_cols)
ax.set_yticks(range(len(gap_rows)), gap_rows)
ax.set_title("Research gap: from popularity bias to cultural prominence audit", pad=14, weight="bold")
for i, row in enumerate(gap_rows):
    for j, col in enumerate(gap_cols):
        label = ""
        if j == 2 and i == 4:
            label = "Core"
        elif j == 2 and i in [2, 3]:
            label = "Bridge"
        elif j == 0 and i == 0:
            label = "Strong"
        ax.text(j, i, label, ha="center", va="center", color=COLORS["ink"], fontsize=10, weight="bold")
ax.tick_params(axis="both", length=0)
fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02, label="Relative literature maturity / project emphasis")
fig.tight_layout()
fig.savefig(OUTPUTS / "00_research_gap_matrix.png", dpi=200, bbox_inches="tight")
plt.show()
"""))

cells.append(md("""
## 3. State of the Art

Existing recommender-system research has already shown that recommendation algorithms can amplify popularity bias: popular items receive disproportionate exposure, while long-tail items remain underrepresented. Klimashevskaia et al. (2024) review causes, measurement strategies and mitigation approaches for popularity bias in recommender systems.

Movie-domain studies and multistakeholder recommender research also show that different algorithms can create different exposure patterns, especially for niche users and long-tail items. Re-ranking is a common mitigation approach because it can be applied after any base recommender.

However, less work operationalises cultural diversity as a governance-oriented ranking visibility problem. Our project therefore does not only ask whether recommenders are accurate, but whether they make European, non-English and long-tail movies visible in Top-K rankings.

The gap is not "popularity bias exists". The gap is **cultural prominence under multimodal recommender settings**, with explicit metrics that connect catalogue share, user preference, test relevance and rank-discounted exposure.
"""))

cells.append(md("""
## 4. Research Questions

### Main Research Question

**Do movie recommender systems trained on real interaction data under-expose European, non-English and long-tail films in Top-20 recommendations, and can multimodal features or governance-aware re-ranking reduce this cultural prominence gap without substantially harming recommendation utility?**

### Subquestions

**RQ1 - Detection:**  
How does the exposure of European, non-English and long-tail films in Top-20 recommendations compare to their catalogue share, interaction share and relevant test-item share?

**RQ2 - Model comparison:**  
Do text-aware and image-aware recommendation models produce different cultural prominence patterns than classical collaborative filtering baselines?

**RQ3 - Mitigation:**  
Can a post-processing re-ranking strategy improve cultural prominence while preserving acceptable NDCG@20 and Recall@20?
"""))

cells.append(md("""
## 5. Data Sources and Licence Ledger

The data pipeline is intentionally licence-aware. Raw MovieLens and M3L files are used locally but should not be redistributed in the submission package.
"""))

cells.append(code("""
data_sources = pd.DataFrame([
    {
        "source": "M3L-20M / Binge Watch",
        "role": "Main interaction dataset plus multimodal features",
        "files_used": "MMRec_M3L-20M.zip, TEXT_mpnet.zip, IMG_clip-image.zip",
        "licence": "Zenodo record: Creative Commons Attribution 4.0 International for dataset record",
        "redistribution": "Do not redistribute raw data in the submission package; cite source",
        "why_needed": "Real MovieLens-based interactions plus MPNet text and CLIP-image features",
    },
    {
        "source": "MovieLens 20M",
        "role": "Identifier bridge",
        "files_used": "movies.csv/movie.csv and links.csv/link.csv",
        "licence": "GroupLens research-use terms",
        "redistribution": "No redistribution without permission",
        "why_needed": "Provides MovieLens movieId, title, genres, imdbId and tmdbId",
    },
    {
        "source": "Wikidata",
        "role": "Cultural metadata",
        "files_used": "SPARQL results cached as wikidata_movie_metadata.csv",
        "licence": "CC0 / No rights reserved for main structured data",
        "redistribution": "Allowed, but cite data access and query date",
        "why_needed": "Country of origin, original language and release date",
    },
])
display(data_sources)
data_sources.to_csv(OUTPUTS / "data_source_ledger.csv", index=False)
"""))

cells.append(md("""
## 6. Data Loading

The next cells inspect the available files before making assumptions about internal paths. This is important because the current project directory contains both extracted folders and archived source files.
"""))

cells.append(code("""
def list_zip_contents(zip_path, max_rows=50):
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
    return pd.DataFrame({"path": names[:max_rows], "total_files_in_zip": len(names)})

for fname in required_files:
    path = DATA_RAW / fname
    found = resolved_inputs[fname]
    print(f"\\n--- {fname} ---")
    if path.exists() and path.is_file():
        display(list_zip_contents(path, max_rows=30))
    elif found and found.is_file() and found.suffix.lower() == ".zip":
        display(list_zip_contents(found, max_rows=30))
    elif found and found.is_dir():
        sample_paths = sorted(str(p.relative_to(found)) for p in found.rglob("*") if p.is_file())[:30]
        display(pd.DataFrame({"path": sample_paths, "total_files_in_folder": sum(1 for p in found.rglob('*') if p.is_file())}))
    else:
        print("Not available")
"""))

cells.append(code("""
def read_csv_from_zip_flexible(zip_path, candidate_names):
    with zipfile.ZipFile(zip_path) as z:
        names = z.namelist()
        normalized = {n.lower().replace("\\\\", "/"): n for n in names}
        for wanted in candidate_names:
            wanted_norm = wanted.lower().replace("\\\\", "/")
            if wanted_norm in normalized:
                with z.open(normalized[wanted_norm]) as f:
                    return pd.read_csv(f)
        for n in names:
            lower = n.lower().replace("\\\\", "/")
            if any(lower.endswith("/" + wanted.lower()) or lower.endswith(wanted.lower()) for wanted in candidate_names):
                with z.open(n) as f:
                    return pd.read_csv(f)
    raise FileNotFoundError(f"None of {candidate_names} found in {zip_path}")

def load_movielens_bridge():
    source = resolved_inputs["ml-20m.zip"]
    if source.is_dir():
        movie_candidates = [source / "movies.csv", source / "movie.csv"]
        link_candidates = [source / "links.csv", source / "link.csv"]
        movie_path = first_existing(movie_candidates)
        link_path = first_existing(link_candidates)
        if movie_path is None or link_path is None:
            raise FileNotFoundError("MovieLens folder exists but movie/link CSV files were not found.")
        movies_df = pd.read_csv(movie_path)
        links_df = pd.read_csv(link_path)
    else:
        movies_df = read_csv_from_zip_flexible(source, ["ml-20m/movies.csv", "movies.csv", "movie.csv"])
        links_df = read_csv_from_zip_flexible(source, ["ml-20m/links.csv", "links.csv", "link.csv"])
    return movies_df, links_df

movies, links = load_movielens_bridge()

if "imdbId" in links.columns:
    links["imdb_id_str"] = links["imdbId"].apply(lambda x: f"tt{int(x):07d}" if pd.notna(x) else None)

display(movies.head())
display(links.head())
print(f"MovieLens bridge loaded: {len(movies):,} movies and {len(links):,} link rows.")
"""))

cells.append(code("""
def find_candidate_files(zip_path, keywords):
    with zipfile.ZipFile(zip_path, "r") as z:
        names = z.namelist()
    candidates = []
    for n in names:
        lower = n.lower()
        if any(k in lower for k in keywords):
            candidates.append(n)
    return candidates

def infer_sep_from_name(path_or_name):
    name = str(path_or_name).lower()
    if name.endswith(".tsv") or name.endswith(".inter") or name.endswith(".txt"):
        return "\\t"
    return ","

def standardise_interactions(df):
    rename = {}
    for col in df.columns:
        low = col.lower()
        if low in ["userid", "user_id", "user"]:
            rename[col] = "user_id"
        elif low in ["itemid", "movieid", "item_id", "item"]:
            rename[col] = "item_id"
        elif low in ["rating", "weight", "score"]:
            rename[col] = "rating_or_weight"
        elif low in ["x_label", "split", "label"]:
            rename[col] = "split"
    out = df.rename(columns=rename).copy()
    required = {"user_id", "item_id", "rating_or_weight", "split"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(f"Interaction file missing standard columns after rename: {missing}")
    if pd.api.types.is_numeric_dtype(out["split"]):
        out["split"] = out["split"].map({0: "train", 1: "valid", 2: "test"}).fillna(out["split"].astype(str))
    else:
        out["split"] = out["split"].astype(str).str.lower()
    return out[["user_id", "item_id", "rating_or_weight", "split"]]

def load_m3l_interactions():
    source = resolved_inputs["MMRec_M3L-20M.zip"]
    dtypes = {"userID": "int32", "itemID": "int32", "rating": "float32", "x_label": "int8"}
    if source.is_dir():
        inter_path = first_existing([source / "m3l-20m.inter", source / "ml20m.inter", source / "m3l-20m" / "m3l-20m.inter"])
        if inter_path is None:
            candidates = list(source.rglob("*.inter"))
            if candidates:
                inter_path = candidates[0]
        if inter_path is None:
            raise FileNotFoundError("No .inter file found in extracted M3L directory.")
        raw = pd.read_csv(inter_path, sep=infer_sep_from_name(inter_path), dtype=dtypes)
    else:
        candidates = find_candidate_files(source, ["m3l-20m.inter", "ml20m.inter", ".inter"])
        if not candidates:
            raise FileNotFoundError("No interaction file found in M3L zip.")
        inner = candidates[0]
        with zipfile.ZipFile(source) as z:
            with z.open(inner) as f:
                raw = pd.read_csv(f, sep=infer_sep_from_name(inner), dtype=dtypes)
    return standardise_interactions(raw)

interactions = load_m3l_interactions()

display(interactions.head())
display(interactions["split"].value_counts().rename_axis("split").reset_index(name="rows"))
display(interactions[["user_id", "item_id"]].nunique().to_frame("unique_count"))

assert interactions is not None and len(interactions) > 0, "No interaction data loaded. Do not create synthetic data."
"""))

cells.append(md("""
## 7. M3L Internal ID to MovieLens ID Mapping

The MMRec interaction file uses compact internal item IDs. The feature JSON files use original MovieLens IDs. To avoid an invalid join, we reconstruct the mapping by matching the MPNet `.npy` feature rows to the MPNet JSON vectors and cache the result.
"""))

cells.append(code("""
def resolve_feature_matrix(relative_path):
    extracted = PROJECT_ROOT / "m3l-20m" / relative_path
    if extracted.exists():
        return extracted
    source = resolved_inputs["MMRec_M3L-20M.zip"]
    if source.is_dir():
        candidate = source / relative_path
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Feature matrix not found: {relative_path}")

def resolve_feature_json_dir(dirname):
    extracted = PROJECT_ROOT / dirname
    if extracted.exists():
        return extracted
    return None

mpnet_matrix_path = resolve_feature_matrix(Path("text") / "mpnet.npy")
clip_matrix_path = resolve_feature_matrix(Path("image") / "clip_image.npy")
mpnet_json_dir = resolve_feature_json_dir("TEXT_mpnet")

print("MPNet matrix:", mpnet_matrix_path)
print("CLIP image matrix:", clip_matrix_path)
print("MPNet JSON directory:", mpnet_json_dir)

def vector_fingerprint(vec):
    arr = np.asarray(vec, dtype=np.float64)
    idx = [0, 1, 2, 3, 4, 5, 10, 50, 100, 200, 400, min(len(arr) - 1, 767)]
    return tuple(np.round(arr[idx], 8))

def build_internal_to_movielens_mapping(cache_path=DATA_INTERIM / "m3l_internal_to_movielens.csv"):
    if cache_path.exists():
        mapping = pd.read_csv(cache_path)
        return mapping
    if mpnet_json_dir is None:
        raise FileNotFoundError("TEXT_mpnet JSON directory is required to reconstruct the ID mapping.")
    matrix = np.load(mpnet_matrix_path, mmap_mode="r")
    lookup = {}
    collisions = defaultdict(list)
    json_files = list(mpnet_json_dir.glob("*.json"))
    for path in tqdm(json_files, desc="Indexing MPNet JSON vectors"):
        with open(path, "r") as f:
            obj = json.load(f)
        movie_id_str = next(iter(obj))
        fp = vector_fingerprint(obj[movie_id_str])
        if fp in lookup:
            collisions[fp].append(int(movie_id_str))
        else:
            lookup[fp] = int(movie_id_str)
    rows = []
    misses = []
    for internal_item_id in tqdm(range(matrix.shape[0]), desc="Matching M3L internal IDs"):
        fp = vector_fingerprint(matrix[internal_item_id])
        movie_id = lookup.get(fp)
        if movie_id is None:
            misses.append(internal_item_id)
        else:
            rows.append({"item_id": internal_item_id, "movieId": movie_id})
    mapping = pd.DataFrame(rows)
    if misses:
        raise ValueError(f"Could not map {len(misses)} internal item IDs. First misses: {misses[:10]}")
    mapping.to_csv(cache_path, index=False)
    return mapping

item_id_map = build_internal_to_movielens_mapping()
display(item_id_map.head(10))
print(f"Mapped {len(item_id_map):,} M3L internal item IDs to MovieLens movieId values.")
"""))

cells.append(md("""
## 8. Wikidata Metadata

Wikidata is queried by IMDb ID and cached locally. The proposal run queries a bounded, popularity-prioritised subset so that the notebook remains executable during the proposal phase. For the final empirical analysis, set `RUN_FULL = True` or increase `MAX_WIKIDATA_IDS_FOR_PROPOSAL_RUN`.
"""))

cells.append(code("""
interaction_cols = ["user_id", "item_id", "rating_or_weight"]
train_interactions = interactions.loc[interactions["split"].eq("train"), interaction_cols].copy()
valid_interactions = interactions.loc[interactions["split"].eq("valid"), interaction_cols].copy()
test_interactions = interactions.loc[interactions["split"].eq("test"), interaction_cols].copy()

item_counts = train_interactions.groupby("item_id").size().rename("train_interaction_count").reset_index()
item_priority = item_counts.merge(item_id_map, on="item_id", how="left").sort_values("train_interaction_count", ascending=False)

movie_meta = (
    item_id_map
    .merge(movies, on="movieId", how="left")
    .merge(links, on="movieId", how="left")
    .merge(item_counts, on="item_id", how="left")
)
movie_meta["train_interaction_count"] = movie_meta["train_interaction_count"].fillna(0)

display(movie_meta.head())
print(f"M3L catalogue items: {movie_meta['item_id'].nunique():,}")
print(f"Items with MovieLens title: {movie_meta['title'].notna().sum():,}")
print(f"Items with IMDb ID: {movie_meta['imdb_id_str'].notna().sum():,}")
"""))

cells.append(code("""
WIKIDATA_ENDPOINT = "https://query.wikidata.org/sparql"

def query_wikidata_by_imdb(imdb_ids, batch_size=100, sleep=0.25):
    all_rows = []
    imdb_ids = [x for x in imdb_ids if isinstance(x, str) and x.startswith("tt")]
    for i in tqdm(range(0, len(imdb_ids), batch_size), desc="Querying Wikidata"):
        batch = imdb_ids[i:i + batch_size]
        values = " ".join(f'"{x}"' for x in batch)
        query = f'''
        SELECT ?imdb ?film ?filmLabel ?countryLabel ?languageLabel ?publicationDate WHERE {{
          VALUES ?imdb {{ {values} }}
          ?film wdt:P345 ?imdb .
          OPTIONAL {{ ?film wdt:P495 ?country . }}
          OPTIONAL {{ ?film wdt:P364 ?language . }}
          OPTIONAL {{ ?film wdt:P577 ?publicationDate . }}
          SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
        }}
        '''
        try:
            r = requests.get(
                WIKIDATA_ENDPOINT,
                params={"query": query, "format": "json"},
                headers={"User-Agent": "WU-Data-Algorithmic-Governance-Student-Project/1.0"},
                timeout=60,
            )
            if r.status_code != 200:
                print("Wikidata error", r.status_code, r.text[:200])
                time.sleep(3)
                continue
            data = r.json()["results"]["bindings"]
        except Exception as exc:
            print(f"Wikidata request failed: {exc}")
            time.sleep(3)
            continue
        for row in data:
            all_rows.append({
                "imdb_id_str": row.get("imdb", {}).get("value"),
                "wikidata_uri": row.get("film", {}).get("value"),
                "title_wikidata": row.get("filmLabel", {}).get("value"),
                "country": row.get("countryLabel", {}).get("value"),
                "original_language": row.get("languageLabel", {}).get("value"),
                "publication_date": row.get("publicationDate", {}).get("value"),
            })
        time.sleep(sleep)
    return pd.DataFrame(all_rows)

wikidata_cache = DATA_INTERIM / "wikidata_movie_metadata.csv"

priority_movie_ids = item_priority["movieId"].dropna().astype(int).tolist()
priority_imdb_ids = (
    movie_meta[movie_meta["movieId"].isin(priority_movie_ids)]
    .dropna(subset=["imdb_id_str"])
    .drop_duplicates("imdb_id_str")
    .set_index("movieId")
    .reindex(priority_movie_ids)["imdb_id_str"]
    .dropna()
    .tolist()
)
if not RUN_FULL:
    priority_imdb_ids = priority_imdb_ids[:MAX_WIKIDATA_IDS_FOR_PROPOSAL_RUN]
else:
    priority_imdb_ids = movie_meta["imdb_id_str"].dropna().unique().tolist()

if wikidata_cache.exists():
    wikidata = pd.read_csv(wikidata_cache)
else:
    wikidata = pd.DataFrame(columns=["imdb_id_str", "wikidata_uri", "title_wikidata", "country", "original_language", "publication_date"])

cached_ids = set(wikidata["imdb_id_str"].dropna()) if len(wikidata) else set()
missing_imdb_ids = [x for x in priority_imdb_ids if x not in cached_ids]
print(f"Wikidata cache rows before query: {len(wikidata):,}")
print(f"IMDb IDs requested for this run: {len(priority_imdb_ids):,}")
print(f"IMDb IDs missing from cache: {len(missing_imdb_ids):,}")

if missing_imdb_ids:
    new_wikidata = query_wikidata_by_imdb(missing_imdb_ids)
    if len(new_wikidata):
        wikidata = pd.concat([wikidata, new_wikidata], ignore_index=True).drop_duplicates()
        wikidata.to_csv(wikidata_cache, index=False)
elif not wikidata_cache.exists():
    wikidata.to_csv(wikidata_cache, index=False)

display(wikidata.head())
print(f"Wikidata cache rows after query: {len(wikidata):,}")
"""))

cells.append(md("""
## 9. Build Audit Dataset

Cultural labels are deliberately transparent. A film is labelled European if at least one Wikidata country of origin is in the European country set below. A film is labelled non-English if Wikidata reports at least one original language and none of the reported original languages is English. Long-tail labels are based on the training interaction distribution.
"""))

cells.append(code("""
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

def sorted_unique(values):
    return sorted({x for x in values.dropna().astype(str) if x and x.lower() != "nan"})

def aggregate_metadata(wd):
    if wd is None or len(wd) == 0:
        return pd.DataFrame(columns=["imdb_id_str", "country", "original_language", "publication_date", "title_wikidata", "wikidata_uri"])
    return (
        wd.groupby("imdb_id_str")
        .agg({
            "country": sorted_unique,
            "original_language": sorted_unique,
            "publication_date": "first",
            "title_wikidata": "first",
            "wikidata_uri": "first",
        })
        .reset_index()
    )

wd_agg = aggregate_metadata(wikidata)

movie_meta = movie_meta.drop(columns=[c for c in ["country", "original_language", "publication_date", "title_wikidata", "wikidata_uri"] if c in movie_meta.columns], errors="ignore")
movie_meta = movie_meta.merge(wd_agg, on="imdb_id_str", how="left")

movie_meta["has_wikidata_match"] = movie_meta["wikidata_uri"].notna()
movie_meta["has_country"] = movie_meta["country"].apply(lambda xs: isinstance(xs, list) and len(xs) > 0)
movie_meta["has_language"] = movie_meta["original_language"].apply(lambda xs: isinstance(xs, list) and len(xs) > 0)

movie_meta["is_european"] = movie_meta["country"].apply(
    lambda xs: any(c in EUROPE_COUNTRIES for c in xs) if isinstance(xs, list) else False
)
movie_meta["is_non_english"] = movie_meta["original_language"].apply(
    lambda xs: (not any(lang in ENGLISH_LABELS for lang in xs)) if isinstance(xs, list) and len(xs) > 0 else False
)

q80 = movie_meta["train_interaction_count"].quantile(0.80)
q20 = movie_meta["train_interaction_count"].quantile(0.20)
movie_meta["is_blockbuster_head"] = movie_meta["train_interaction_count"] >= q80
movie_meta["is_long_tail"] = movie_meta["train_interaction_count"] <= q20

movie_meta.to_csv(DATA_PROCESSED / "movie_audit_metadata.csv", index=False)

display(movie_meta[["item_id", "movieId", "title", "country", "original_language", "is_european", "is_non_english", "is_long_tail", "train_interaction_count"]].head(10))
display(movie_meta[["has_wikidata_match", "has_country", "has_language", "is_european", "is_non_english", "is_long_tail"]].mean().to_frame("share").round(3))
"""))

cells.append(md("""
## 10. Proposal-Ready Data Visualisations

The following figures are designed for an 8-10 minute proposal presentation. They show scale, pipeline logic, metadata coverage and the first catalogue-versus-interaction audit view.
"""))

cells.append(code("""
def savefig(name):
    path = OUTPUTS / name
    plt.savefig(path, dpi=220, bbox_inches="tight")
    print(f"Saved {path}")

# Visual 1: dataset scale, using the Zenodo source metadata for M3L-20M.
scale = pd.DataFrame({
    "metric": ["Users", "Items", "Ratings"],
    "value": [138_493, 19_009, 18_777_965],
})

fig, ax = plt.subplots(figsize=(8, 4.2))
bars = ax.bar(scale["metric"], scale["value"], color=[COLORS["blue"], COLORS["green"], COLORS["gold"]])
ax.set_title("M3L-20M dataset scale", weight="bold")
ax.set_ylabel("Count")
ax.set_yscale("log")
for bar, value in zip(bars, scale["value"]):
    ax.text(bar.get_x() + bar.get_width() / 2, value * 1.12, f"{value:,}", ha="center", va="bottom", fontsize=10, weight="bold")
ax.text(0.5, -0.22, "Source metadata: Zenodo record for Binge Watch / M3L-20M. Sparsity: 99.29%.", transform=ax.transAxes, ha="center", fontsize=9, color=COLORS["gray"])
fig.tight_layout()
savefig("01_dataset_scale.png")
plt.show()
"""))

cells.append(code("""
# Visual 2: data pipeline diagram.
fig, ax = plt.subplots(figsize=(10, 5.2))
ax.axis("off")

boxes = [
    ("M3L-20M\\ninteractions", 0.12, 0.76),
    ("MPNet text\\nembeddings", 0.12, 0.58),
    ("CLIP poster\\nembeddings", 0.12, 0.40),
    ("MovieLens\\nlinks.csv", 0.12, 0.22),
    ("Wikidata\\ncultural metadata", 0.40, 0.22),
    ("Audit\\ndataset", 0.62, 0.49),
    ("Recommendation\\nmodels", 0.82, 0.62),
    ("Cultural prominence\\nmetrics", 0.82, 0.38),
    ("Re-ranking\\ntrade-off", 0.82, 0.16),
]

for text, x, y in boxes:
    ax.text(x, y, text, ha="center", va="center", fontsize=10, weight="bold",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#F7F9FB", edgecolor=COLORS["ink"], linewidth=1.2))

arrows = [
    ((0.23, 0.76), (0.52, 0.54)),
    ((0.23, 0.58), (0.52, 0.52)),
    ((0.23, 0.40), (0.52, 0.50)),
    ((0.23, 0.22), (0.31, 0.22)),
    ((0.49, 0.22), (0.58, 0.43)),
    ((0.68, 0.52), (0.76, 0.60)),
    ((0.68, 0.47), (0.76, 0.39)),
    ((0.82, 0.32), (0.82, 0.22)),
]
for start, end in arrows:
    ax.annotate("", xy=end, xytext=start, arrowprops=dict(arrowstyle="->", color=COLORS["gray"], lw=1.5))

ax.set_title("Audit pipeline: from data sources to governance-aware recommendation evaluation", weight="bold", pad=12)
fig.tight_layout()
savefig("02_data_pipeline.png")
plt.show()
"""))

cells.append(code("""
# Visual 3: metadata join funnel.
join_funnel = pd.DataFrame([
    {"stage": "M3L items", "items": movie_meta["item_id"].nunique()},
    {"stage": "Matched to MovieLens title", "items": movie_meta["title"].notna().sum()},
    {"stage": "Matched to IMDb ID", "items": movie_meta["imdb_id_str"].notna().sum()},
    {"stage": "Matched to Wikidata cache", "items": movie_meta["has_wikidata_match"].sum()},
    {"stage": "Has country", "items": movie_meta["has_country"].sum()},
    {"stage": "Has original language", "items": movie_meta["has_language"].sum()},
])

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.barh(join_funnel["stage"][::-1], join_funnel["items"][::-1], color=COLORS["teal"])
ax.set_title("Metadata join funnel for the proposal-stage audit dataset", weight="bold")
ax.set_xlabel("Number of M3L items")
for i, value in enumerate(join_funnel["items"][::-1]):
    ax.text(value + max(join_funnel["items"]) * 0.01, i, f"{int(value):,}", va="center", fontsize=9)
ax.text(0, -0.18, "Wikidata coverage reflects the current local cache and configured proposal query limit.", transform=ax.transAxes, fontsize=9, color=COLORS["gray"])
fig.tight_layout()
savefig("03_join_funnel.png")
plt.show()
"""))

cells.append(code("""
# Visual 4: long-tail distribution.
counts = movie_meta["train_interaction_count"].clip(lower=1)

fig, ax = plt.subplots(figsize=(8.5, 4.6))
ax.hist(counts, bins=np.logspace(np.log10(counts.min()), np.log10(counts.max()), 55), color=COLORS["blue"], alpha=0.82)
ax.set_xscale("log")
ax.set_yscale("log")
ax.axvline(q20 if q20 > 0 else 1, color=COLORS["red"], linestyle="--", label=f"20th percentile: {q20:.0f}")
ax.axvline(q80 if q80 > 0 else 1, color=COLORS["green"], linestyle="--", label=f"80th percentile: {q80:.0f}")
ax.set_title("Long-tail structure in the M3L-20M training interactions", weight="bold")
ax.set_xlabel("Item popularity: training interaction count (log scale)")
ax.set_ylabel("Number of movies (log scale)")
ax.legend()
fig.tight_layout()
savefig("04_popularity_long_tail.png")
plt.show()
"""))

cells.append(code("""
# Visual 5: catalogue share vs training interaction share.
train_with_meta = train_interactions.merge(movie_meta[["item_id", "is_european", "is_non_english", "is_long_tail", "is_blockbuster_head"]], on="item_id", how="left")

groups_for_share = {
    "European": "is_european",
    "Non-English": "is_non_english",
    "Long-tail": "is_long_tail",
    "Blockbuster-head": "is_blockbuster_head",
}

share_rows = []
for label, col in groups_for_share.items():
    share_rows.append({
        "group": label,
        "Catalogue share": movie_meta[col].mean(),
        "Training interaction share": train_with_meta[col].fillna(False).mean(),
    })
catalogue_interaction_share = pd.DataFrame(share_rows)
display(catalogue_interaction_share.round(3))

fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(catalogue_interaction_share))
width = 0.36
ax.bar(x - width / 2, catalogue_interaction_share["Catalogue share"], width, label="Catalogue share", color=COLORS["green"])
ax.bar(x + width / 2, catalogue_interaction_share["Training interaction share"], width, label="Training interaction share", color=COLORS["blue"])
ax.set_xticks(x, catalogue_interaction_share["group"])
ax.set_ylim(0, max(0.05, catalogue_interaction_share[["Catalogue share", "Training interaction share"]].to_numpy().max() * 1.25))
ax.set_ylabel("Share")
ax.set_title("Catalogue share vs interaction share", weight="bold")
ax.legend()
ax.text(0.5, -0.20, "European and non-English shares depend on current Wikidata cache coverage; long-tail/head are interaction-derived.", transform=ax.transAxes, ha="center", fontsize=9, color=COLORS["gray"])
fig.tight_layout()
savefig("05_catalogue_vs_interaction_share.png")
plt.show()
"""))

cells.append(code("""
# Visual 6: metadata and feature coverage.
mpnet_shape = np.load(mpnet_matrix_path, mmap_mode="r").shape
clip_shape = np.load(clip_matrix_path, mmap_mode="r").shape

coverage = pd.DataFrame([
    {"field": "Has IMDb ID", "share": movie_meta["imdb_id_str"].notna().mean()},
    {"field": "Has Wikidata match", "share": movie_meta["has_wikidata_match"].mean()},
    {"field": "Has country", "share": movie_meta["has_country"].mean()},
    {"field": "Has language", "share": movie_meta["has_language"].mean()},
    {"field": "Has text embedding", "share": mpnet_shape[0] / movie_meta["item_id"].nunique()},
    {"field": "Has image embedding", "share": clip_shape[0] / movie_meta["item_id"].nunique()},
])

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.bar(coverage["field"], coverage["share"], color=[COLORS["blue"], COLORS["teal"], COLORS["green"], COLORS["gold"], COLORS["violet"], COLORS["red"]])
ax.set_ylim(0, 1.05)
ax.set_ylabel("Share of M3L items")
ax.set_title("Metadata and multimodal feature coverage", weight="bold")
ax.tick_params(axis="x", rotation=25)
for i, value in enumerate(coverage["share"]):
    ax.text(i, min(1.02, value + 0.03), f"{value:.0%}", ha="center", fontsize=9, weight="bold")
fig.tight_layout()
savefig("06_metadata_feature_coverage.png")
plt.show()
"""))

cells.append(md("""
## 11. Recommendation Models

The proposal-stage run uses a deterministic user sample and a bounded candidate universe for speed. All rows still come from the real M3L/MovieLens data; no synthetic data is used. The final report can remove these bounds by increasing the configuration values at the top of the notebook.
"""))

cells.append(code("""
# Build proposal-stage users and candidate item universe.
train_user_counts = train_interactions.groupby("user_id").size()
test_user_counts = test_interactions.groupby("user_id").size()
eligible_users = sorted(set(train_user_counts[train_user_counts >= 5].index) & set(test_user_counts[test_user_counts >= 1].index))

sample_size = min(MAX_USERS_FOR_PROPOSAL_RUN, len(eligible_users))
sample_users = np.array(eligible_users)
rng.shuffle(sample_users)
sample_users = np.sort(sample_users[:sample_size])

sample_train = train_interactions.loc[np.isin(train_interactions["user_id"].to_numpy(), sample_users)].copy()
sample_test = test_interactions.loc[np.isin(test_interactions["user_id"].to_numpy(), sample_users)].copy()

top_pop_items = item_counts.sort_values("train_interaction_count", ascending=False)["item_id"].head(POPULAR_ITEMS_FOR_CANDIDATES).astype(int).tolist()
protected_items = set(sample_test["item_id"].astype(int)) | set(top_pop_items)
candidate_items = set(top_pop_items) | set(sample_train["item_id"].astype(int)) | set(sample_test["item_id"].astype(int))

if len(candidate_items) > MAX_ITEMS_FOR_PROPOSAL_RUN:
    count_lookup = item_counts.set_index("item_id")["train_interaction_count"].to_dict()
    protected_sorted = sorted(protected_items, key=lambda x: (-count_lookup.get(x, 0), x))
    remaining = sorted(candidate_items - protected_items, key=lambda x: (-count_lookup.get(x, 0), x))
    candidate_items = protected_sorted + remaining
    candidate_items = candidate_items[:max(MAX_ITEMS_FOR_PROPOSAL_RUN, len(protected_sorted))]
else:
    candidate_items = sorted(candidate_items)

candidate_items = np.array(sorted(candidate_items), dtype=np.int32)
sample_users = np.array(sample_users, dtype=np.int32)

user_to_idx = {u: i for i, u in enumerate(sample_users)}
item_to_idx = {it: j for j, it in enumerate(candidate_items)}

sample_train = sample_train[sample_train["item_id"].isin(item_to_idx)]
sample_test = sample_test[sample_test["item_id"].isin(item_to_idx)]

rows = sample_train["user_id"].map(user_to_idx).to_numpy()
cols = sample_train["item_id"].map(item_to_idx).to_numpy()
data = (sample_train["rating_or_weight"].to_numpy() > 0).astype(np.float32)
R_train = sparse.csr_matrix((data, (rows, cols)), shape=(len(sample_users), len(candidate_items)))

seen_by_user_idx = {}
for uid, group in sample_train.groupby("user_id"):
    seen_by_user_idx[int(uid)] = set(group["item_id"].map(item_to_idx).dropna().astype(int))

test_relevant_by_user = {}
for uid, group in sample_test[sample_test["rating_or_weight"] > 0].groupby("user_id"):
    items = set(group["item_id"].astype(int))
    if items:
        test_relevant_by_user[int(uid)] = items

candidate_meta = (
    movie_meta[["item_id", "is_european", "is_non_english", "is_long_tail", "is_blockbuster_head"]]
    .set_index("item_id")
    .reindex(candidate_items)
    .reset_index()
)
candidate_meta_indexed = candidate_meta.set_index("item_id")

run_report = pd.DataFrame([
    {"object": "sample users", "count": len(sample_users)},
    {"object": "candidate items", "count": len(candidate_items)},
    {"object": "sample train rows", "count": len(sample_train)},
    {"object": "sample test rows", "count": len(sample_test)},
    {"object": "users with relevant test items", "count": len(test_relevant_by_user)},
])
display(run_report)
"""))

cells.append(code("""
def mask_seen(scores):
    scores = np.asarray(scores, dtype=np.float32).copy()
    for uid in sample_users:
        ui = user_to_idx[int(uid)]
        seen = seen_by_user_idx.get(int(uid), set())
        if seen:
            scores[ui, list(seen)] = -np.inf
    return scores

def recommend_from_scores(scores, k=TOP_K):
    masked = mask_seen(scores)
    recs = {}
    rec_scores = {}
    for uid in sample_users:
        ui = user_to_idx[int(uid)]
        s = masked[ui]
        finite = np.isfinite(s)
        if finite.sum() == 0:
            recs[int(uid)] = []
            rec_scores[int(uid)] = []
            continue
        k_eff = min(k, int(finite.sum()))
        idx = np.argpartition(-s, k_eff - 1)[:k_eff]
        idx = idx[np.argsort(-s[idx])]
        items = [int(candidate_items[j]) for j in idx]
        recs[int(uid)] = items
        rec_scores[int(uid)] = [float(s[j]) for j in idx]
    return recs, rec_scores

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
        if std > 1e-8:
            out[i, mask] = (vals - vals.mean()) / std
        else:
            out[i, mask] = 0.0
        out[i, ~mask] = -np.inf
    return out

# Model A: popularity baseline.
candidate_pop = (
    item_counts.set_index("item_id")
    .reindex(candidate_items)["train_interaction_count"]
    .fillna(0)
    .to_numpy(dtype=np.float32)
)
popularity_scores = np.tile(np.log1p(candidate_pop), (len(sample_users), 1)).astype(np.float32)
popularity_recs, _ = recommend_from_scores(popularity_scores)

print("Popularity baseline ready.")
"""))

cells.append(code("""
# Model B: ItemKNN using cosine similarity on the sample user-item matrix.
item_user = R_train.T.astype(np.float32)
item_norms = np.sqrt(item_user.multiply(item_user).sum(axis=1)).A1
item_norms[item_norms == 0] = 1.0
item_user_norm = item_user.multiply(1 / item_norms[:, None])
item_similarity = (item_user_norm @ item_user_norm.T).toarray().astype(np.float32)
np.fill_diagonal(item_similarity, 0.0)

# Keep only the strongest neighbours per item to reduce noise.
knn_keep = min(50, item_similarity.shape[1])
if knn_keep < item_similarity.shape[1]:
    threshold_idx = np.argpartition(-item_similarity, knn_keep, axis=1)[:, knn_keep:]
    rows_idx = np.arange(item_similarity.shape[0])[:, None]
    item_similarity[rows_idx, threshold_idx] = 0.0

itemknn_scores = (R_train @ item_similarity).astype(np.float32)
itemknn_recs, _ = recommend_from_scores(itemknn_scores)

print("ItemKNN ready.")
"""))

cells.append(code("""
# Model C: TruncatedSVD collaborative filtering.
n_components = min(64, max(2, min(R_train.shape) - 1))
svd = TruncatedSVD(n_components=n_components, random_state=SEED)
user_factors = svd.fit_transform(R_train).astype(np.float32)
svd_scores = (user_factors @ svd.components_.astype(np.float32)).astype(np.float32)
svd_recs, _ = recommend_from_scores(svd_scores)

print(f"SVD ready with {n_components} components. Explained variance ratio: {svd.explained_variance_ratio_.sum():.3f}")
"""))

cells.append(code("""
# Models D and E: content-based MPNet text and CLIP image recommenders.
def content_scores_from_features(feature_path):
    features = np.load(feature_path, mmap_mode="r")[candidate_items].astype(np.float32)
    features = normalize(features, norm="l2", axis=1)
    profile_sum = R_train @ features
    counts_per_user = np.maximum(R_train.sum(axis=1).A1, 1.0).astype(np.float32)
    profiles = profile_sum / counts_per_user[:, None]
    profiles = normalize(profiles, norm="l2", axis=1)
    return (profiles @ features.T).astype(np.float32)

mpnet_scores = content_scores_from_features(mpnet_matrix_path)
clip_scores = content_scores_from_features(clip_matrix_path)

mpnet_recs, _ = recommend_from_scores(mpnet_scores)
clip_recs, _ = recommend_from_scores(clip_scores)

print("MPNet text and CLIP-image content recommenders ready.")
"""))

cells.append(code("""
# Model F: explicit hybrid.
hybrid_scores = (
    0.50 * rowwise_standardise(svd_scores)
    + 0.25 * rowwise_standardise(mpnet_scores)
    + 0.25 * rowwise_standardise(clip_scores)
).astype(np.float32)
hybrid_recs, _ = recommend_from_scores(hybrid_scores)

all_recs = {
    "Popularity": popularity_recs,
    "ItemKNN": itemknn_recs,
    "SVD": svd_recs,
    "MPNet-content": mpnet_recs,
    "CLIP-image-content": clip_recs,
    "Hybrid": hybrid_recs,
}

print("Hybrid recommender ready.")
"""))

cells.append(md("""
## 12. Metrics

We report utility metrics and cultural prominence metrics. The key audit idea is to distinguish catalogue availability, relevant test items, user history and rank-discounted recommendation exposure.
"""))

cells.append(code("""
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

group_columns = {
    "European": "is_european",
    "Non-English": "is_non_english",
    "Long-tail": "is_long_tail",
}

group_flags = {
    group: candidate_meta_indexed[col].fillna(False).astype(bool).to_dict()
    for group, col in group_columns.items()
}

train_history_groups = sample_train.merge(movie_meta[["item_id", *group_columns.values()]], on="item_id", how="left")
test_relevant_groups = sample_test[sample_test["rating_or_weight"] > 0].merge(movie_meta[["item_id", *group_columns.values()]], on="item_id", how="left")

history_group_share = {
    group: train_history_groups.groupby("user_id")[col].mean().fillna(0).to_dict()
    for group, col in group_columns.items()
}
relevant_group_share = {
    group: test_relevant_groups.groupby("user_id")[col].mean().fillna(0).to_dict()
    for group, col in group_columns.items()
}
user_group_targets = {
    int(uid): {
        group: max(float(history_group_share[group].get(int(uid), 0.0)), float(relevant_group_share[group].get(int(uid), 0.0)))
        for group in group_columns
    }
    for uid in sample_users
}

def discounted_group_exposure(items, group_name, k=TOP_K):
    if not items:
        return np.nan
    numerator = 0.0
    denominator = 0.0
    flags = group_flags[group_name]
    for rank, item in enumerate(items[:k], start=1):
        w = rank_discount(rank)
        denominator += w
        numerator += w * float(flags.get(item, False))
    return numerator / denominator if denominator > 0 else np.nan

def evaluate_recommendations(recs_dict, k=TOP_K):
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
            for group, col in group_columns.items():
                exposure = discounted_group_exposure(items, group, k)
                relevant_share = float(relevant_group_share[group].get(uid, 0.0))
                target = float(user_group_targets.get(uid, {}).get(group, 0.0))
                exposure_values[group].append(exposure)
                prominence_gaps[group].append(exposure - relevant_share)
                pacpg_values[group].append(exposure - target)
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

baseline_comparison = evaluate_recommendations(all_recs)
display(baseline_comparison.round(4))
"""))

cells.append(md("""
## 13. Governance-Aware Re-ranking

The mitigation is a transparent post-processing step. It starts from the Hybrid model's top-100 candidate list and adds a small cultural bonus when the current partial ranking is below the user's observed/relevant target for European, non-English or long-tail items.
"""))

cells.append(code("""
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
        if rel_std > 1e-8:
            rel_scores = (rel_scores - rel_scores.mean()) / rel_std
        else:
            rel_scores = rel_scores * 0
        rel_lookup = dict(zip(pool_items, rel_scores))
        selected = []
        target_by_group = user_group_targets.get(uid, {group: 0.0 for group in group_columns})
        for _ in range(k):
            best_item = None
            best_score = -np.inf
            current = selected if selected else []
            denom = len(current) + 1
            current_counts = {
                group: sum(bool(group_flags[group].get(x, False)) for x in current)
                for group in group_columns
            }
            for item in pool_items:
                if item in selected:
                    continue
                bonus = 0.0
                for group in group_columns:
                    target = float(target_by_group.get(group, 0.0))
                    flag = bool(group_flags[group].get(item, False))
                    current_share = (current_counts[group] + int(flag)) / denom
                    if current_share < target and group_flags[group].get(item, False):
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

lambdas = [0.0, 0.1, 0.3, 0.5, 0.7, 1.0]
rerank_recs = {f"Hybrid + reranking lambda={lam:.1f}": rerank_hybrid(lam) for lam in lambdas}

all_recs_with_rerank = dict(all_recs)
all_recs_with_rerank["Hybrid + reranking lambda=0.3"] = rerank_recs["Hybrid + reranking lambda=0.3"]
all_recs_with_rerank["Hybrid + reranking lambda=0.7"] = rerank_recs["Hybrid + reranking lambda=0.7"]

rerank_tradeoff = evaluate_recommendations(rerank_recs)
display(rerank_tradeoff.round(4))
"""))

cells.append(code("""
# Visual 7: utility vs prominence frontier.
base_hybrid = rerank_tradeoff[rerank_tradeoff["Model"].eq("Hybrid + reranking lambda=0.0")].iloc[0]
tradeoff = rerank_tradeoff.copy()
tradeoff["lambda"] = tradeoff["Model"].str.extract(r"lambda=([0-9.]+)").astype(float)
tradeoff["PACPG improvement"] = (
    abs(base_hybrid["PACPG European"]) + abs(base_hybrid["PACPG Non-English"]) + abs(base_hybrid["PACPG Long-tail"])
    - (
        tradeoff["PACPG European"].abs()
        + tradeoff["PACPG Non-English"].abs()
        + tradeoff["PACPG Long-tail"].abs()
    )
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(tradeoff[f"NDCG@{TOP_K}"], tradeoff["PACPG improvement"], s=90, color=COLORS["teal"])
for _, row in tradeoff.iterrows():
    ax.annotate(f"lambda={row['lambda']:.1f}", (row[f"NDCG@{TOP_K}"], row["PACPG improvement"]), textcoords="offset points", xytext=(6, 6), fontsize=9)
ax.axhline(0, color=COLORS["gray"], linewidth=1, linestyle="--")
ax.set_title("Utility vs cultural prominence frontier", weight="bold")
ax.set_xlabel(f"NDCG@{TOP_K}")
ax.set_ylabel("Reduction in absolute PACPG across audit groups")
fig.tight_layout()
savefig("07_utility_vs_prominence_frontier.png")
plt.show()
"""))

cells.append(md("""
## 14. Result Dashboard

These tables and plots are not final empirical findings. They are proposal-stage outputs from a bounded real-data run that verifies the audit architecture, metric implementation and re-ranking workflow.
"""))

cells.append(code("""
model_comparison = evaluate_recommendations(all_recs_with_rerank)
ordered_models = [
    "Popularity",
    "ItemKNN",
    "SVD",
    "MPNet-content",
    "CLIP-image-content",
    "Hybrid",
    "Hybrid + reranking lambda=0.3",
    "Hybrid + reranking lambda=0.7",
]
model_comparison["Model"] = pd.Categorical(model_comparison["Model"], categories=ordered_models, ordered=True)
model_comparison = model_comparison.sort_values("Model").reset_index(drop=True)
model_comparison.to_csv(OUTPUTS / "model_comparison.csv", index=False)
display(model_comparison.round(4))
"""))

cells.append(code("""
# Visual 8: group exposure by model.
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
ax.set_title("Group exposure by recommender model", weight="bold")
ax.legend()
fig.tight_layout()
savefig("08_group_exposure_by_model.png")
plt.show()
"""))

cells.append(code("""
# Visual 9: accuracy/fairness summary.
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
ax.set_title("Accuracy and cultural prominence summary", weight="bold")
fig.tight_layout()
savefig("09_accuracy_fairness_summary.png")
plt.show()
"""))

cells.append(md("""
## 15. Workplan

The workplan is aligned with the proposal presentation date and the final project deadline from the course slides.
"""))

cells.append(code("""
workplan = pd.DataFrame([
    {"week": "Week 1", "task": "Proposal, data download, source ledger, research question", "start": 1, "duration": 1},
    {"week": "Week 2", "task": "Data ingestion, Wikidata matching, EDA, metadata coverage", "start": 2, "duration": 1},
    {"week": "Week 3", "task": "Baseline recommenders: popularity, ItemKNN, SVD", "start": 3, "duration": 1},
    {"week": "Week 4", "task": "Multimodal models: MPNet, CLIP-image, hybrid", "start": 4, "duration": 1},
    {"week": "Week 5", "task": "Cultural prominence metrics, re-ranking mitigation, trade-off plots", "start": 5, "duration": 1},
    {"week": "Week 6", "task": "Final notebook, final presentation, limitations, governance discussion", "start": 6, "duration": 1},
])
display(workplan[["week", "task"]])

fig, ax = plt.subplots(figsize=(10, 4.8))
y = np.arange(len(workplan))
ax.barh(y, workplan["duration"], left=workplan["start"], color=[COLORS["blue"], COLORS["teal"], COLORS["green"], COLORS["gold"], COLORS["red"], COLORS["violet"]])
ax.set_yticks(y, workplan["week"])
ax.set_xticks(range(1, 8), [f"W{i}" for i in range(1, 8)])
ax.set_xlabel("Project timeline")
ax.set_title("Weekly workplan for the cultural prominence audit", weight="bold")
ax.invert_yaxis()
for i, row in workplan.iterrows():
    ax.text(row["start"] + 0.03, i, row["task"], va="center", fontsize=9, color="white", weight="bold")
ax.set_xlim(1, 7.2)
fig.tight_layout()
savefig("10_workplan_gantt.png")
plt.show()
"""))

cells.append(md("""
## Proposal Summary

**Problem:** VOD platforms may offer diverse catalogues while ranking systems still concentrate visibility on mainstream content.

**Research gap:** Popularity bias is well studied, but cultural prominence bias in multimodal movie recommenders is less directly operationalised.

**Data:** M3L-20M interactions and text/image features, MovieLens 20M bridge and Wikidata cultural metadata.

**Methods:** Popularity baseline, ItemKNN, SVD, MPNet content-based recommendation, CLIP-image content-based recommendation, hybrid scoring and governance-aware re-ranking.

**Metrics:** NDCG@20, Recall@20, MAP@20, catalogue coverage, discounted group exposure, Prominence Gap and Preference-Adjusted Cultural Prominence Gap.

**Contribution:** A reproducible audit pipeline for cultural visibility in recommender systems.
"""))

cells.append(md("""
## 16. References and Library Ledger

The ledger below documents the software dependencies used in this notebook. The reference list focuses on datasets, governance context and popularity-bias literature.
"""))

cells.append(code("""
def version_or_na(package_name):
    try:
        return metadata.version(package_name)
    except metadata.PackageNotFoundError:
        return "not installed"

library_ledger = pd.DataFrame([
    {"library": "Python", "version": platform.python_version(), "role": "Runtime", "licence": "PSF License", "citation / homepage": "https://www.python.org/"},
    {"library": "pandas", "version": version_or_na("pandas"), "role": "Tabular data loading and joins", "licence": "BSD-3-Clause", "citation / homepage": "https://pandas.pydata.org/"},
    {"library": "numpy", "version": version_or_na("numpy"), "role": "Array computation", "licence": "BSD-3-Clause", "citation / homepage": "https://numpy.org/"},
    {"library": "scipy", "version": version_or_na("scipy"), "role": "Sparse matrices", "licence": "BSD-3-Clause", "citation / homepage": "https://scipy.org/"},
    {"library": "scikit-learn", "version": version_or_na("scikit-learn"), "role": "SVD, normalization and similarity utilities", "licence": "BSD-3-Clause", "citation / homepage": "https://scikit-learn.org/"},
    {"library": "matplotlib", "version": version_or_na("matplotlib"), "role": "Visualisation", "licence": "PSF-compatible", "citation / homepage": "https://matplotlib.org/"},
    {"library": "requests", "version": version_or_na("requests"), "role": "Wikidata SPARQL access", "licence": "Apache-2.0", "citation / homepage": "https://requests.readthedocs.io/"},
    {"library": "tqdm", "version": version_or_na("tqdm"), "role": "Progress bars", "licence": "MPL-2.0 / MIT", "citation / homepage": "https://tqdm.github.io/"},
    {"library": "pyarrow", "version": version_or_na("pyarrow"), "role": "Optional columnar data support", "licence": "Apache-2.0", "citation / homepage": "https://arrow.apache.org/docs/python/"},
    {"library": "jupyter", "version": version_or_na("jupyter"), "role": "Notebook environment", "licence": "BSD-3-Clause", "citation / homepage": "https://jupyter.org/"},
])

library_ledger.to_csv(OUTPUTS / "library_ledger.csv", index=False)
display(library_ledger)
"""))

cells.append(md("""
## References

- Spillo, G., Petruzzelli, A., Musto, C., de Gemmis, M., Lops, P., & Semeraro, G. (2026). *Binge Watch: Reproducible Multimodal Benchmarks Datasets for Large-Scale Movie Recommendation on MovieLens-10M and 20M*. Zenodo. https://zenodo.org/records/18499145
- Spillo, G., Petruzzelli, A., Musto, C., de Gemmis, M., Lops, P., & Semeraro, G. (2026). *Binge Watch: Reproducible Multimodal Benchmarks Datasets for Large-Scale Movie Recommendation on MovieLens-10M and 20M*. arXiv:2602.15505.
- Harper, F. M., & Konstan, J. A. (2015). *The MovieLens Datasets: History and Context*. ACM Transactions on Interactive Intelligent Systems, 5(4), Article 19. https://doi.org/10.1145/2827872
- GroupLens. *MovieLens 20M Dataset README*. https://files.grouplens.org/datasets/movielens/ml-20m-README.html
- Wikidata. *Data access*. https://www.wikidata.org/wiki/Wikidata:Data_access
- European Commission. *Audiovisual Media Services Directive - AVMSD*. https://digital-strategy.ec.europa.eu/en/policies/audiovisual-and-media-services
- Klimashevskaia, A., Jannach, D., Elahi, M., & Trattner, C. (2024). *A Survey on Popularity Bias in Recommender Systems*. User Modeling and User-Adapted Interaction. https://doi.org/10.1007/s11257-024-09406-0
- Abdollahpouri, H., Mansoury, M., Burke, R., & Mobasher, B. (2019). *The Unfairness of Popularity Bias in Recommendation*. arXiv:1907.13286.
- Abdollahpouri, H., Burke, R., & Mobasher, B. (2019). *Managing Popularity Bias in Recommender Systems with Personalized Re-ranking*. FLAIRS 2019 / arXiv:1901.07555.
"""))

cells.append(md("""
## 17. Export Outputs

This final section writes an output manifest for the proposal package and performs sanity checks.
"""))

cells.append(code("""
outputs_manifest = [
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
    "data_source_ledger.csv",
    "library_ledger.csv",
    "model_comparison.csv",
]

readme = [
    "# Proposal Outputs",
    "",
    "Generated by `notebooks/does_algorithm_hide_europe_realdata.ipynb`.",
    "",
    "## Important scope note",
    "",
    "These outputs are from a proposal-stage real-data run. No synthetic data was used. Wikidata coverage depends on the local cache and the configured proposal query limit. Do not treat the bounded run as final empirical evidence until the full run is executed.",
    "",
    "## Files",
]
for name in outputs_manifest:
    path = OUTPUTS / name
    status = "present" if path.exists() else "missing"
    readme.append(f"- `{name}` - {status}")
readme.extend([
    "",
    "## Data governance",
    "",
    "Raw MovieLens and M3L files are not redistributed. Keep `data/raw/`, `old/`, `m3l-20m/`, `TEXT_mpnet/`, `IMG_clip-image/` and `MovieLens 20M Dataset/` out of any public submission archive unless the course explicitly permits local-only data sharing.",
])

(OUTPUTS / "README_outputs.md").write_text("\\n".join(readme), encoding="utf-8")
display(Markdown("\\n".join(readme)))
"""))

cells.append(code("""
assert NO_SYNTHETIC_DATA is True
assert resolved_inputs["MMRec_M3L-20M.zip"] is not None, "Missing real M3L interaction data."
assert resolved_inputs["ml-20m.zip"] is not None, "Missing MovieLens bridge."
assert "data_sources" in globals()
assert len(interactions) > 0
assert len(model_comparison) > 0

print("Notebook completed with real data only.")
print("No synthetic fallback was used.")
print(f"Output directory: {OUTPUTS}")
"""))


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3 (DS AI Proposal)",
        "language": "python",
        "name": "ds-ai-proposal",
    },
    "language_info": {
        "name": "python",
        "pygments_lexer": "ipython3",
    },
}

NOTEBOOK_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NOTEBOOK_PATH)
print(f"Wrote {NOTEBOOK_PATH}")
