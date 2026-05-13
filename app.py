import os
import numpy as np
import pandas as pd
import streamlit as st

# CONFIG
st.set_page_config(page_title="Song Recommender", layout="wide")
st.title("Song Recommendation System")

CSV_PATH = "cleaned_data_with_cluster.csv"
SIM_PATH = "similarity_matrix.npy"
TOP_N_RECS = 20
MAX_SONGS_SHOWN = 2000  # random sample size shown in table

META_COLS = ["artist", "track", "album"]


# LOADERS
@st.cache_data(show_spinner=False)
def load_data():
    return pd.read_csv(CSV_PATH, low_memory=False)


@st.cache_resource(show_spinner=False)
def load_similarity_memmap():
    # memory-map so it doesn't load the whole NxN into RAM
    return np.load(SIM_PATH, mmap_mode="r")


@st.cache_resource(show_spinner=False)
def build_cluster_index(cluster_labels: np.ndarray):
    d = {}
    for i, c in enumerate(cluster_labels):
        d.setdefault(int(c), []).append(i)
    for k in d:
        d[k] = np.array(d[k], dtype=np.int32)
    return d


# FAST RECOMMENDER (optimized)
def recommend_fast_same_cluster(
    favorite_indices: np.ndarray,
    similarity_matrix,
    cluster_labels: np.ndarray,
    cluster_to_indices: dict,
    n_recommendations: int = TOP_N_RECS,
):
    if favorite_indices.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    fav_clusters = set(cluster_labels[favorite_indices].tolist())

    # Candidate pool = union of indices from those clusters
    candidates = []
    for c in fav_clusters:
        candidates.append(cluster_to_indices.get(int(c), np.array([], dtype=np.int32)))
    if not candidates:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    candidate_idx = np.unique(np.concatenate(candidates))
    if candidate_idx.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    # Remove favorites from candidates
    fav_set = set(favorite_indices.tolist())
    candidate_idx = np.array([i for i in candidate_idx if i not in fav_set], dtype=np.int32)
    if candidate_idx.size == 0:
        return np.array([], dtype=np.int32), np.array([], dtype=np.float32)

    # avg similarity over candidate columns only
    sim_sub = similarity_matrix[favorite_indices][:, candidate_idx]  
    avg_sim = sim_sub.mean(axis=0).astype(np.float32)                

    # Top-k
    k = min(n_recommendations, avg_sim.size)
    top_local = np.argpartition(-avg_sim, k - 1)[:k]
    top_local = top_local[np.argsort(-avg_sim[top_local])]

    top_indices = candidate_idx[top_local]
    top_scores = avg_sim[top_local]
    return top_indices, top_scores


# BOOTSTRAP
if not os.path.exists(CSV_PATH):
    st.error(f"Missing file: {CSV_PATH}")
    st.stop()

if not os.path.exists(SIM_PATH):
    st.error(f"Missing file: {SIM_PATH}")
    st.stop()

with st.spinner("Loading data..."):
    df = load_data()

if "cluster_kmeans" not in df.columns:
    st.error("Missing 'cluster_kmeans' column in CSV")
    st.stop()

for c in META_COLS:
    if c not in df.columns:
        st.error(f"Missing '{c}' column in CSV. Update META_COLS to match your file.")
        st.stop()

cluster_labels = df["cluster_kmeans"].to_numpy()

with st.spinner("Loading similarity matrix..."):
    similarity_matrix = load_similarity_memmap()

if len(df) != similarity_matrix.shape[0] or similarity_matrix.shape[0] != similarity_matrix.shape[1]:
    st.error(f"Mismatch: CSV rows = {len(df)}, similarity = {similarity_matrix.shape}")
    st.stop()

cluster_to_indices = build_cluster_index(cluster_labels)

# SESSION STATE
if "favorites" not in st.session_state:
    st.session_state.favorites = []  # list of indices
if "recommendations" not in st.session_state:
    st.session_state.recommendations = None
if "fav_key_version" not in st.session_state:
    st.session_state.fav_key_version = 0
if "sample_seed" not in st.session_state:
    st.session_state.sample_seed = 42


# SONG LIST
st.subheader("Song list")

search_query = st.text_input("Search by artist, track, or album", "")

if search_query.strip():
    mask = (
        df["artist"].astype(str).str.contains(search_query, case=False, na=False)
        | df["track"].astype(str).str.contains(search_query, case=False, na=False)
        | df["album"].astype(str).str.contains(search_query, case=False, na=False)
    )
    filtered_df = df.loc[mask].copy()
else:
    filtered_df = df.copy()

st.caption(f"Total matches: {len(filtered_df):,} out of {len(df):,}")

# RANDOM SAMPLE + CHECKBOX TABLE
cA, cB = st.columns([1, 1])
with cA:
    refresh_btn = st.button("Refresh random songs", use_container_width=True)
with cB:
    clear_btn = st.button("Clear favorites", use_container_width=True)

if refresh_btn:
    st.session_state.sample_seed += 1

if clear_btn:
    st.session_state.favorites = []
    st.session_state.recommendations = None
    st.session_state.fav_key_version += 1

# random sample from filtered results
if len(filtered_df) > MAX_SONGS_SHOWN:
    sampled_df = filtered_df.sample(n=MAX_SONGS_SHOWN, random_state=int(st.session_state.sample_seed))
else:
    sampled_df = filtered_df.copy()

show_df = sampled_df[["artist", "track", "album", "cluster_kmeans"]].copy()
show_df.rename(columns={"cluster_kmeans": "cluster"}, inplace=True)

# force string display
show_df["artist"] = show_df["artist"].astype(str)
show_df["track"] = show_df["track"].astype(str)
show_df["album"] = show_df["album"].astype(str)

# checkbox column
show_df.insert(0, "Select", show_df.index.isin(st.session_state.favorites))

edited_df = st.data_editor(
    show_df,
    use_container_width=True,
    height=520,
    key=f"song_table_{st.session_state.fav_key_version}",
    column_config={
        "Select": st.column_config.CheckboxColumn("Select", help="Tick to add/remove favorites")
    },
    disabled=["artist", "track", "album", "cluster"],
)

# update favorites based on visible rows only
selected_indices = edited_df.index[edited_df["Select"] == True].tolist()
unselected_indices = edited_df.index[edited_df["Select"] == False].tolist()

fav_set = set(st.session_state.favorites)
fav_set |= set(selected_indices)
fav_set -= set(unselected_indices)

st.session_state.favorites = sorted(list(fav_set))
st.session_state.recommendations = None

# FAVORITES & RECOMMENDATIONS (BELOW ROW)
st.divider()
left, right = st.columns(2)

with left:
    st.subheader("Favorite songs")
    if len(st.session_state.favorites) == 0:
        st.info("No favorites yet.")
    else:
        fav_rows = df.loc[st.session_state.favorites, ["artist", "track", "album", "cluster_kmeans"]].copy()
        fav_rows.rename(columns={"cluster_kmeans": "cluster"}, inplace=True)
        st.dataframe(fav_rows.reset_index(drop=True), use_container_width=True, height=360)

with right:
    st.subheader("Recommended songs")

    gen_btn = st.button("Generate recommendations", type="primary", use_container_width=True)

    if gen_btn:
        if len(st.session_state.favorites) == 0:
            st.warning("Add at least one favorite first.")
        else:
            fav_indices = np.array(st.session_state.favorites, dtype=np.int32)

            top_idx, top_scores = recommend_fast_same_cluster(
                favorite_indices=fav_indices,
                similarity_matrix=similarity_matrix,
                cluster_labels=cluster_labels,
                cluster_to_indices=cluster_to_indices,
                n_recommendations=TOP_N_RECS
            )

            rec_data = []
            for idx, score in zip(top_idx, top_scores):
                row = df.iloc[int(idx)]
                rec_data.append({
                    "Artist": row["artist"],
                    "Track": row["track"],
                    "Album": row["album"],
                    "Cluster": int(row["cluster_kmeans"]),
                    "Similarity": float(score),
                })

            st.session_state.recommendations = pd.DataFrame(rec_data)

    if st.session_state.recommendations is None:
        st.info("Click 'Generate recommendations' to see results.")
    else:
        st.dataframe(st.session_state.recommendations, use_container_width=True, height=360)
