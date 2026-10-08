"""Streamlit app for the Product Recommendation System.

Loads the model bundle created by product_recommendation.py and serves:
  1. Cluster-based recommendations for a user (KMeans / MiniBatchKMeans)
  2. Item-item similar products (cosine similarity)
  3. Popularity-based recommendations
  4. Model insights (cluster sizes, rating distribution, model comparison)
"""

from pathlib import Path

import altair as alt  # installed together with Streamlit
import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).parent / "models" / "product_recommendation_models.joblib"
NEW_USER_LABEL = "New / Unknown User"

LABELS = {
    "productid": "Product ID",
    "similarity_score": "Cosine similarity",
    "average_rating": "Average rating",
    "rating_count": "Number of ratings",
    "cluster": "Cluster",
    "users": "Number of users",
    "rating": "Rating",
    "silhouette": "Silhouette score",
    "davies_bouldin": "Davies-Bouldin score",
}

st.set_page_config(page_title="Product Recommendation System", layout="wide")

RATING_COLUMN_CONFIG = {
    "average_rating": st.column_config.ProgressColumn(
        "Average rating", min_value=0, max_value=5, format="%.2f"
    ),
    "rating_count": st.column_config.NumberColumn("Number of ratings", format="%d"),
}

st.title("Product Recommendation System")
st.write("This app uses saved trained models for product recommendation.")

if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: {MODEL_PATH}. Run product_recommendation.py and "
        "place the generated .joblib file inside the 'models' folder."
    )
    st.stop()


@st.cache_resource(show_spinner="Loading models... (first load can take a minute)")
def load_bundle(path: str) -> dict:
    return joblib.load(path)


bundle = load_bundle(str(MODEL_PATH))

cosine_model = bundle["cosine_model"]
user_item_matrix = bundle["user_item_matrix"]
user_ids = bundle["user_ids"]
product_ids = bundle["product_ids"]
user_to_idx = bundle["user_to_idx"]
product_to_idx = bundle["product_to_idx"]
global_top_products = bundle["global_top_products"]
popular_products = bundle["popular_products"]

user_options = sorted(user_ids.tolist())
product_options = sorted(product_ids.tolist())

CLUSTER_ARTIFACTS = {
    "KMeans": (
        bundle["kmeans_user_cluster_map"],
        bundle["kmeans_cluster_top_products"],
    ),
    "MiniBatchKMeans": (
        bundle["minibatch_user_cluster_map"],
        bundle["minibatch_cluster_top_products"],
    ),
}


@st.cache_resource(show_spinner=False)
def build_product_stats(_matrix, _product_ids):
    """Rating count and average rating for every product in the training data."""
    counts = _matrix.getnnz(axis=0)
    totals = np.asarray(_matrix.sum(axis=0)).ravel()
    return pd.DataFrame(
        {"rating_count": counts, "average_rating": totals / counts},
        index=pd.Index(_product_ids, name="productid"),
    )


product_stats = build_product_stats(user_item_matrix, product_ids)


# ------------------------- Recommendation logic -------------------------
def get_seen_products(user_id):
    """Products this user has already rated, read from the sparse matrix."""
    user_index = user_to_idx.get(user_id)
    if user_index is None:
        return set()
    return set(product_ids[user_item_matrix[user_index].indices].tolist())


def recommend_for_user(user_id, model_name="KMeans", top_n=10):
    """Return (cluster, recommended product ids) for a user."""
    user_cluster_map, cluster_top_products = CLUSTER_ARTIFACTS[model_name]
    already_seen = get_seen_products(user_id)

    if user_id in user_cluster_map:
        cluster = user_cluster_map[user_id]
        candidates = cluster_top_products.get(cluster, [])
    else:
        cluster = NEW_USER_LABEL
        candidates = []

    recommendations = [p for p in candidates if p not in already_seen][:top_n]

    # Top up with globally popular products (also covers new users).
    if len(recommendations) < top_n:
        fallback = [
            p for p in global_top_products
            if p not in recommendations and p not in already_seen
        ]
        recommendations += fallback[: top_n - len(recommendations)]

    return cluster, recommendations


def recommend_similar_products(product_id, top_n=10):
    """Return the top_n products most similar to product_id (cosine similarity)."""
    if product_id not in product_to_idx:
        return pd.DataFrame(columns=["productid", "similarity_score"])

    product_index = product_to_idx[product_id]
    n_neighbors = min(top_n + 1, len(product_ids))

    # One product's rating vector across all users (1 x n_users).
    query = user_item_matrix[:, product_index].T.tocsr()
    distances, indices = cosine_model.kneighbors(query, n_neighbors=n_neighbors)
    indices = indices.flatten()
    scores = 1 - distances.flatten()

    keep = indices != product_index  # drop the query product itself
    indices, scores = indices[keep][:top_n], scores[keep][:top_n]

    return pd.DataFrame(
        {"productid": product_ids[indices], "similarity_score": scores.round(4)}
    )


def get_popular_products(top_n=10):
    return popular_products.head(top_n).reset_index(drop=True)


# ----------------------------- Chart helpers -----------------------------
def label(column):
    return LABELS.get(column, column)


def ranked_bar_chart(data, category, value, title, domain=None):
    """Horizontal bars sorted from the largest to the smallest value."""
    scale = alt.Scale(domain=domain) if domain else alt.Scale()
    chart = (
        alt.Chart(data, title=title)
        .mark_bar()
        .encode(
            x=alt.X(f"{value}:Q", scale=scale, title=label(value)),
            y=alt.Y(f"{category}:N", sort="-x", title=label(category)),
            tooltip=[category, value],
        )
        .properties(width="container", height=max(200, 26 * len(data)))
    )
    st.altair_chart(chart)


def column_chart(data, category, value, title):
    """Vertical bars in the order the rows are given."""
    chart = (
        alt.Chart(data, title=title)
        .mark_bar()
        .encode(
            x=alt.X(f"{category}:N", sort=None, title=label(category)),
            y=alt.Y(f"{value}:Q", title=label(value)),
            tooltip=[category, value],
        )
        .properties(width="container", height=300)
    )
    st.altair_chart(chart)


def cluster_size_chart(user_cluster_map, title, highlight=None):
    """Users per cluster; the `highlight` cluster is drawn in orange."""
    sizes = pd.Series(list(user_cluster_map.values())).value_counts().sort_index()
    data = pd.DataFrame({"cluster": sizes.index.astype(str), "users": sizes.values})

    colour = alt.condition(
        alt.datum.cluster == str(highlight),
        alt.value("#e4572e"),
        alt.value("#4c78a8"),
    )
    chart = (
        alt.Chart(data, title=title)
        .mark_bar()
        .encode(
            x=alt.X("cluster:N", sort=None, title=label("cluster")),
            y=alt.Y("users:Q", title=label("users")),
            color=colour,
            tooltip=["cluster", "users"],
        )
        .properties(width="container", height=300)
    )
    st.altair_chart(chart)


# ----------------------------- Sidebar -----------------------------
st.sidebar.header("Settings")

model_choice = st.sidebar.selectbox("Choose clustering model", list(CLUSTER_ARTIFACTS))
top_n = st.sidebar.slider("Number of recommendations", min_value=5, max_value=20, value=10)

# ------------------------------ Tabs ------------------------------
tab1, tab2, tab3, tab4 = st.tabs(
    ["User Recommendations", "Similar Products", "Popular Products", "Model Insights"]
)

with tab1:
    st.subheader("User-Based Product Recommendations")

    selected_user = st.selectbox("Select User ID", user_options, key="user_select")

    if st.button("Recommend Products for User"):
        cluster, recommendations = recommend_for_user(
            selected_user, model_name=model_choice, top_n=top_n
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Selected model", model_choice)
        m2.metric("User cluster", str(cluster))
        m3.metric("Products already rated", len(get_seen_products(selected_user)))

        recommendation_df = (
            product_stats.reindex(recommendations)
            .rename_axis("recommended_productid")
            .reset_index()
        )
        recommendation_df.index = range(1, len(recommendation_df) + 1)
        st.dataframe(recommendation_df, column_config=RATING_COLUMN_CONFIG)

        cluster_size_chart(
            CLUSTER_ARTIFACTS[model_choice][0],
            title=f"Users per cluster - {model_choice} (this user's cluster in orange)",
            highlight=cluster,
        )

with tab2:
    st.subheader("Cosine Similarity Product Recommendations")

    selected_product = st.selectbox("Select Product ID", product_options, key="product_select")

    if st.button("Find Similar Products"):
        similar_products = recommend_similar_products(selected_product, top_n=top_n)

        if similar_products.empty:
            st.warning("No similar products found for this product.")
        else:
            chart_col, table_col = st.columns([2, 1])
            with chart_col:
                ranked_bar_chart(
                    similar_products,
                    "productid",
                    "similarity_score",
                    f"Products most similar to {selected_product}",
                    domain=[0, 1],
                )
            with table_col:
                st.dataframe(
                    similar_products,
                    hide_index=True,
                    column_config={
                        "similarity_score": st.column_config.ProgressColumn(
                            "Cosine similarity", min_value=0, max_value=1, format="%.4f"
                        )
                    },
                )

with tab3:
    st.subheader("Popularity-Based Recommendations")

    top_popular = get_popular_products(top_n)

    left, right = st.columns(2)
    with left:
        ranked_bar_chart(top_popular, "productid", "rating_count", "Number of ratings")
    with right:
        ranked_bar_chart(
            top_popular, "productid", "average_rating", "Average rating", domain=[0, 5]
        )

    st.dataframe(top_popular, hide_index=True, column_config=RATING_COLUMN_CONFIG)

with tab4:
    st.subheader("Model Insights")

    st.markdown("#### Users per cluster")
    cluster_size_chart(
        CLUSTER_ARTIFACTS[model_choice][0], title=f"Users per cluster - {model_choice}"
    )

    st.markdown("#### Rating distribution (filtered training data)")
    rating_counts = pd.Series(user_item_matrix.data).value_counts().sort_index()
    rating_df = pd.DataFrame(
        {
            "rating": [f"{r:g}" for r in rating_counts.index],
            "rating_count": rating_counts.values,
        }
    )
    column_chart(rating_df, "rating", "rating_count", "Rating distribution")

    st.markdown("#### Clustering model comparison")
    comparison = bundle.get("clustering_comparison")
    if comparison is None:
        st.info(
            "Model comparison is not in the saved bundle yet. Add "
            "'clustering_comparison' to the bundle in product_recommendation.py, "
            "re-run it and replace the .joblib file."
        )
    else:
        st.dataframe(comparison.round(4), hide_index=True)

        scores = comparison.rename(
            columns={
                "Silhouette (higher=better)": "silhouette",
                "Davies-Bouldin (lower=better)": "davies_bouldin",
            }
        )
        c1, c2 = st.columns(2)
        with c1:
            column_chart(scores, "Model", "silhouette", "Silhouette score (higher is better)")
        with c2:
            column_chart(
                scores, "Model", "davies_bouldin", "Davies-Bouldin score (lower is better)"
            )
