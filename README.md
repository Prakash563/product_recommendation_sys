# Product Recommendation System

An e-commerce product recommendation system built from user ratings. It combines a popularity baseline, item-item collaborative filtering (cosine similarity) and user clustering (KMeans, MiniBatchKMeans, DBSCAN, Agglomerative), and serves the results through an interactive **Streamlit** app.

**Live app:** `<https://appuctrecommendationsys-4m4egrq4kfnca9gydv5fth.streamlit.app guys>`

<!-- Add screenshots here, for example:
![User recommendations](screenshots/user_recommendations.png)
-->

---

## Features

The Streamlit app has four tabs:

| Tab | What it does |
|---|---|
| **User Recommendations** | Select a user ID and a clustering model (KMeans or MiniBatchKMeans) to get recommended products, with each product's average rating and rating count. Shows the user's cluster and highlights it in a cluster-size chart. |
| **Similar Products** | Select a product ID to get the most similar products by cosine similarity, shown as a bar chart and a table. |
| **Popular Products** | Top products by average rating and number of ratings. |
| **Model Insights** | Users per cluster, rating distribution of the training data, and a comparison of the four clustering models. |

The sidebar controls which clustering model is used and how many recommendations are shown (5 to 20).

---

## Dataset

`ratings.csv` has no header row and four columns:

| Column | Description |
|---|---|
| `userId` | Unique user identifier |
| `productId` | Unique product identifier |
| `Rating` | Rating given by the user (1 to 5) |
| `timestamp` | Not used in this project |

After filtering to active users and products, the training data contains **157,174 users x 32,483 products** and **1,353,204 ratings**. Ratings are heavily skewed toward 5 stars.

---

## Approach

1. **Load and clean:** read IDs as text, drop `timestamp`, and remove duplicate (user, product) pairs.
2. **Exploratory analysis:** rating distribution, most-rated products and most active users, ratings per user and per product, and matrix sparsity.
3. **Filter:** keep users with at least 5 ratings and products with at least 10 ratings.
4. **Sparse user-item matrix:** stored as a CSR matrix (users x products).
5. **Popularity baseline:** products with at least 50 ratings, ranked by average rating and then rating count.
6. **Item-item collaborative filtering:** brute-force nearest neighbours with cosine distance on product rating vectors.
7. **User clustering:** `TruncatedSVD` (50 components) turns each user into an embedding. The elbow method guides the cluster count, and four algorithms are compared.
8. **Cluster-based recommendations:** each cluster gets a ranked product list (mean rating x log(1 + rating count)). Products the user has already rated are removed, and a global popularity list fills any gaps and covers new users.
9. **Save:** all models and lookup tables are stored in a single `joblib` bundle that the app loads.

### Clustering model comparison

| Model | Silhouette (higher is better) | Davies-Bouldin (lower is better) | Clusters | Unassigned users |
|---|---|---|---|---|
| KMeans | 0.5042 | 1.6606 | 4 | 0 |
| MiniBatchKMeans | 0.4660 | 3.8018 | 4 | 0 |
| DBSCAN | 0.0482 | 1.1462 | 499 | 55,625 |
| Agglomerative | 0.4514 | 0.8678 | 4 | 0 |

- **KMeans** has the best silhouette score on the full user base.
- **DBSCAN** fragments users into 499 small clusters and leaves about 35% of users unassigned. Its Davies-Bouldin score is computed on the clustered users only, so it is not directly comparable.
- **Agglomerative** is memory-heavy, so it was scored on a random sample of 3,000 users and is not strictly comparable with the others.

---

## Project structure

```
product-recommendation-system/
├── app.py                          # Streamlit app
├── product_recommendation.py       # Training pipeline: EDA, models, evaluation, saves the bundle
├── requirements.txt                # Dependencies for the Streamlit app
├── README.md
└── models/
    └── product_recommendation_models.joblib   # Saved models and lookup tables
```

---

## Run locally

```bash
git clone https://github.com/<your-username>/product-recommendation-system.git
cd product-recommendation-system

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

The app opens at `http://localhost:8501`.

## Retrain the models

1. Place `ratings.csv` in the project root (or edit `DATA_PATH` in `product_recommendation.py`; inside Google Colab the script reads it from Google Drive).
2. Install the extra plotting packages used by the training script:
   ```bash
   pip install matplotlib seaborn
   ```
3. Run the pipeline:
   ```bash
   python product_recommendation.py
   ```
4. The script writes `models/product_recommendation_models.joblib`. Commit the new file and redeploy.

---

## Deploy on Streamlit Community Cloud

1. Push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub.
3. Click **New app** and choose the repository and the `main` branch.
4. Set the main file path to `app.py` (Advanced settings lets you choose the Python version).
5. Click **Deploy**.

After replacing the model file in the repository, open **Manage app** and choose **Reboot app** so the app reloads the new file.

### `requirements.txt`

```
streamlit
pandas
numpy
scipy
scikit-learn
joblib
```

Pin `scikit-learn` and the other packages to the versions you trained with. A model saved with one scikit-learn version can fail to load on another.

---

## Notes and limitations

- Cluster-based recommendations are **partially personalized**: users in the same cluster share a ranked product list, minus the products each user has already rated.
- Users who were filtered out of the training data are treated as new users and receive globally popular products.
- Model quality is judged with clustering metrics (silhouette and Davies-Bouldin) and similarity scores. There is no hold-out evaluation of recommendation accuracy.

## Tech stack

Python, pandas, NumPy, SciPy, scikit-learn, joblib, Streamlit, Altair, Matplotlib and Seaborn (training only).

## Author

Prakash
