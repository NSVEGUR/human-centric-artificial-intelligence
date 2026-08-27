"""
Task 1  - feature representation for the movie utility model U(x) = w^T x.

Design rationale (see the PDF report for the full write-up):

- Genres (multi-hot): the single strongest signal of taste in movie
  recommendation. We keep the 20 genres that occur at least 100 times in the
  ~5000-movie catalog; rarer tags (e.g. "Film-Noir", "Game-Show") are too
  sparse to ever be pinned down from a handful of interactions and would
  only add noise dimensions to w.
- Content rating (one-hot, collapsed to 5 buckets): a coarse but reliable
  proxy for tone/audience (family-friendly vs. mature).
- Continuous attributes (z-scored, log-transformed where heavy-tailed):
  imdb_score (quality), duration (pacing), recency (title_year), budget,
  gross, popularity (num_voted_users), and "star power" (director/cast
  Facebook likes). These are the standard covariates available in this
  metadata-only dataset (no user ratings), and they are exactly the kind of
  attributes a person can *see* on a movie's page before watching it, which
  matters because the elicitation interface has to show real, glanceable
  information for each item.

Dimensionality is kept moderate (~30 features) on purpose: w must be
estimable from a handful of pairwise/ranking interactions, so an
overly expressive feature space (e.g. one-hot actor/director identity)
would make the estimation problem hopeless in a short session.
"""

import numpy as np
import pandas as pd

# Only keep genres seen at least this many times in the catalog.
MIN_GENRE_COUNT = 100

RATING_GROUPS = {
    "G": "family", "TV-G": "family", "TV-Y": "family", "TV-Y7": "family",
    "PG": "pg", "TV-PG": "pg", "Approved": "pg", "Passed": "pg", "GP": "pg",
    "PG-13": "pg13", "TV-14": "pg13",
    "R": "mature", "TV-MA": "mature", "M": "mature", "X": "mature", "NC-17": "mature",
}
RATING_BUCKETS = ["family", "pg", "pg13", "mature", "unrated"]

NUMERIC_RAW = [
    "imdb_score", "duration", "title_year",
    "budget", "gross", "num_voted_users",
    "director_facebook_likes", "cast_total_facebook_likes", "movie_facebook_likes",
]
# these are heavy-tailed counts -> log1p before z-scoring
LOG_TRANSFORM = {
    "budget", "gross", "num_voted_users",
    "director_facebook_likes", "cast_total_facebook_likes", "movie_facebook_likes",
}


def _rating_bucket(raw):
    if pd.isna(raw):
        return "unrated"
    return RATING_GROUPS.get(str(raw).strip(), "unrated")


def clean_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate listings and rows with no genre / no title."""
    df = df.copy()
    df["movie_title"] = df["movie_title"].astype(str).str.replace("\xa0", "", regex=False).str.strip()
    df = df.dropna(subset=["movie_title", "genres"])
    df = df.drop_duplicates(subset=["movie_title", "title_year"], keep="first")
    df = df.reset_index(drop=True)
    return df


def genre_vocabulary(df: pd.DataFrame) -> list[str]:
    counts = {}
    for genres in df["genres"].dropna():
        for g in genres.split("|"):
            counts[g] = counts.get(g, 0) + 1
    return sorted([g for g, n in counts.items() if n >= MIN_GENRE_COUNT])


def build_features(df: pd.DataFrame):
    """
    Returns (X, feature_names, df) where X is an (n_movies, d) float32 array
    aligned row-for-row with df (index reset).
    """
    df = clean_catalog(df)
    genres = genre_vocabulary(df)

    # ── genre multi-hot ──────────────────────────────────────────────
    genre_matrix = np.zeros((len(df), len(genres)), dtype=np.float32)
    genre_index = {g: i for i, g in enumerate(genres)}
    for row, genre_str in enumerate(df["genres"].fillna("")):
        for g in genre_str.split("|"):
            if g in genre_index:
                genre_matrix[row, genre_index[g]] = 1.0

    # ── content rating one-hot ───────────────────────────────────────
    buckets = df["content_rating"].apply(_rating_bucket)
    rating_matrix = np.zeros((len(df), len(RATING_BUCKETS)), dtype=np.float32)
    bucket_index = {b: i for i, b in enumerate(RATING_BUCKETS)}
    for row, b in enumerate(buckets):
        rating_matrix[row, bucket_index[b]] = 1.0

    # ── numeric features: log-transform heavy-tailed ones, then z-score ──
    numeric_cols = []
    numeric_raw = np.zeros((len(df), len(NUMERIC_RAW)), dtype=np.float64)
    for j, col in enumerate(NUMERIC_RAW):
        vals = pd.to_numeric(df[col], errors="coerce")
        if col in LOG_TRANSFORM:
            vals = np.log1p(vals.clip(lower=0))
        median = vals.median()
        vals = vals.fillna(median)
        numeric_raw[:, j] = vals.to_numpy()
        numeric_cols.append(col)

    mean = numeric_raw.mean(axis=0)
    std = numeric_raw.std(axis=0)
    std[std == 0] = 1.0
    numeric_z = ((numeric_raw - mean) / std).astype(np.float32)

    X = np.concatenate([genre_matrix, rating_matrix, numeric_z], axis=1)
    feature_names = (
        [f"genre:{g}" for g in genres]
        + [f"rating:{b}" for b in RATING_BUCKETS]
        + [f"num:{c}" for c in numeric_cols]
    )

    scaler_stats = {"mean": mean, "std": std, "columns": numeric_cols}
    return X.astype(np.float32), feature_names, df, scaler_stats
