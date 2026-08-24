"""
Loads the IMDB 5000 Movie Dataset and builds the feature matrix once, at
Django import time (same pattern as project2/project3: no per-request
retraining or reprocessing).
"""

import os
import pandas as pd

from .features import build_features

CSV_PATH = os.path.join(os.path.dirname(__file__), "data", "movie_metadata.csv")

_raw = pd.read_csv(CSV_PATH)
FEATURE_MATRIX, FEATURE_NAMES, CATALOG_DF, SCALER_STATS = build_features(_raw)
N_CATALOG = len(CATALOG_DF)


def _short_keywords(raw, limit=3):
    if pd.isna(raw):
        return []
    return [k.replace("-", " ") for k in str(raw).split("|")[:limit]]


def _movie_record(i: int) -> dict:
    row = CATALOG_DF.iloc[i]
    score = row.get("imdb_score")
    year = row.get("title_year")
    return {
        "id": int(i),
        "title": row["movie_title"],
        "year": int(year) if pd.notna(year) else None,
        "genres": [g for g in str(row["genres"]).split("|") if g][:4],
        "score": round(float(score), 1) if pd.notna(score) else None,
        "rating": row.get("content_rating") if pd.notna(row.get("content_rating")) else "Unrated",
        "keywords": _short_keywords(row.get("plot_keywords")),
        "director": row.get("director_name") if pd.notna(row.get("director_name")) else None,
    }


# Precompute display records once (5000 tiny dicts, cheap).
MOVIES = [_movie_record(i) for i in range(N_CATALOG)]


def get_movie(i: int) -> dict:
    return MOVIES[i]


def get_movies(ids: list[int]) -> list[dict]:
    return [MOVIES[i] for i in ids]
