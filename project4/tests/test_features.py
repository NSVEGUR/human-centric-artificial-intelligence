import numpy as np
from django.test import SimpleTestCase

from project4 import data as p4data


class TestFeatureMatrix(SimpleTestCase):
    def test_shape_matches_catalog(self):
        self.assertEqual(p4data.FEATURE_MATRIX.shape[0], p4data.N_CATALOG)
        self.assertEqual(p4data.FEATURE_MATRIX.shape[1], len(p4data.FEATURE_NAMES))

    def test_no_nans(self):
        self.assertFalse(np.isnan(p4data.FEATURE_MATRIX).any())

    def test_genre_features_are_binary(self):
        genre_cols = [i for i, n in enumerate(p4data.FEATURE_NAMES) if n.startswith("genre:")]
        block = p4data.FEATURE_MATRIX[:, genre_cols]
        self.assertTrue(np.isin(block, [0.0, 1.0]).all())

    def test_rating_one_hot_sums_to_one(self):
        rating_cols = [i for i, n in enumerate(p4data.FEATURE_NAMES) if n.startswith("rating:")]
        sums = p4data.FEATURE_MATRIX[:, rating_cols].sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(p4data.N_CATALOG))

    def test_catalog_has_no_duplicate_title_year(self):
        pairs = list(zip(p4data.CATALOG_DF["movie_title"], p4data.CATALOG_DF["title_year"]))
        self.assertEqual(len(pairs), len(set(pairs)))


class TestMovieRecords(SimpleTestCase):
    def test_get_movies_returns_requested_ids(self):
        movies = p4data.get_movies([0, 1, 2])
        self.assertEqual([m["id"] for m in movies], [0, 1, 2])

    def test_movie_record_has_display_fields(self):
        movie = p4data.get_movie(0)
        for key in ("id", "title", "year", "genres", "score", "rating", "keywords"):
            self.assertIn(key, movie)
