import numpy as np
from django.test import SimpleTestCase

from project4 import data
from project4.preference_model import DEFAULT_L2, fit_w
from project4.selection import (
    CANDIDATE_PAIRS, posterior_covariance, score_pairs,
    select_next_pair, select_random_pair,
)
from project4.simulation import (
    draw_participant, held_out_log_likelihood, _answer_ranking,
)


class PosteriorCovarianceTests(SimpleTestCase):
    def test_prior_only_is_isotropic(self):
        X = data.FEATURE_MATRIX.astype("float64")
        dim = X.shape[1]
        sigma = posterior_covariance(X, np.zeros(dim), [])
        expected = np.eye(dim) / (2.0 * DEFAULT_L2)
        self.assertTrue(np.allclose(sigma, expected))

    def test_observations_shrink_uncertainty(self):
        X = data.FEATURE_MATRIX.astype("float64")
        dim = X.shape[1]
        w = np.zeros(dim)
        before = np.trace(posterior_covariance(X, w, []))
        after = np.trace(posterior_covariance(X, w, [(0, 1), (2, 3), (4, 5)]))
        self.assertLess(after, before)

    def test_covariance_is_symmetric_positive_definite(self):
        X = data.FEATURE_MATRIX.astype("float64")
        sigma = posterior_covariance(X, np.zeros(X.shape[1]), [(10, 20), (30, 40)])
        self.assertTrue(np.allclose(sigma, sigma.T))
        self.assertTrue((np.linalg.eigvalsh(sigma) > 0).all())


class ScoringTests(SimpleTestCase):
    def test_identical_items_score_zero(self):

        X = data.FEATURE_MATRIX.astype("float64")
        sigma = posterior_covariance(X, np.zeros(X.shape[1]), [])
        same = np.array([[7, 7]])
        self.assertAlmostEqual(
            float(score_pairs(X, np.zeros(X.shape[1]), sigma, same)[0]), 0.0
        )

    def test_predictable_pair_scores_below_uncertain_pair(self):

        X = np.array([[1.0, 0.0], [-1.0, 0.0], [0.0, 1.0], [0.0, -1.0]])
        w = np.array([8.0, 0.0])          # strong preference on feature 0
        sigma = np.eye(2)
        obvious = score_pairs(X, w, sigma, np.array([[0, 1]]))[0]   # decided by w
        uncertain = score_pairs(X, w, sigma, np.array([[2, 3]]))[0]  # orthogonal to w
        self.assertLess(obvious, uncertain)


class SelectionTests(SimpleTestCase):
    def test_cold_start_returns_a_valid_pair(self):

        X = data.FEATURE_MATRIX.astype("float64")
        rng = np.random.default_rng(0)
        i, j = select_next_pair(X, np.zeros(X.shape[1]), [], rng)
        self.assertNotEqual(i, j)
        self.assertTrue(0 <= i < X.shape[0] and 0 <= j < X.shape[0])

    def test_excluded_items_are_never_selected(self):

        X = data.FEATURE_MATRIX.astype("float64")
        rng = np.random.default_rng(1)
        banned = set(range(0, 500))
        for _ in range(5):
            i, j = select_next_pair(X, np.zeros(X.shape[1]), [], rng, exclude=banned)
            self.assertNotIn(i, banned)
            self.assertNotIn(j, banned)

    def test_random_baseline_also_honours_exclusions(self):
        X = data.FEATURE_MATRIX.astype("float64")
        rng = np.random.default_rng(2)
        banned = set(range(0, 4000))
        i, j = select_random_pair(X, rng, exclude=banned)
        self.assertNotIn(i, banned)
        self.assertNotIn(j, banned)

    def test_adaptive_beats_random_on_synthetic_participants(self):

        X = data.FEATURE_MATRIX.astype("float64")
        dim = X.shape[1]
        rng = np.random.default_rng(4)
        from project4.simulation import learning_curve, _held_out_pairs

        gains = []
        for _ in range(3):
            w_true = draw_participant(dim, rng)
            held_out = _held_out_pairs(X, w_true, rng, n=30)
            rnd = learning_curve(X, w_true, "random", 6, rng, held_out)
            adp = learning_curve(X, w_true, "adaptive", 6, rng, held_out)
            gains.append(adp[-1] - rnd[-1])
        self.assertGreaterEqual(np.mean(gains), 0.0)


class LogLikelihoodDVTests(SimpleTestCase):
    def test_returns_none_without_observations(self):
        X = data.FEATURE_MATRIX.astype("float64")
        self.assertIsNone(held_out_log_likelihood(X, np.zeros(X.shape[1]), []))

    def test_chance_model_scores_log_half(self):

        X = data.FEATURE_MATRIX.astype("float64")
        ll = held_out_log_likelihood(X, np.zeros(X.shape[1]), [(1, 2), (3, 4)])
        self.assertAlmostEqual(ll, float(np.log(0.5)), places=6)

    def test_better_fit_scores_higher(self):
        X = data.FEATURE_MATRIX.astype("float64")
        dim = X.shape[1]
        rng = np.random.default_rng(5)
        w_true = draw_participant(dim, rng)
        pairs = []
        for _ in range(40):
            i, j = select_random_pair(X, rng)
            p = 1 / (1 + np.exp(-(X[i] - X[j]) @ w_true))
            pairs.append((i, j) if rng.random() < p else (j, i))
        w_fit = fit_w(X, dim, pairs=pairs)
        self.assertGreater(
            held_out_log_likelihood(X, w_fit, pairs),
            held_out_log_likelihood(X, np.zeros(dim), pairs),
        )

    def test_dv_is_continuous_unlike_accuracy(self):

        X = data.FEATURE_MATRIX.astype("float64")
        dim = X.shape[1]
        rng = np.random.default_rng(6)
        values = set()
        for _ in range(20):
            w = draw_participant(dim, rng)
            values.add(round(held_out_log_likelihood(X, w, [(1, 2), (3, 4), (5, 6)]), 9))
        self.assertEqual(len(values), 20)


class FatigueModelTests(SimpleTestCase):
    def test_zero_fatigue_ranks_more_faithfully_than_high_fatigue(self):

        X = data.FEATURE_MATRIX.astype("float64")
        dim = X.shape[1]
        rng = np.random.default_rng(8)
        w_true = draw_participant(dim, rng)
        items = [int(i) for i in rng.choice(X.shape[0], 10, replace=False)]
        ideal = sorted(items, key=lambda m: -(X[m] @ w_true))

        def displacement(order):
            pos = {m: k for k, m in enumerate(order)}
            return np.mean([abs(pos[m] - k) for k, m in enumerate(ideal)])

        clean = np.mean([displacement(_answer_ranking(X, w_true, items, rng, 0.0))
                         for _ in range(40)])
        noisy = np.mean([displacement(_answer_ranking(X, w_true, items, rng, 1.5))
                         for _ in range(40)])
        self.assertLess(clean, noisy)

    def test_ranking_is_always_a_permutation(self):
        X = data.FEATURE_MATRIX.astype("float64")
        rng = np.random.default_rng(9)
        w_true = draw_participant(X.shape[1], rng)
        items = [int(i) for i in rng.choice(X.shape[0], 10, replace=False)]
        for phi in (0.0, 0.5, 2.0):
            order = _answer_ranking(X, w_true, items, rng, phi)
            self.assertEqual(sorted(order), sorted(items))


class ConfigTests(SimpleTestCase):
    def test_candidate_pool_is_sane(self):
        self.assertGreaterEqual(CANDIDATE_PAIRS, 50)
        self.assertLessEqual(CANDIDATE_PAIRS, 2000)