import numpy as np
from django.test import SimpleTestCase

from project4.preference_model import fit_w, evaluate_pairs, predict_pair_prob


class TestBradleyTerryFit(SimpleTestCase):
    def test_recovers_separable_preference(self):
        # two features; true w strongly prefers high feature-0 items
        rng = np.random.default_rng(0)
        X = rng.normal(size=(20, 2))
        true_w = np.array([5.0, 0.0])
        pairs = []
        for i in range(0, 18, 2):
            a, b = i, i + 1
            winner, loser = (a, b) if X[a] @ true_w > X[b] @ true_w else (b, a)
            pairs.append((winner, loser))

        w = fit_w(X, dim=2, pairs=pairs, l2=0.1)
        acc, n = evaluate_pairs(w, X, pairs)
        self.assertEqual(n, len(pairs))
        self.assertGreaterEqual(acc, 0.85)

    def test_predict_pair_prob_matches_sigmoid(self):
        w = np.array([1.0, -1.0])
        x_i = np.array([1.0, 0.0])
        x_j = np.array([0.0, 1.0])
        p = predict_pair_prob(w, x_i, x_j)
        expected = 1 / (1 + np.exp(-((x_i - x_j) @ w)))
        self.assertAlmostEqual(p, expected, places=6)


class TestPlackettLuceFit(SimpleTestCase):
    def test_recovers_ranking_order(self):
        from scipy.stats import kendalltau

        rng = np.random.default_rng(1)
        X = rng.normal(size=(10, 2))
        true_w = np.array([8.0, 0.0])  # well-separated utilities along dim 0
        utilities = X @ true_w
        order = list(np.argsort(-utilities))  # best to worst

        # several repeats of the same (noise-free) ranking, with a light L2
        # prior just strong enough to keep the optimizer well-posed
        w = fit_w(X, dim=2, rankings=[order] * 5, l2=0.001)
        fitted_utilities = X @ w
        fitted_order = list(np.argsort(-fitted_utilities))

        tau, _ = kendalltau(order, fitted_order)
        self.assertGreater(tau, 0.9)
        self.assertEqual(fitted_order[0], order[0])
        self.assertEqual(fitted_order[-1], order[-1])

    def test_n_equals_2_matches_bradley_terry(self):
        """Plackett-Luce on a 2-item ranking should equal BT on the same pair."""
        rng = np.random.default_rng(2)
        X = rng.normal(size=(2, 3))
        w = rng.normal(size=3)

        from project4.preference_model import _pl_nll_and_grad, _bt_nll_and_grad
        pl_nll, pl_grad = _pl_nll_and_grad(w, X, [[0, 1]])
        bt_nll, bt_grad = _bt_nll_and_grad(w, X, [(0, 1)])

        self.assertAlmostEqual(pl_nll, bt_nll, places=6)
        np.testing.assert_allclose(pl_grad, bt_grad, atol=1e-6)
