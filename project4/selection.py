"""
Extension - adaptive (informative) item selection.

The assignment notes that investigating more informative selection
strategies would be an interesting extension. This implements one.

It is deliberately NOT used in the Task 3 study: both conditions there use
random selection as specified, and applying an adaptive rule to only one
would confound selection strategy with elicitation format. Only
simulation.py exercises this module.

Criterion (Bayesian D-optimal design for Bradley-Terry): score a candidate
pair by p(1-p) * d^T Sigma d, where d = x_i - x_j, p = sigmoid(d^T w), and
Sigma is the Laplace posterior covariance. The first factor prefers
comparisons whose outcome is uncertain; the second prefers pairs that probe
directions of w we know least about. At w = 0 both are uninformative about
direction, so the score reduces to ||d||^2 - the cold start needs no
special case.
"""
import numpy as np
from scipy.special import expit as sigmoid

from .preference_model import DEFAULT_L2

# Candidate pairs scored per selection step. 300 keeps a single step at a
# few milliseconds on the ~4,900-movie catalog while capturing most of the
# available gain; raising it shows sharply diminishing returns.
CANDIDATE_PAIRS = 300


def posterior_covariance(X, w, pairs, l2=DEFAULT_L2):
    dim = X.shape[1]
    H = 2.0 * l2 * np.eye(dim)
    for winner, loser in pairs:
        d = X[winner] - X[loser]
        p = sigmoid(d @ w)
        H += p * (1.0 - p) * np.outer(d, d)
    return np.linalg.inv(H)


def score_pairs(X, w, sigma, candidates):
    
    D = X[candidates[:, 0]] - X[candidates[:, 1]]
    p = sigmoid(D @ w)
    # d^T Sigma d for each row, without forming the full m x m product
    quad = np.einsum("ij,jk,ik->i", D, sigma, D)
    return p * (1.0 - p) * quad


def select_next_pair(X, w, pairs, rng, n_candidates=CANDIDATE_PAIRS, exclude=()):
    
    n_items = X.shape[0]
    available = np.setdiff1d(np.arange(n_items), np.asarray(list(exclude), dtype=int))

    draw = min(2 * n_candidates, len(available))
    picked = rng.choice(available, draw, replace=False)
    half = len(picked) // 2
    candidates = np.stack([picked[:half], picked[half:2 * half]], axis=1)

    sigma = posterior_covariance(X, w, pairs)
    scores = score_pairs(X, w, sigma, candidates)
    best = int(np.argmax(scores))
    return int(candidates[best, 0]), int(candidates[best, 1])


def select_random_pair(X, rng, exclude=()):
    n_items = X.shape[0]
    available = np.setdiff1d(np.arange(n_items), np.asarray(list(exclude), dtype=int))
    i, j = rng.choice(available, 2, replace=False)
    return int(i), int(j)