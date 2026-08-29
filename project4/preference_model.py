"""
Task 2  - the Bradley-Terry model and its extension to full rankings
(Plackett-Luce), plus MAP fitting of a participant's preference vector w
from a handful of interactions.

Bradley-Terry (pairwise):
    P(i > j | w) = sigmoid(U(x_i) - U(x_j)),  U(x) = w^T x

Plackett-Luce (ranking i1 > i2 > ... > in):
    P(i1 > ... > in | w) = prod_{k=1}^{n-1} exp(U(i_k)) / sum_{l=k}^{n} exp(U(i_l))

This is the natural extension of Bradley-Terry via Luce's choice axiom: a
full ranking is modeled as a sequence of "pick the most preferred item
among those remaining" choices, each obeying the same softmax/BT choice
rule. Setting n=2 recovers exactly the Bradley-Terry pairwise model, which
is the formal justification for calling it an extension rather than a
different model. See the PDF report for the full derivation.

We fit w by maximum a posteriori estimation: minimize the negative
log-likelihood plus an L2 penalty (equivalent to a zero-mean Gaussian prior
on w). The prior is necessary because (a) BT/PL likelihoods only depend on
utility *differences*, so w is unidentifiable without regularization, and
(b) a real elicitation session only yields a handful of comparisons, far
fewer than the dimensionality of x, so the MLE alone would badly overfit.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit as sigmoid


DEFAULT_L2 = 2.0


def _bt_nll_and_grad(w, X, pairs):
    if not pairs:
        return 0.0, np.zeros_like(w)
    winners = np.array([p[0] for p in pairs])
    losers = np.array([p[1] for p in pairs])
    u = X @ w
    diff = u[winners] - u[losers]
    # nll = softplus(-diff); numerically stable
    nll = np.logaddexp(0.0, -diff).sum()
    p = sigmoid(diff)  # P(winner beats loser | w)
    coef = (1.0 - p)  # d(nll_k)/d(diff_k)... see module docstring for derivation
    grad = (coef[:, None] * (X[losers] - X[winners])).sum(axis=0)
    return nll, grad


def _pl_nll_and_grad(w, X, rankings):

    if not rankings:
        return 0.0, np.zeros_like(w)
    nll = 0.0
    grad = np.zeros_like(w)
    for order in rankings:
        order = list(order)
        n = len(order)
        for k in range(n - 1):
            remaining = order[k:]
            u_rem = X[remaining] @ w
            m = u_rem.max()
            exp_shifted = np.exp(u_rem - m)
            z = exp_shifted.sum()
            log_z = m + np.log(z)
            nll += log_z - u_rem[0]
            softmax = exp_shifted / z
            # gradient: softmax-weighted remaining features minus the chosen one
            grad += (softmax[:, None] * X[remaining]).sum(axis=0) - X[remaining[0]]
    return nll, grad


def fit_w(X, dim, pairs=None, rankings=None, l2=DEFAULT_L2, x0=None):

    pairs = pairs or []
    rankings = rankings or []
    if x0 is None:
        x0 = np.zeros(dim)

    def objective(w):
        nll_bt, grad_bt = _bt_nll_and_grad(w, X, pairs)
        nll_pl, grad_pl = _pl_nll_and_grad(w, X, rankings)
        reg = l2 * float(w @ w)
        grad_reg = 2.0 * l2 * w
        return nll_bt + nll_pl + reg, grad_bt + grad_pl + grad_reg

    result = minimize(objective, x0, jac=True, method="L-BFGS-B")
    return result.x


def predict_pair_prob(w, x_i, x_j):
    """P(i preferred to j | w)."""
    return float(sigmoid((x_i - x_j) @ w))


def evaluate_pairs(w, X, pairs):
    """Accuracy of w at predicting held-out ground-truth pairwise choices.

    pairs: list of (chosen_idx, rejected_idx) representing the participant's
    actual choices. Returns (accuracy, n).
    """
    if not pairs:
        return None, 0
    correct = 0
    for chosen, rejected in pairs:
        prob = predict_pair_prob(w, X[chosen], X[rejected])
        if prob >= 0.5:
            correct += 1
    return correct / len(pairs), len(pairs)


@dataclass
class RankingToPairwiseView:
    """Utility to convert a full ranking into its (n-1) 'effective' adjacent
    comparisons, used only for reporting comparable information budgets
    between the two elicitation designs (see report, Task 3)."""

    order: list = field(default_factory=list)

    def effective_comparisons(self):
        return max(len(self.order) - 1, 0)
