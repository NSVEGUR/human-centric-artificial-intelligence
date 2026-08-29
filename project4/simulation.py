"""
Simulation study backing two claims in the report:

  (A) how much the adaptive selection rule in selection.py actually helps;
  (B) what sample size Task 3 needs, estimated from the DV we measure
      rather than from an assumed effect size.

Caveat for both: synthetic participants obey Bradley-Terry / Plackett-Luce
exactly and real people do not, so these are optimistic ceilings. This
flatters adaptive selection especially, since it seeks out near-tied
comparisons - exactly where real humans are least consistent.

Offline only. Nothing here runs during a participant session; results are
precomputed by `python manage.py p4_simulate` and cached to
data/simulation_results.json for the report to read.
"""
import numpy as np
from scipy.special import expit as sigmoid

from .preference_model import fit_w, evaluate_pairs
from .selection import select_next_pair, select_random_pair
from .study_config import (
    PAIRWISE_ELICIT_N, RANKING_ELICIT_N, RANKING_SIZE, VALIDATION_N,
)

# Scale of the synthetic taste vectors. 0.5 on standardized features gives
# utility gaps that produce choice probabilities spread across the whole
# range rather than clustering at 0 or 1 (i.e. simulated people who have
# real but not deterministic preferences).
W_TRUE_SCALE = 0.5

# Held-out set used *inside* the simulation to measure model quality. This
# is intentionally larger than the study's VALIDATION_N: here we want a
# low-variance estimate of how good a fitted w is, whereas VALIDATION_N is
# constrained by how long we can keep a real participant.
SIM_VALIDATION_N = 40


#simulated participants

def draw_participant(dim, rng):
    return rng.normal(size=dim) * W_TRUE_SCALE


def _answer_pair(X, w_true, i, j, rng):
    p = sigmoid((X[i] - X[j]) @ w_true)
    return (i, j) if rng.random() < p else (j, i)


def _answer_ranking(X, w_true, items, rng, fatigue=0.0):
    remaining = list(items)
    order = []
    step = 0
    while remaining:
        u = X[remaining] @ w_true
        temperature = 1.0 + fatigue * step
        scaled = u / temperature
        e = np.exp(scaled - scaled.max())
        order.append(remaining.pop(int(rng.choice(len(remaining), p=e / e.sum()))))
        step += 1
    return order


def _held_out_pairs(X, w_true, rng, n=SIM_VALIDATION_N, exclude=()):
    avail = np.setdiff1d(np.arange(X.shape[0]), np.asarray(list(exclude), dtype=int))
    out = []
    for _ in range(n):
        i, j = rng.choice(avail, 2, replace=False)
        out.append(_answer_pair(X, w_true, int(i), int(j), rng))
    return out


def held_out_log_likelihood(X, w, pairs):
    if not pairs:
        return None
    lls = [np.log(max(sigmoid((X[a] - X[b]) @ w), 1e-12)) for a, b in pairs]
    return float(np.mean(lls))


#(A) adaptive vs random selection 
def learning_curve(X, w_true, strategy, n_trials, rng, held_out):
    
    dim = X.shape[1]
    pairs, seen, w = [], set(), np.zeros(dim)
    accuracies = []
    for _ in range(n_trials):
        if strategy == "adaptive":
            i, j = select_next_pair(X, w, pairs, rng, exclude=seen)
        else:
            i, j = select_random_pair(X, rng, exclude=seen)
        seen.update((i, j))
        pairs.append(_answer_pair(X, w_true, i, j, rng))
        w = fit_w(X, dim, pairs=pairs)
        accuracies.append(evaluate_pairs(w, X, held_out)[0])
    return accuracies


def compare_selection_strategies(X, n_participants=25, n_trials=PAIRWISE_ELICIT_N, seed=7):
    
    rng = np.random.default_rng(seed)
    dim = X.shape[1]
    curves = {"random": [], "adaptive": []}
    for _ in range(n_participants):
        w_true = draw_participant(dim, rng)
        held_out = _held_out_pairs(X, w_true, rng)
        for strategy in curves:
            curves[strategy].append(
                learning_curve(X, w_true, strategy, n_trials, rng, held_out)
            )

    result = {k: np.asarray(v).mean(axis=0).tolist() for k, v in curves.items()}
    final_random = result["random"][-1]
    reached = next(
        (k + 1 for k, a in enumerate(result["adaptive"]) if a >= final_random), None
    )
    return {
        "n_participants": n_participants,
        "n_trials": n_trials,
        "random": result["random"],
        "adaptive": result["adaptive"],
        "trials_for_adaptive_to_match_random": reached,
    }


#(B) simulation-based power analysis

def simulate_one_session(X, rng, fatigue_levels):
    dim = X.shape[1]
    w_true = draw_participant(dim, rng)
    seen = set()

    pairs = []
    for _ in range(PAIRWISE_ELICIT_N):
        i, j = select_random_pair(X, rng, exclude=seen)
        seen.update((i, j))
        pairs.append(_answer_pair(X, w_true, i, j, rng))

    # the same ten-movie lists are ranked at every fatigue level
    ranking_items = []
    for _ in range(RANKING_ELICIT_N):
        avail = np.setdiff1d(np.arange(X.shape[0]), np.asarray(list(seen), dtype=int))
        items = [int(m) for m in rng.choice(avail, RANKING_SIZE, replace=False)]
        seen.update(items)
        ranking_items.append(items)

    held_out = _held_out_pairs(X, w_true, rng, n=VALIDATION_N, exclude=seen)

    w_pairwise = fit_w(X, dim, pairs=pairs)
    acc_p = evaluate_pairs(w_pairwise, X, held_out)[0]
    ll_p = held_out_log_likelihood(X, w_pairwise, held_out)

    out = {}
    for fatigue in fatigue_levels:
        rankings = [_answer_ranking(X, w_true, items, rng, fatigue=fatigue)
                    for items in ranking_items]
        w_ranking = fit_w(X, dim, rankings=rankings)
        out[fatigue] = (
            acc_p,
            evaluate_pairs(w_ranking, X, held_out)[0],
            ll_p,
            held_out_log_likelihood(X, w_ranking, held_out),
        )
    return out


def power_analysis(X, fatigue_levels=(0.0, 0.3, 0.6), pool_size=200,
                   sample_sizes=(20, 40, 60, 80, 100, 140), n_bootstrap=400,
                   alpha=0.05, seed=11):
    from scipy.stats import wilcoxon

    rng = np.random.default_rng(seed)
    sessions = [simulate_one_session(X, rng, fatigue_levels) for _ in range(pool_size)]

    out = {}
    for fatigue in fatigue_levels:
        pool = np.array([s[fatigue] for s in sessions])
        dvs = {
            "accuracy": pool[:, 0] - pool[:, 1],
            "log_likelihood": pool[:, 2] - pool[:, 3],
        }

        entry = {
            "mean_pairwise_accuracy": float(pool[:, 0].mean()),
            "mean_ranking_accuracy": float(pool[:, 1].mean()),
            "dvs": {},
        }
        for dv_name, diffs in dvs.items():
            powers = {}
            for n in sample_sizes:
                rejects = 0
                for _ in range(n_bootstrap):
                    sample = rng.choice(diffs, n, replace=True)
                    if np.allclose(sample, 0):
                        continue
                    try:
                        if wilcoxon(sample).pvalue < alpha:
                            rejects += 1
                    except ValueError:
                        continue
                powers[n] = rejects / n_bootstrap
            sd = float(diffs.std(ddof=1))
            entry["dvs"][dv_name] = {
                "mean_difference": float(diffs.mean()),
                "sd_difference": sd,
                "effect_size_dz": float(diffs.mean() / sd) if sd > 0 else None,
                "distinct_values": int(len(set(np.round(diffs, 9)))),
                "power_by_n": powers,
                "smallest_n_with_80_power": next(
                    (n for n in sorted(powers) if powers[n] >= 0.80), None
                ),
            }
        out[str(fatigue)] = entry
    return out