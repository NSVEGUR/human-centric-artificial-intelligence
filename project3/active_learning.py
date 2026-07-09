import numpy as np
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

from .data import LABEL_NAMES, load_ag_news
from .classifier import predict_proba
from .experts import sports_expert, tech_expert, sports_per_class, tech_per_class

print("  [4/4] Running Active Learning simulation...")

# ── Setup ─────────────────────────────────────────────────────────────────────
np.random.seed(0)

_, _, test_texts, test_labels = load_ag_news()
test_texts = list(test_texts)
test_labels = np.array(test_labels)
N = len(test_texts)

N_POOL = 2000
N_EVAL = N - N_POOL
N_QUERIES = 200
EVAL_EVERY = 10

# Split: first N_POOL as unlabeled pool, rest as evaluation set
pool_idx = np.random.choice(N, N_POOL, replace=False)
eval_mask = np.ones(N, dtype=bool)
eval_mask[pool_idx] = False
eval_idx = np.where(eval_mask)[0]

pool_texts = [test_texts[i] for i in pool_idx]
pool_labels = test_labels[pool_idx]
eval_texts = [test_texts[i] for i in eval_idx]
eval_labels = test_labels[eval_idx]

# Precompute classifier probabilities on pool and eval sets
print("         Computing classifier probabilities...")
pool_probas = predict_proba(pool_texts)   # (N_POOL, 4)
eval_probas = predict_proba(eval_texts)   # (N_EVAL, 4)

# Precompute ground-truth expert predictions on pool (simulate querying)
# We choose the "best expert" per-instance for Task 4 queries
print("         Precomputing expert predictions on pool...")
pool_sports_preds = np.array(sports_expert.predict(pool_texts, pool_labels))
pool_tech_preds = np.array(tech_expert.predict(pool_texts, pool_labels))

sports_frac = {k: v / 100.0 for k, v in sports_per_class.items()}
tech_frac = {k: v / 100.0 for k, v in tech_per_class.items()}

# Best expert per class (oracle knowledge for choosing which expert to query)
best_expert_per_class_true = np.array([
    max(sports_frac[LABEL_NAMES[k]], tech_frac[LABEL_NAMES[k]])
    for k in range(4)
])

# Pool predicted classes from classifier
pool_pred_class = np.argmax(pool_probas, axis=1)

# Select which expert's prediction to use when queried (best for that predicted class)
pool_expert_preds = np.where(
    np.array([sports_frac[LABEL_NAMES[c]] >= tech_frac[LABEL_NAMES[c]]
              for c in pool_pred_class]),
    pool_sports_preds,
    pool_tech_preds,
)

# Precompute eval expert preds (for deferral evaluation on eval set)
eval_sports_preds = np.array(sports_expert.predict(eval_texts, eval_labels))
eval_tech_preds = np.array(tech_expert.predict(eval_texts, eval_labels))
eval_pred_class = np.argmax(eval_probas, axis=1)
eval_expert_preds = np.where(
    np.array([sports_frac[LABEL_NAMES[c]] >= tech_frac[LABEL_NAMES[c]]
              for c in eval_pred_class]),
    eval_sports_preds,
    eval_tech_preds,
)


# ── Utility functions ─────────────────────────────────────────────────────────
def _least_confidence(probas):
    return 1.0 - probas.max(axis=1)

def _margin(probas):
    sorted_p = np.sort(probas, axis=1)
    return 1.0 - (sorted_p[:, -1] - sorted_p[:, -2])

def _entropy(probas):
    eps = 1e-10
    return -(probas * np.log(probas + eps)).sum(axis=1)


def _evaluate_deferral(est_per_class_acc, eval_probas, eval_expert_preds, eval_labels):
    """Compute team accuracy using learned expert competence estimate."""
    est_frac = est_per_class_acc  # shape (4,)
    p_expert_correct = eval_probas @ est_frac
    clf_uncertainty = 1.0 - eval_probas.max(axis=1)
    expert_error = 1.0 - p_expert_correct

    defer_mask = clf_uncertainty > expert_error  # alpha=1 (Bayes-optimal)
    clf_preds = np.argmax(eval_probas, axis=1)
    team_preds = np.where(defer_mask, eval_expert_preds, clf_preds)
    return accuracy_score(eval_labels, team_preds) * 100


# ── Active Learning Simulation ────────────────────────────────────────────────
STRATEGIES = {
    'Random':           lambda p: np.random.rand(len(p)),
    'Least Confidence': _least_confidence,
    'Margin':           _margin,
    'Entropy':          _entropy,
}

# Baseline: oracle (using true expert competence from the start)
oracle_acc = _evaluate_deferral(
    best_expert_per_class_true, eval_probas, eval_expert_preds, eval_labels
)

results = {}  # strategy_name -> list of (n_queries, team_acc)

for strategy_name, utility_fn in STRATEGIES.items():
    print(f"         Strategy: {strategy_name}...")
    np.random.seed(42)

    # Running estimates of expert per-class accuracy
    correct_by_class = np.zeros(4)
    total_by_class = np.zeros(4)
    # Laplace smoothing prior: start with 0.5 (uniform uncertainty)
    prior_correct = np.ones(4) * 5.0
    prior_total = np.ones(4) * 10.0

    queried = set()
    curve = []

    for q in range(1, N_QUERIES + 1):
        # Compute utility over remaining pool
        available = [i for i in range(N_POOL) if i not in queried]
        avail_probas = pool_probas[available]
        utils = utility_fn(avail_probas)

        # Select highest utility
        best_local = int(np.argmax(utils))
        best_idx = available[best_local]
        queried.add(best_idx)

        # "Query expert": get expert prediction and true label
        expert_pred = pool_expert_preds[best_idx]
        true_label = pool_labels[best_idx]

        # Update competence estimate for the true class
        correct_by_class[true_label] += int(expert_pred == true_label)
        total_by_class[true_label] += 1

        # Estimate per-class accuracy with Laplace smoothing
        est_per_class_acc = (correct_by_class + prior_correct) / (total_by_class + prior_total)

        # Evaluate every EVAL_EVERY queries
        if q % EVAL_EVERY == 0:
            team_acc = _evaluate_deferral(est_per_class_acc, eval_probas, eval_expert_preds, eval_labels)
            curve.append((q, round(team_acc, 3)))

    results[strategy_name] = curve

print(f"         Oracle team accuracy: {oracle_acc:.2f}%")
print("─" * 52)


def get_al_stats():
    return {
        "strategies": list(results.keys()),
        "curves": {
            name: {"queries": [p[0] for p in curve], "accuracy": [p[1] for p in curve]}
            for name, curve in results.items()
        },
        "oracle_acc": round(oracle_acc, 2),
        "n_queries": N_QUERIES,
        "n_pool": N_POOL,
        "n_eval": N_EVAL,
    }


def build_al_learning_curves_plot():
    stats = get_al_stats()

    colors = {
        'Random':           '#f59e0b',
        'Least Confidence': '#60a5fa',
        'Margin':           '#10b981',
        'Entropy':          '#8b5cf6',
    }

    fig = go.Figure()

    for name, color in colors.items():
        curve = stats["curves"][name]
        fig.add_trace(go.Scatter(
            x=curve["queries"],
            y=curve["accuracy"],
            mode='lines+markers',
            name=name,
            line=dict(color=color, width=2),
            marker=dict(size=4),
        ))

    # Oracle dashed line
    fig.add_hline(
        y=stats["oracle_acc"],
        line=dict(color='#ec4899', width=1.5, dash='dot'),
        annotation_text=f"Oracle (true competence): {stats['oracle_acc']:.1f}%",
        annotation_position="bottom right",
        annotation_font=dict(color='#ec4899', size=10),
    )

    fig.update_layout(
        title="Active Learning: Team Accuracy vs. Expert Queries",
        xaxis_title="Number of Expert Queries",
        yaxis_title="Team Accuracy (%)",
        yaxis=dict(range=[65, 100]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        legend=dict(bgcolor='rgba(0,0,0,0.3)', bordercolor='rgba(255,255,255,0.1)', borderwidth=1),
        xaxis_gridcolor='rgba(255,255,255,0.06)',
        yaxis_gridcolor='rgba(255,255,255,0.06)',
    )

    return fig.to_json()
