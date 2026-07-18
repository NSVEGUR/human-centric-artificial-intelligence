import numpy as np
from sklearn.metrics import accuracy_score
import plotly.graph_objects as go

from .data import LABEL_NAMES
from .classifier import predict_proba, test_preds, test_acc
from .experts import (
    sports_expert, tech_expert,
    sports_preds, tech_preds,
    sports_per_class, tech_per_class,
    sports_acc, tech_acc,
    test_texts, test_labels,
)

print("  [3/4] Computing Learning-to-Defer evaluation...")

test_labels_arr = np.array(test_labels)
clf_probas = predict_proba(test_texts)           # shape (N_test, 4)
clf_preds_arr = np.array(test_preds)
sports_preds_arr = np.array(sports_preds)
tech_preds_arr = np.array(tech_preds)

# Per-class expert accuracy as fractions
sports_frac = {k: v / 100.0 for k, v in sports_per_class.items()}
tech_frac = {k: v / 100.0 for k, v in tech_per_class.items()}

# Best expert per class (whichever has higher per-class accuracy)
best_expert_per_class = np.array([
    max(sports_frac[LABEL_NAMES[k]], tech_frac[LABEL_NAMES[k]])
    for k in range(4)
])

# For each instance, defer to the expert with higher accuracy on the predicted class
clf_pred_class = np.argmax(clf_probas, axis=1)
use_sports = np.array([
    sports_frac[LABEL_NAMES[c]] >= tech_frac[LABEL_NAMES[c]]
    for c in clf_pred_class
])
best_expert_preds = np.where(use_sports, sports_preds_arr, tech_preds_arr)

# ── Bayes-optimal deferral rule ───────────────────────────────────────────────
# P(best expert correct | x) = Σ_k P(y=k|x) × best_expert_per_class[k]
# Classifier uncertainty = 1 − max_k P(y=k|x)
# Defer if classifier_uncertainty > α × expert_error
#   α=0 → always defer   α→∞ → never defer   α=1 → Bayes-optimal

p_expert_correct = clf_probas @ best_expert_per_class  # (N,)
clf_uncertainty  = 1.0 - clf_probas.max(axis=1)        # (N,)
expert_error     = 1.0 - p_expert_correct              # (N,)

alphas = np.linspace(0.0, 4.0, 200)
coverages = []
team_accuracies = []

for alpha in alphas:
    defer_mask = clf_uncertainty > alpha * expert_error
    team_preds = np.where(defer_mask, best_expert_preds, clf_preds_arr)
    coverages.append(float(1.0 - defer_mask.mean()))
    team_accuracies.append(float(accuracy_score(test_labels_arr, team_preds) * 100))

# Bayes-optimal operating point (α=1)
alpha1_idx = int(np.argmin(np.abs(alphas - 1.0)))
optimal_coverage     = coverages[alpha1_idx]
optimal_team_acc     = team_accuracies[alpha1_idx]
optimal_deferral_rate = 1.0 - optimal_coverage

ai_only_acc        = test_acc * 100
sports_only_acc    = sports_acc * 100
tech_only_acc      = tech_acc * 100
best_expert_only_acc = accuracy_score(test_labels_arr, best_expert_preds) * 100

print(
    f"         AI-only: {ai_only_acc:.2f}%   Team (α=1): {optimal_team_acc:.2f}%   "
    f"Coverage: {optimal_coverage:.2%}   Deferral: {optimal_deferral_rate:.2%}"
)


def get_deferral_stats():
    return {
        "baselines": {
            "ai_only":      round(ai_only_acc, 2),
            "sports_expert": round(sports_only_acc, 2),
            "tech_expert":   round(tech_only_acc, 2),
            "best_expert":   round(best_expert_only_acc, 2),
        },
        "optimal": {
            "team_acc":      round(optimal_team_acc, 2),
            "coverage":      round(optimal_coverage * 100, 2),
            "deferral_rate": round(optimal_deferral_rate * 100, 2),
        },
        "coverages":       coverages,
        "team_accuracies": team_accuracies,
        "label_names":     LABEL_NAMES,
    }


def build_accuracy_coverage_plot():
    stats = get_deferral_stats()

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=stats["coverages"],
        y=stats["team_accuracies"],
        mode='lines',
        name='L2D System (sweep α)',
        line=dict(color='#60a5fa', width=2.5),
    ))

    fig.add_trace(go.Scatter(
        x=[optimal_coverage],
        y=[optimal_team_acc],
        mode='markers',
        name=f'Optimal (α=1): {optimal_team_acc:.1f}%',
        marker=dict(size=10, color='#60a5fa', symbol='circle',
                    line=dict(color='white', width=2)),
    ))

    baselines = [
        ("AI Only",         stats["baselines"]["ai_only"],        '#f59e0b', 'dot'),
        ("Sports Expert",   stats["baselines"]["sports_expert"],  '#10b981', 'dash'),
        ("Sci/Tech Expert", stats["baselines"]["tech_expert"],    '#8b5cf6', 'dash'),
        ("Best Expert",     stats["baselines"]["best_expert"],    '#ec4899', 'dot'),
    ]
    for name, val, color, dash in baselines:
        fig.add_hline(
            y=val,
            line=dict(color=color, width=1.5, dash=dash),
            annotation_text=f"{name} ({val:.1f}%)",
            annotation_position="bottom right",
            annotation_font=dict(color=color, size=10),
        )

    fig.update_layout(
        title="L2D: Team Accuracy vs. Coverage",
        xaxis_title="Coverage (fraction handled by AI classifier)",
        yaxis_title="Team Accuracy (%)",
        xaxis=dict(range=[0, 1.02], tickformat=".0%"),
        yaxis=dict(range=[65, 100]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        legend=dict(bgcolor='rgba(0,0,0,0.3)', bordercolor='rgba(255,255,255,0.1)', borderwidth=1),
        xaxis_gridcolor='rgba(255,255,255,0.06)',
        yaxis_gridcolor='rgba(255,255,255,0.06)',
    )
    return fig.to_json()


def build_deferral_summary_plot():
    stats = get_deferral_stats()

    labels = ['AI Only', 'Sports Expert', 'Sci/Tech Expert', 'Best Expert\n(oracle)', 'L2D Team\n(α=1)']
    values = [
        stats["baselines"]["ai_only"],
        stats["baselines"]["sports_expert"],
        stats["baselines"]["tech_expert"],
        stats["baselines"]["best_expert"],
        stats["optimal"]["team_acc"],
    ]
    colors = ['#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#60a5fa']

    fig = go.Figure(go.Bar(
        x=labels,
        y=values,
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition='outside',
        textfont=dict(size=11),
    ))

    fig.update_layout(
        title="Accuracy Comparison: AI vs. Expert vs. L2D Team",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 105]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
        showlegend=False,
        xaxis_gridcolor='rgba(0,0,0,0)',
        yaxis_gridcolor='rgba(255,255,255,0.06)',
    )
    return fig.to_json()
