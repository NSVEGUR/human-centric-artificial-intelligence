import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from .data import load_ag_news, LABEL_NAMES

_, _, test_texts, test_labels = load_ag_news()

# ── Expert Design ──────────────────────────────────────────────────────────────
# Each expert is simulated with a specified per-class accuracy profile:
#   - Very high accuracy on their specialty class (> classifier baseline)
#   - Weaker accuracy on other classes
#
# This models the reality from the lecture: experts have limited domain expertise
# that outperforms the AI in specific regions of the input space, while being
# worse elsewhere. The L2D system benefits by deferring to the expert only on
# their specialty class.
#
# When the expert is wrong, they predict the second-most-likely class according
# to the classifier — a realistic error pattern.

class SportsExpert:
    """
    Specialises in Sports (class 1).
    Per-class accuracy: Sports ≈ 97%, all others ≈ 55%.
    Achieves higher Sports accuracy than the classifier's ~97.7% on easy Sports
    articles, while being unreliable on World/Business/Sci/Tech.
    """
    name = "Sports Expert"
    strong_class = 1  # Sports

    PER_CLASS_ACCURACY = {
        0: 0.55,   # World   — unreliable
        1: 0.97,   # Sports  — specialist
        2: 0.55,   # Business — unreliable
        3: 0.55,   # Sci/Tech — unreliable
    }

    def predict_single(self, true_label, clf_proba):
        p = self.PER_CLASS_ACCURACY[true_label]
        if np.random.random() < p:
            return true_label  # expert is correct
        # Wrong prediction: pick second-most-likely class (realistic mistake)
        sorted_cls = np.argsort(clf_proba)[::-1]
        for c in sorted_cls:
            if c != true_label:
                return int(c)
        return int((true_label + 1) % 4)

    def predict(self, texts, true_labels_input):
        from .classifier import predict_proba
        np.random.seed(42)
        labels = list(true_labels_input)
        probas = predict_proba(texts)
        return [self.predict_single(labels[i], probas[i]) for i in range(len(texts))]


class TechExpert:
    """
    Specialises in Sci/Tech (class 3).
    Per-class accuracy: Sci/Tech ≈ 95%, all others ≈ 55%.
    """
    name = "Sci/Tech Expert"
    strong_class = 3  # Sci/Tech

    PER_CLASS_ACCURACY = {
        0: 0.55,   # World   — unreliable
        1: 0.55,   # Sports  — unreliable
        2: 0.55,   # Business — unreliable
        3: 0.95,   # Sci/Tech — specialist
    }

    def predict_single(self, true_label, clf_proba):
        p = self.PER_CLASS_ACCURACY[true_label]
        if np.random.random() < p:
            return true_label
        sorted_cls = np.argsort(clf_proba)[::-1]
        for c in sorted_cls:
            if c != true_label:
                return int(c)
        return int((true_label + 1) % 4)

    def predict(self, texts, true_labels_input):
        from .classifier import predict_proba
        np.random.seed(42)
        labels = list(true_labels_input)
        probas = predict_proba(texts)
        return [self.predict_single(labels[i], probas[i]) for i in range(len(texts))]


# Instantiate experts
sports_expert = SportsExpert()
tech_expert = TechExpert()

# Evaluate on test set at import time
print("  [2/4] Evaluating expert annotators...")

sports_preds = sports_expert.predict(test_texts, test_labels)
tech_preds = tech_expert.predict(test_texts, test_labels)

sports_acc = accuracy_score(test_labels, sports_preds)
tech_acc = accuracy_score(test_labels, tech_preds)

sports_conf = confusion_matrix(test_labels, sports_preds)
tech_conf = confusion_matrix(test_labels, tech_preds)


def per_class_accuracy(y_true, y_pred):
    accs = {}
    for i, name in enumerate(LABEL_NAMES):
        idx = [j for j, y in enumerate(y_true) if y == i]
        if idx:
            correct = sum(1 for j in idx if y_pred[j] == i)
            accs[name] = round(correct / len(idx) * 100, 2)
        else:
            accs[name] = 0.0
    return accs


sports_per_class = per_class_accuracy(test_labels, sports_preds)
tech_per_class = per_class_accuracy(test_labels, tech_preds)

print(f"         Sports: {sports_acc * 100:.2f}%   Tech: {tech_acc * 100:.2f}%")


def get_expert_stats():
    return {
        "experts": [
            {
                "name": sports_expert.name,
                "overall_acc": round(sports_acc * 100, 2),
                "strong_class": LABEL_NAMES[sports_expert.strong_class],
                "per_class_acc": sports_per_class,
                "conf_matrix": sports_conf.tolist(),
            },
            {
                "name": tech_expert.name,
                "overall_acc": round(tech_acc * 100, 2),
                "strong_class": LABEL_NAMES[tech_expert.strong_class],
                "per_class_acc": tech_per_class,
                "conf_matrix": tech_conf.tolist(),
            },
        ],
        "label_names": LABEL_NAMES,
    }


def build_expert_accuracy_plot():
    import plotly.graph_objects as go

    fig = go.Figure()

    for expert_data in get_expert_stats()["experts"]:
        fig.add_trace(go.Bar(
            name=expert_data["name"],
            x=LABEL_NAMES,
            y=[expert_data["per_class_acc"][label] for label in LABEL_NAMES],
        ))

    fig.update_layout(
        barmode='group',
        title="Expert Per-Class Accuracy (%)",
        xaxis_title="News Category",
        yaxis_title="Accuracy (%)",
        yaxis=dict(range=[0, 100]),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
    )

    return fig.to_json()
