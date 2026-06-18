import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

from .data import load_ag_news, LABEL_NAMES

#load test data
_, _, test_texts, test_labels = load_ag_news()

SPORTS_KEYWORDS = [
    'game', 'play', 'team', 'player', 'win', 'lose', 'score', 'coach',
    'league', 'season', 'match', 'tournament', 'championship', 'athlete',
    'football', 'basketball', 'soccer', 'baseball', 'tennis', 'golf',
    'olympic', 'sport', 'cup', 'stadium', 'fans', 'referee', 'injury',
    'pitcher', 'quarterback', 'nfl', 'nba', 'mlb', 'nhl', 'fifa',
    'goal', 'point', 'shot', 'kick', 'run', 'race', 'cyclist', 'swimmer'
]

TECH_KEYWORDS = [
    'software', 'hardware', 'computer', 'technology', 'internet', 'network',
    'digital', 'data', 'system', 'device', 'mobile', 'phone', 'chip',
    'processor', 'microsoft', 'apple', 'google', 'linux', 'windows',
    'server', 'security', 'virus', 'code', 'program', 'algorithm', 'ai',
    'robot', 'science', 'research', 'space', 'nasa', 'launch', 'satellite',
    'browser', 'email', 'wireless', 'broadband', 'download', 'upload',
    'database', 'encryption', 'firewall', 'semiconductor', 'quantum'
]


class SportsExpert:
    #Specializes in Sports
    name = "Sports Expert"
    strong_class = 1  # Sports

    def predict_single(self, text, clf_proba):
        text_lower = text.lower()
        keyword_hits = sum(1 for kw in SPORTS_KEYWORDS if kw in text_lower)

        if keyword_hits >= 2:
            if np.random.random() < 0.88:
                return 1  # Sports - expert is confident
            else:
                return int(np.argmax(clf_proba))
        else:
            # outside domain - mostly follow classifier but with noise
            if np.random.random() < 0.65:
                return int(np.argmax(clf_proba))
            else:
                return np.random.randint(0, 4)

    def predict(self, texts):
        from .classifier import predict_proba
        np.random.seed(42)
        probas = predict_proba(texts)
        return [self.predict_single(t, probas[i]) for i, t in enumerate(texts)]


class TechExpert:
    #Specizlizes in Sci/Tech
    name = "Sci/Tech Expert"
    strong_class = 3  # Sci/Tech

    def predict_single(self, text, clf_proba):
        text_lower = text.lower()
        keyword_hits = sum(1 for kw in TECH_KEYWORDS if kw in text_lower)

        if keyword_hits >= 2:
            if np.random.random() < 0.88:
                return 3  # Sci/Tech - expert is confident
            else:
                return int(np.argmax(clf_proba))
        else:
            if np.random.random() < 0.65:
                return int(np.argmax(clf_proba))
            else:
                return np.random.randint(0, 4)

    def predict(self, texts):
        from .classifier import predict_proba
        np.random.seed(42)
        probas = predict_proba(texts)
        return [self.predict_single(t, probas[i]) for i, t in enumerate(texts)]


# Instantiate experts
sports_expert = SportsExpert()
tech_expert = TechExpert()

# Evaluate on test set at import time
print("Evaluating experts on test set...")

sports_preds = sports_expert.predict(test_texts)
tech_preds = tech_expert.predict(test_texts)

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

print(f"Sports Expert overall accuracy: {sports_acc:.4f}")
print(f"Tech Expert overall accuracy: {tech_acc:.4f}")


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