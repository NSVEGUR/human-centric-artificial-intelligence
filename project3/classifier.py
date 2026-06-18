import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objects as go
import json

from .data import load_ag_news, LABEL_NAMES

print("Training AG News classifier...")
train_texts, train_labels, test_texts, test_labels = load_ag_news()

model = Pipeline([
    ('tfidf', TfidfVectorizer(max_features=50000, ngram_range=(1, 2), sublinear_tf=True)),
    ('clf', LogisticRegression(max_iter=1000, C=5.0, random_state=42)),
])

model.fit(train_texts, train_labels)

test_preds = model.predict(test_texts)
test_acc = accuracy_score(test_labels, test_preds)
conf_matrix = confusion_matrix(test_labels, test_preds)

print(f"Classifier test accuracy: {test_acc:.4f}")


def get_classifier_stats():
    return {
        "test_acc": round(test_acc * 100, 2),
        "label_names": LABEL_NAMES,
        "conf_matrix": conf_matrix.tolist(),
    }


def build_confusion_matrix_plot():
    cm = conf_matrix
    z = cm / cm.sum(axis=1, keepdims=True) 

    fig = go.Figure(data=go.Heatmap(
        z=z,
        x=LABEL_NAMES,
        y=LABEL_NAMES,
        colorscale='Blues',
        text=cm,
        texttemplate="%{text}",
        showscale=True,
    ))

    fig.update_layout(
        title="Confusion Matrix (normalized by row)",
        xaxis_title="Predicted",
        yaxis_title="True",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#e0e0e0'),
    )

    return fig.to_json()


def predict(texts):
    return model.predict(texts)


def predict_proba(texts):
    return model.predict_proba(texts)
