from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objects as go
import numpy as np

from .data import load_and_preprocess

train_X, test_X, train_y, test_y, mad_vals, num_cols = load_and_preprocess()

# scale the features - logistic regression really needs this
# fit only on train, then apply same transform to test
scaler = StandardScaler()
scaled_train_X = scaler.fit_transform(train_X)
scaled_test_X = scaler.transform(test_X)

# C is inverse regularization in sklearn - smaller C = more regularized = simpler
# tried a few ranges, this spread seems to cover both extremes well enough
c_values = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]

all_lr_models = []

for c in c_values:
    model = LogisticRegression(C=c, max_iter=1000, random_state=42, solver="lbfgs")
    model.fit(scaled_train_X, train_y)

    test_preds = model.predict(scaled_test_X)
    test_acc = accuracy_score(test_y, test_preds)

    train_preds = model.predict(scaled_train_X)
    train_acc = accuracy_score(train_y, train_preds)

    # omega = L1 norm of weights = sum of absolute values of all coefficients
    # coef_ shape is (n_classes, n_features) for multiclass
    omega = float(np.sum(np.abs(model.coef_)))

    all_lr_models.append({
        "model": model,
        "C": c,
        "omega": omega,
        "test_acc": test_acc,
        "train_acc": train_acc,
    })

# same normalization trick as decision_tree.py
max_omega = max(m["omega"] for m in all_lr_models)
min_omega = min(m["omega"] for m in all_lr_models)


def get_best_lr(lam):
    # same formula as trees - minimize (1 - test_acc) + lambda * norm_omega
    # lam=0 picks most accurate, higher lam prefers smaller weights

    winner = None
    lowest = float("inf")

    for m in all_lr_models:
        norm_omega = (m["omega"] - min_omega) / (max_omega - min_omega + 1e-9)
        score = (1 - m["test_acc"]) + lam * norm_omega

        if score < lowest:
            lowest = score
            winner = m

    return winner


def build_lr_plotly(lr_info, sort_by="magnitude", filter_class=None, filter_feature=None):
    # bar chart of coefficients for each class
    # each bar = one feature, height = how much influence it has
    # positive = pushes towards this class, negative = pushes away

    model = lr_info["model"]
    feat_names = list(train_X.columns)
    species = list(model.classes_)

    # one color per species - reusing same colors as decision tree for consistency
    bar_colors = ["#4C72B0", "#55A868", "#C44E52"]

    fig = go.Figure()

    show_species = species if filter_class is None else [filter_class]

    for i, sp in enumerate(species):
        if sp not in show_species:
            continue

        weights = model.coef_[i]
        pairs = list(zip(feat_names, weights))

        # partial match on feature name so "island" catches island_Biscoe etc
        if filter_feature:
            pairs = [(f, w) for f, w in pairs if filter_feature in f]

        if sort_by == "magnitude":
            pairs = sorted(pairs, key=lambda x: abs(x[1]), reverse=True)
        else:
            # alphabetical
            pairs = sorted(pairs, key=lambda x: x[0])

        feats = [p[0] for p in pairs]
        weights = [p[1] for p in pairs]

        fig.add_trace(go.Bar(
            name=sp,
            x=feats,
            y=weights,
            marker_color=bar_colors[i % len(bar_colors)],
            hovertemplate="<b>%{x}</b><br>Coefficient: %{y:.4f}<extra>" + sp + "</extra>",
        ))

    fig.update_layout(
        title=f"Logistic Regression — C={lr_info['C']} | Omega={lr_info['omega']:.3f} | Test Acc: {lr_info['test_acc']:.2%} | Train Acc: {lr_info['train_acc']:.2%}",
        xaxis_title="Feature",
        yaxis_title="Coefficient Value",
        barmode="group",
        xaxis_tickangle=-45,
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=500,
        margin=dict(l=20, r=20, t=60, b=120),  # big bottom margin for rotated labels
        legend=dict(orientation="h", y=-0.35)
    )

    return fig.to_json()


def build_tradeoff_plot(selected_lr):
    # same as the tree version but x axis is omega instead of leaf count
    # basically copy pasted from decision_tree.py and changed the x axis

    all_omega = [m["omega"] for m in all_lr_models]
    test_accs = [m["test_acc"] * 100 for m in all_lr_models]
    train_accs = [m["train_acc"] * 100 for m in all_lr_models]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=all_omega, y=train_accs,
        mode="lines+markers",
        name="Train Accuracy",
        line=dict(color="#4C72B0", width=2),
        marker=dict(size=6),
        hovertemplate="Omega: %{x:.3f}<br>Train Acc: %{y:.2f}%<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=all_omega, y=test_accs,
        mode="lines+markers",
        name="Test Accuracy",
        line=dict(color="#55A868", width=2),
        marker=dict(size=6),
        hovertemplate="Omega: %{x:.3f}<br>Test Acc: %{y:.2f}%<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=[selected_lr["omega"]],
        y=[selected_lr["test_acc"] * 100],
        mode="markers",
        name="Selected Model",
        marker=dict(size=14, color="#C44E52", symbol="star"),
        hovertemplate=f"Selected — Omega: {selected_lr['omega']:.3f}<br>Test Acc: {selected_lr['test_acc']:.2%}<extra></extra>"
    ))

    fig.update_layout(
        title="Accuracy vs Complexity Tradeoff (L1 Norm)",
        xaxis_title="Complexity — L1 Norm of Coefficients",
        yaxis_title="Accuracy (%)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=300,
        margin=dict(l=20, r=20, t=50, b=40),
        legend=dict(orientation="h", y=-0.25),
        yaxis=dict(range=[50, 102])
    )

    return fig.to_json()


def build_confusion_plot(lr_info):
    # same as decision tree confusion matrix
    # only difference is we use scaled_test_X here since LR needs scaled input

    model = lr_info["model"]
    preds = model.predict(scaled_test_X)
    species = list(model.classes_)

    cm = confusion_matrix(test_y, preds, labels=species)

    hover_text = []
    for i in range(len(species)):
        row = []
        for j in range(len(species)):
            total = cm[i].sum()
            pct = cm[i][j] / total * 100 if total > 0 else 0
            row.append(f"Actual: {species[i]}<br>Predicted: {species[j]}<br>Count: {cm[i][j]}<br>({pct:.1f}%)")
        hover_text.append(row)

    fig = go.Figure(go.Heatmap(
        z=cm,
        x=species,
        y=species,
        colorscale="Blues",
        text=cm,
        texttemplate="%{text}",
        hovertext=hover_text,
        hoverinfo="text",
        showscale=True
    ))

    fig.update_layout(
        title="Confusion Matrix (Test Set)",
        xaxis_title="Predicted Species",
        yaxis_title="Actual Species",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=300,
        margin=dict(l=20, r=20, t=50, b=40)
    )

    return fig.to_json()


def build_gap_plot(lr_info):
    # train vs test gap - copied from decision_tree.py, same logic
    # LR usually has smaller gap than trees since its less prone to overfitting

    tr_acc = round(lr_info["train_acc"] * 100, 2)
    te_acc = round(lr_info["test_acc"] * 100, 2)
    gap = round(tr_acc - te_acc, 2)

    fig = go.Figure()

    fig.add_trace(go.Bar(
        name="Train Accuracy",
        x=["Accuracy"], y=[tr_acc],
        marker_color="#4C72B0",
        text=[f"{tr_acc}%"],
        textposition="outside",
        hovertemplate=f"Train: {tr_acc}%<extra></extra>"
    ))

    fig.add_trace(go.Bar(
        name="Test Accuracy",
        x=["Accuracy"], y=[te_acc],
        marker_color="#55A868",
        text=[f"{te_acc}%"],
        textposition="outside",
        hovertemplate=f"Test: {te_acc}%<extra></extra>"
    ))

    title_color = "#C44E52" if gap > 5 else "#27ae60"
    title_text = f"Gap: {gap}% {'(possible overfitting)' if gap > 5 else '(looks fine)'}"

    fig.update_layout(
        title=f"Train vs Test Accuracy — {title_text}",
        title_font_color=title_color,
        yaxis_title="Accuracy (%)",
        barmode="group",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=300,
        margin=dict(l=20, r=20, t=50, b=40),
        yaxis=dict(range=[0, 110]),
        legend=dict(orientation="h", y=-0.25)
    )

    return fig.to_json()