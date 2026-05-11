from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import plotly.graph_objects as go
import numpy as np

from .data import load_and_preprocess

# load data at import time so we dont redo this on every request
train_X, test_X, train_y, test_y, mad_vals, num_cols = load_and_preprocess()

# going up to 30 leaves, not sure if thats enough but seems fine for this dataset
all_trees = []

for num_leaves in range(2, 31):
    model = DecisionTreeClassifier(max_leaf_nodes=num_leaves, random_state=42)
    model.fit(train_X, train_y)

    test_preds = model.predict(test_X)
    test_acc = accuracy_score(test_y, test_preds)

    train_preds = model.predict(train_X)
    train_acc = accuracy_score(train_y, train_preds)

    # get_n_leaves() sometimes gives less than num_leaves
    # happens when the tree cant find useful splits anymore
    actual_leaves = model.get_n_leaves()

    all_trees.append({
        "model": model,
        "n_leaves": actual_leaves,
        "test_acc": test_acc,
        "train_acc": train_acc,
    })

# need min and max for normalization later
biggest_tree = max(t["n_leaves"] for t in all_trees)
smallest_tree = min(t["n_leaves"] for t in all_trees)


def get_best_tree(lam):
    # picks the tree that scores lowest on our objective
    #
    # task sheet says minimize: acctest + lambda * Omega
    # but minimizing accuracy makes no sense so acctest must mean error rate
    # i think they meant (1 - accuracy) which is the 0-1 loss averaged over samples
    # thats what the lecture formula actually says anyway
    #
    # score = (1 - test_acc) + lambda * norm_leaves
    # lam=0 just picks most accurate, higher lam starts preferring simpler trees

    winner = None
    lowest = float("inf")

    for t in all_trees:
        # normalize to 0-1 so the leaf count doesnt completely overpower the error term
        norm_leaves = (t["n_leaves"] - smallest_tree) / (biggest_tree - smallest_tree + 1e-9)
        score = (1 - t["test_acc"]) + lam * norm_leaves

        if score < lowest:
            lowest = score
            winner = t

    return winner


def build_tree_plotly(tree_info):
    # builds the interactive tree visualization
    # had to do this manually because sklearn doesnt have a plotly export
    # the basic idea: traverse the tree recursively, assign x,y coords to each node

    model = tree_info["model"]
    sk_tree = model.tree_
    feat_names = list(train_X.columns)
    species = model.classes_

    pos_x = {}
    pos_y = {}
    labels = {}
    line_x = []
    line_y = []

    # this needs to be a list not an int
    # otherwise the nested function cant modify it (python closure thing)
    pos_counter = [0]

    def place_node(node_id, depth):
        left_child = sk_tree.children_left[node_id]
        right_child = sk_tree.children_right[node_id]

        if left_child == -1:
            # leaf - assign next x position
            pos_x[node_id] = pos_counter[0]
            pos_y[node_id] = -depth
            pos_counter[0] += 1
        else:
            # internal node - do children first, then center parent between them
            place_node(left_child, depth + 1)
            place_node(right_child, depth + 1)
            pos_x[node_id] = (pos_x[left_child] + pos_x[right_child]) / 2
            pos_y[node_id] = -depth

        if left_child == -1:
            best_class = np.argmax(sk_tree.value[node_id][0])
            labels[node_id] = f"class: {species[best_class]}<br>samples: {sk_tree.n_node_samples[node_id]}"
        else:
            feat = feat_names[sk_tree.feature[node_id]]
            thresh = round(sk_tree.threshold[node_id], 2)
            labels[node_id] = f"{feat} <= {thresh}<br>samples: {sk_tree.n_node_samples[node_id]}"

    place_node(0, 0)

    # build edge lines - None separates each line segment so plotly doesnt join them
    for node_id in range(sk_tree.node_count):
        lc = sk_tree.children_left[node_id]
        rc = sk_tree.children_right[node_id]
        if lc != -1:
            line_x += [pos_x[node_id], pos_x[lc], None]
            line_y += [pos_y[node_id], pos_y[lc], None]
            line_x += [pos_x[node_id], pos_x[rc], None]
            line_y += [pos_y[node_id], pos_y[rc], None]

    leaves   = [i for i in range(sk_tree.node_count) if sk_tree.children_left[i] == -1]
    internal = [i for i in range(sk_tree.node_count) if sk_tree.children_left[i] != -1]

    fig = go.Figure()

    # edges first so they go behind the nodes
    fig.add_trace(go.Scatter(
        x=line_x, y=line_y,
        mode="lines",
        line=dict(color="#aaa", width=1.5),
        hoverinfo="none",
        showlegend=False
    ))

    # decision nodes - blue squares
    fig.add_trace(go.Scatter(
        x=[pos_x[i] for i in internal],
        y=[pos_y[i] for i in internal],
        mode="markers+text",
        marker=dict(size=18, color="#4C72B0", symbol="square"),
        text=[labels[i] for i in internal],
        textposition="top center",
        hovertext=[labels[i] for i in internal],
        hoverinfo="text",
        name="Decision Node",
        textfont=dict(size=8)
    ))

    # leaf nodes - green circles
    # size 18 matches the decision nodes, looked weird when they were different
    fig.add_trace(go.Scatter(
        x=[pos_x[i] for i in leaves],
        y=[pos_y[i] for i in leaves],
        mode="markers+text",
        marker=dict(size=18, color="#55A868", symbol="circle"),
        text=[labels[i] for i in leaves],
        textposition="top center",
        hovertext=[labels[i] for i in leaves],
        hoverinfo="text",
        name="Leaf Node",
        textfont=dict(size=8)
    ))

    fig.update_layout(
        title=f"Decision Tree — Leaves: {tree_info['n_leaves']} | Test Acc: {tree_info['test_acc']:.2%} | Train Acc: {tree_info['train_acc']:.2%}",
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=600,
        margin=dict(l=20, r=20, t=60, b=20),  # b=20 might need more if labels get cut off
        legend=dict(orientation="h", y=-0.05)
    )

    return fig.to_json()


def build_tradeoff_plot(selected_tree):
    # shows all trained trees plotted as accuracy vs number of leaves
    # red star = currently selected model
    # useful for seeing where the accuracy starts to plateau

    leaf_counts = [t["n_leaves"] for t in all_trees]
    test_accs = [t["test_acc"] * 100 for t in all_trees]
    train_accs = [t["train_acc"] * 100 for t in all_trees]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=leaf_counts, y=train_accs,
        mode="lines+markers",
        name="Train Accuracy",
        line=dict(color="#4C72B0", width=2),
        marker=dict(size=6),
        hovertemplate="Leaves: %{x}<br>Train Acc: %{y:.2f}%<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=leaf_counts, y=test_accs,
        mode="lines+markers",
        name="Test Accuracy",
        line=dict(color="#55A868", width=2),
        marker=dict(size=6),
        hovertemplate="Leaves: %{x}<br>Test Acc: %{y:.2f}%<extra></extra>"
    ))

    fig.add_trace(go.Scatter(
        x=[selected_tree["n_leaves"]],
        y=[selected_tree["test_acc"] * 100],
        mode="markers",
        name="Selected Model",
        marker=dict(size=14, color="#C44E52", symbol="star"),
        hovertemplate=f"Selected — Leaves: {selected_tree['n_leaves']}<br>Test Acc: {selected_tree['test_acc']:.2%}<extra></extra>"
    ))

    fig.update_layout(
        title="Accuracy vs Complexity Tradeoff",
        xaxis_title="Number of Leaves (Complexity)",
        yaxis_title="Accuracy (%)",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=300,
        margin=dict(l=20, r=20, t=50, b=40),
        legend=dict(orientation="h", y=-0.25),
        yaxis=dict(range=[50, 102])
    )

    return fig.to_json()


def build_confusion_plot(tree_info):
    # confusion matrix heatmap
    # copied the hover text logic from build_gap_plot and adapted it
    # rows = actual, columns = predicted, diagonal = correct

    model = tree_info["model"]
    preds = model.predict(test_X)
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


def build_gap_plot(tree_info):
    # bar chart for train vs test accuracy
    # 5% gap threshold for overfitting - not sure what the right number is but 5 felt ok

    tr_acc = round(tree_info["train_acc"] * 100, 2)
    te_acc = round(tree_info["test_acc"] * 100, 2)
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