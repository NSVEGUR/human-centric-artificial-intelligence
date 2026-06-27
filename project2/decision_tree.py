import numpy as np
import plotly.graph_objects as go
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from .data import load_and_preprocess

# ── data cache ────────────────────────────────────────────────────────────────
_data_cache = None

def _get_data():
    global _data_cache
    if _data_cache is None:
        X_train, X_test, y_train, y_test, _, _ = load_and_preprocess()
        _data_cache = (X_train, X_test, y_train, y_test)
    return _data_cache

# ── dark theme - matches project 3 style ─────────────────────────────────────
PLOT_BG    = "#1a1d2e"
PAPER_BG   = "rgba(0,0,0,0)"   # transparent - page bg shows through
TEXT       = "#c9d1e0"
GRID       = "#252840"
AXIS       = "#353860"

BLUE   = "#4a7fe5"   # split nodes / adelie
ORANGE = "#f5a623"   # train line / chinstrap
GREEN  = "#3dba6e"   # leaf nodes / gentoo
RED    = "#ff4d6d"   # selected star / overfitting

SPECIES        = ["Adelie", "Chinstrap", "Gentoo"]
SPECIES_COLORS = {"Adelie": BLUE, "Chinstrap": ORANGE, "Gentoo": GREEN}

# ── model cache ───────────────────────────────────────────────────────────────
_all_trees = None

def _train_all_trees():
    global _all_trees
    if _all_trees is not None:
        return
    X_train, X_test, y_train, y_test = _get_data()
    trees = []
    for n in range(2, 31):
        m = DecisionTreeClassifier(max_leaf_nodes=n, random_state=42)
        m.fit(X_train, y_train)
        trees.append({
            "model":     m,
            "n_leaves":  m.get_n_leaves(),
            "train_acc": m.score(X_train, y_train),
            "test_acc":  m.score(X_test,  y_test),
        })
    _all_trees = trees

def get_best_tree(lam):
    _train_all_trees()
    max_leaves = max(t["n_leaves"] for t in _all_trees)
    best, best_score = None, float("inf")
    for t in _all_trees:
        score = (1 - t["test_acc"]) + lam * (t["n_leaves"] / max_leaves)
        if score < best_score:
            best_score, best = score, t
    return best

# ── shared layout ─────────────────────────────────────────────────────────────
def _base_layout(height=320, margin=None):
    m = margin or dict(l=50, r=20, t=30, b=50)
    return dict(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        height=height,
        margin=m,
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif", size=12),
    )

# ── decision tree diagram ─────────────────────────────────────────────────────
def build_tree_plotly(tree_info):
    """
    Renders the decision tree as a clean node-edge graph.
    Labels are Plotly annotations so they never clip or overflow.
    """
    _train_all_trees()
    X_train, _, _, _ = _get_data()

    model      = tree_info["model"]
    sk_tree    = model.tree_
    n_nodes    = sk_tree.node_count
    feat       = sk_tree.feature
    thr        = sk_tree.threshold
    ch_left    = sk_tree.children_left
    ch_right   = sk_tree.children_right
    vals       = sk_tree.value
    feat_names = list(X_train.columns)
    classes    = model.classes_
    is_leaf    = ch_left == -1

    # ── tree layout: recursive, depth-first ──────────────────────────────────
    pos_x = [0.0] * n_nodes
    pos_y = [0.0] * n_nodes

    def _layout(node, x0, y, w):
        pos_x[node] = x0 + w / 2
        pos_y[node] = -y
        l, r = ch_left[node], ch_right[node]
        if l != -1:
            _layout(l, x0,       y + 1, w / 2)
            _layout(r, x0 + w/2, y + 1, w / 2)

    _layout(0, 0, 0, 2 ** model.get_depth())

    # ── edges ────────────────────────────────────────────────────────────────
    ex, ey = [], []
    for nd in range(n_nodes):
        for ch in [ch_left[nd], ch_right[nd]]:
            if ch != -1:
                ex += [pos_x[nd], pos_x[ch], None]
                ey += [pos_y[nd], pos_y[ch], None]

    trace_edges = go.Scatter(
        x=ex, y=ey, mode="lines",
        line=dict(color="#3a4060", width=1.5),
        hoverinfo="none",
    )

    # ── node markers (two separate traces for legend) ─────────────────────────
    sx, sy, sh = [], [], []   # split nodes
    lx, ly, lh = [], [], []   # leaf nodes

    for nd in range(n_nodes):
        majority = classes[int(np.argmax(vals[nd][0]))]
        count    = int(vals[nd][0].sum())
        if is_leaf[nd]:
            lx.append(pos_x[nd]); ly.append(pos_y[nd])
            lh.append(f"<b>Leaf → {majority}</b><br>Samples: {count}")
        else:
            fname = feat_names[feat[nd]] if feat[nd] < len(feat_names) else "?"
            sx.append(pos_x[nd]); sy.append(pos_y[nd])
            sh.append(f"<b>{fname} ≤ {thr[nd]:.2f}</b><br>Samples: {count}")

    trace_splits = go.Scatter(
        x=sx, y=sy, mode="markers", name="Decision node",
        marker=dict(size=18, color=BLUE, symbol="square",
                    line=dict(width=2, color="#7aaeff")),
        customdata=sh,
        hovertemplate="%{customdata}<extra></extra>",
    )
    trace_leaves = go.Scatter(
        x=lx, y=ly, mode="markers", name="Leaf node",
        marker=dict(size=18, color=GREEN, symbol="circle",
                    line=dict(width=2, color="#7adba0")),
        customdata=lh,
        hovertemplate="%{customdata}<extra></extra>",
    )

    # ── annotations: always-visible labels above each node ───────────────────
    # Human-readable short names
    SHORT = {
        "bill_length_mm":    "bill len",
        "bill_depth_mm":     "bill dep",
        "flipper_length_mm": "flipper",
        "body_mass_g":       "mass (g)",
        "island_Biscoe":     "Biscoe?",
        "island_Dream":      "Dream?",
        "island_Torgersen":  "Torgersen?",
        "sex_Male":          "Male?",
        "sex_Female":        "Female?",
    }

    annotations = []
    for nd in range(n_nodes):
        majority = classes[int(np.argmax(vals[nd][0]))]
        count    = int(vals[nd][0].sum())

        if is_leaf[nd]:
            label = f"<b>{majority}</b><br><span style='font-size:8px'>{count} samples</span>"
            bg    = "rgba(30,60,40,0.88)"
            bc    = GREEN
        else:
            fname = feat_names[feat[nd]] if feat[nd] < len(feat_names) else "?"
            short = SHORT.get(fname, fname.replace("_", " "))
            label = f"<b>{short}</b><br><span style='font-size:8px'>≤ {thr[nd]:.2f}</span>"
            bg    = "rgba(20,35,80,0.88)"
            bc    = BLUE

        annotations.append(dict(
            x=pos_x[nd], y=pos_y[nd],
            text=label,
            showarrow=False,
            font=dict(size=9, color="white", family="Inter, monospace"),
            bgcolor=bg,
            bordercolor=bc,
            borderwidth=1,
            borderpad=3,
            xanchor="center",
            yanchor="middle",
        ))

    layout = _base_layout(height=500, margin=dict(l=10, r=10, t=20, b=30))
    layout.update(
        showlegend=True,
        legend=dict(
            orientation="h", x=0.5, xanchor="center", y=-0.04,
            font=dict(color=TEXT, size=11), bgcolor="rgba(0,0,0,0)",
        ),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=annotations,
    )

    return go.Figure(data=[trace_edges, trace_splits, trace_leaves], layout=layout).to_json()


# ── accuracy vs complexity tradeoff ──────────────────────────────────────────
def build_tradeoff_plot(selected_tree):
    _train_all_trees()
    x     = [t["n_leaves"]        for t in _all_trees]
    ytest = [t["test_acc"]  * 100 for t in _all_trees]
    ytrain= [t["train_acc"] * 100 for t in _all_trees]

    traces = [
        go.Scatter(x=x, y=ytrain, mode="lines+markers", name="Train",
                   line=dict(color=ORANGE, width=2, dash="dot"), marker=dict(size=4),
                   hovertemplate="Leaves: %{x}<br>Train: %{y:.1f}%<extra></extra>"),
        go.Scatter(x=x, y=ytest, mode="lines+markers", name="Test",
                   line=dict(color=BLUE, width=2), marker=dict(size=4),
                   hovertemplate="Leaves: %{x}<br>Test: %{y:.1f}%<extra></extra>"),
        go.Scatter(
            x=[selected_tree["n_leaves"]], y=[selected_tree["test_acc"] * 100],
            mode="markers", name="Selected",
            marker=dict(size=14, color=RED, symbol="star"),
            hovertemplate=(
                f"<b>Selected</b><br>Leaves: {selected_tree['n_leaves']}"
                f"<br>Test: {selected_tree['test_acc']*100:.1f}%<extra></extra>"
            ),
        ),
    ]

    layout = _base_layout(height=300)
    layout.update(
        xaxis=dict(title="Leaves (complexity)", color=TEXT,
                   gridcolor=GRID, linecolor=AXIS, tickcolor=TEXT),
        yaxis=dict(title="Accuracy (%)", range=[60, 102], color=TEXT,
                   gridcolor=GRID, linecolor=AXIS, tickcolor=TEXT),
        legend=dict(orientation="h", y=-0.35, font=dict(color=TEXT),
                    bgcolor="rgba(0,0,0,0)"),
    )
    return go.Figure(data=traces, layout=layout).to_json()


# ── confusion matrix ──────────────────────────────────────────────────────────
def build_confusion_plot(tree_info):
    _, X_test, _, y_test = _get_data()
    y_pred = tree_info["model"].predict(X_test)
    cm     = confusion_matrix(y_test, y_pred, labels=SPECIES)
    text   = [[str(cm[i][j]) for j in range(3)] for i in range(3)]

    heatmap = go.Heatmap(
        z=cm.tolist(), x=SPECIES, y=SPECIES,
        colorscale=[[0, PLOT_BG], [1, BLUE]],
        text=text, texttemplate="<b>%{text}</b>",
        textfont=dict(size=18, color="white"),
        hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
        showscale=False,
    )

    layout = _base_layout(height=300, margin=dict(l=80, r=20, t=20, b=70))
    layout.update(
        xaxis=dict(title="Predicted", color=TEXT, tickfont=dict(color=TEXT), side="bottom"),
        yaxis=dict(title="Actual",    color=TEXT, tickfont=dict(color=TEXT), autorange="reversed"),
    )
    return go.Figure(data=[heatmap], layout=layout).to_json()


# ── train vs test gap ─────────────────────────────────────────────────────────
def build_gap_plot(tree_info):
    tr = round(tree_info["train_acc"] * 100, 2)
    te = round(tree_info["test_acc"]  * 100, 2)
    gap = round(tr - te, 2)

    traces = [
        go.Bar(name="Train", x=[""], y=[tr], marker_color=ORANGE,
               text=[f"{tr}%"], textposition="outside", textfont=dict(color=TEXT),
               hovertemplate=f"Train: {tr}%<extra></extra>"),
        go.Bar(name="Test",  x=[""], y=[te], marker_color=BLUE,
               text=[f"{te}%"], textposition="outside", textfont=dict(color=TEXT),
               hovertemplate=f"Test: {te}%<extra></extra>"),
    ]

    gc    = RED if gap > 5 else GREEN
    glabel = f"Gap: {gap}%  {'⚠ overfitting' if gap > 5 else '✓ good fit'}"

    layout = _base_layout(height=300)
    layout.update(
        barmode="group",
        yaxis=dict(range=[0, 115], title="Accuracy (%)", color=TEXT,
                   gridcolor=GRID, tickcolor=TEXT),
        xaxis=dict(color=TEXT, tickcolor=TEXT),
        legend=dict(orientation="h", y=-0.35, font=dict(color=TEXT),
                    bgcolor="rgba(0,0,0,0)"),
        annotations=[dict(
            text=glabel, x=0.5, y=1.06,
            xref="paper", yref="paper", showarrow=False,
            font=dict(color=gc, size=13),
        )],
    )
    return go.Figure(data=traces, layout=layout).to_json()
