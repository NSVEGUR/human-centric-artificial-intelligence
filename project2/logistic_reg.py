import numpy as np
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression
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

# ── dark theme (shared with decision_tree.py) ─────────────────────────────────
PLOT_BG = "#1a1d2e"
PAPER_BG = "rgba(0,0,0,0)"
TEXT  = "#c9d1e0"
GRID  = "#252840"
AXIS  = "#353860"

BLUE   = "#4a7fe5"
ORANGE = "#f5a623"
GREEN  = "#3dba6e"
RED    = "#ff4d6d"

SPECIES        = ["Adelie", "Chinstrap", "Gentoo"]
SPECIES_COLORS = {"Adelie": BLUE, "Chinstrap": ORANGE, "Gentoo": GREEN}

# ── model cache ───────────────────────────────────────────────────────────────
_all_lr_models = None

def _train_all_lr():
    global _all_lr_models
    if _all_lr_models is not None:
        return
    X_train, X_test, y_train, y_test = _get_data()
    C_values = [0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]
    models = []
    for C in C_values:
        m = LogisticRegression(C=C, max_iter=1000, random_state=42)
        m.fit(X_train, y_train)
        omega = float(np.sum(np.abs(m.coef_)))
        models.append({
            "model":     m,
            "C":         C,
            "omega":     omega,
            "train_acc": m.score(X_train, y_train),
            "test_acc":  m.score(X_test,  y_test),
        })
    _all_lr_models = models

def get_best_lr(lam):
    _train_all_lr()
    # Sort ascending by omega (simplest first). lam=0 → most complex, lam=1 → simplest.
    # Direct mapping avoids the tie-breaking collapse that occurs when all models
    # have identical test accuracy (Palmer Penguins is linearly separable).
    sorted_models = sorted(_all_lr_models, key=lambda m: m["omega"])
    idx = round((1 - lam) * (len(sorted_models) - 1))
    return sorted_models[idx]

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

# ── coefficient bar chart ─────────────────────────────────────────────────────
def build_lr_plotly(lr_info, sort_by="magnitude", selected_class=None, selected_feature=None):
    """
    Bar chart showing coefficient weights per feature per species.
    Tall positive bar = pushes towards that species prediction.
    """
    _train_all_lr()
    X_train, _, _, _ = _get_data()

    model       = lr_info["model"]
    class_names = list(model.classes_)
    feat_names  = list(X_train.columns)   # correct order - same as training

    classes_to_show = (
        [selected_class] if selected_class and selected_class in class_names
        else class_names
    )

    fig = go.Figure()

    for cls in classes_to_show:
        idx   = class_names.index(cls)
        coefs = model.coef_[idx]
        pairs = list(zip(feat_names, coefs))

        if selected_feature:
            pairs = [(f, c) for f, c in pairs if selected_feature.lower() in f.lower()]

        if sort_by == "magnitude":
            pairs.sort(key=lambda x: abs(x[1]), reverse=True)
        else:
            pairs.sort(key=lambda x: x[0])

        labels = [p[0].replace("_mm","").replace("_g","").replace("_"," ") for p in pairs]
        coef_v = [p[1] for p in pairs]
        colors = [SPECIES_COLORS.get(cls, BLUE)] * len(coef_v)

        fig.add_trace(go.Bar(
            name=cls,
            x=labels,
            y=coef_v,
            marker_color=SPECIES_COLORS.get(cls, BLUE),
            hovertemplate="<b>%{x}</b><br>Coefficient: %{y:.3f}<extra>" + cls + "</extra>",
        ))

    layout = _base_layout(height=420)
    layout.update(
        barmode="group",
        xaxis=dict(title="Feature", color=TEXT, tickfont=dict(color=TEXT, size=10),
                   tickangle=-30, linecolor=AXIS, gridcolor=GRID),
        yaxis=dict(title="Coefficient weight", color=TEXT, tickfont=dict(color=TEXT),
                   gridcolor=GRID, zerolinecolor=AXIS, zerolinewidth=1.5),
        legend=dict(font=dict(color=TEXT), bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_layout(**layout)
    return fig.to_json()


# ── tradeoff plot ─────────────────────────────────────────────────────────────
def build_tradeoff_plot(selected_lr):
    _train_all_lr()
    x      = [m["omega"]           for m in _all_lr_models]
    ytest  = [m["test_acc"]  * 100 for m in _all_lr_models]
    ytrain = [m["train_acc"] * 100 for m in _all_lr_models]

    traces = [
        go.Scatter(x=x, y=ytrain, mode="lines+markers", name="Train",
                   line=dict(color=ORANGE, width=2, dash="dot"), marker=dict(size=4),
                   hovertemplate="‖w‖₁: %{x:.2f}<br>Train: %{y:.1f}%<extra></extra>"),
        go.Scatter(x=x, y=ytest, mode="lines+markers", name="Test",
                   line=dict(color=BLUE, width=2), marker=dict(size=4),
                   hovertemplate="‖w‖₁: %{x:.2f}<br>Test: %{y:.1f}%<extra></extra>"),
        go.Scatter(
            x=[selected_lr["omega"]], y=[selected_lr["test_acc"] * 100],
            mode="markers", name="Selected",
            marker=dict(size=14, color=RED, symbol="star"),
            hovertemplate=(
                f"<b>Selected</b><br>‖w‖₁: {selected_lr['omega']:.2f}"
                f"<br>Test: {selected_lr['test_acc']*100:.1f}%<extra></extra>"
            ),
        ),
    ]

    layout = _base_layout(height=300)
    layout.update(
        xaxis=dict(title="‖w‖₁ (complexity)", color=TEXT,
                   gridcolor=GRID, linecolor=AXIS, tickcolor=TEXT),
        yaxis=dict(title="Accuracy (%)", range=[60, 102], color=TEXT,
                   gridcolor=GRID, linecolor=AXIS, tickcolor=TEXT),
        legend=dict(orientation="h", y=-0.35, font=dict(color=TEXT),
                    bgcolor="rgba(0,0,0,0)"),
    )
    return go.Figure(data=traces, layout=layout).to_json()


# ── confusion matrix ──────────────────────────────────────────────────────────
def build_confusion_plot(lr_info):
    _, X_test, _, y_test = _get_data()
    y_pred = lr_info["model"].predict(X_test)
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
def build_gap_plot(lr_info):
    tr  = round(lr_info["train_acc"] * 100, 2)
    te  = round(lr_info["test_acc"]  * 100, 2)
    gap = round(tr - te, 2)

    traces = [
        go.Bar(name="Train", x=[""], y=[tr], marker_color=ORANGE,
               text=[f"{tr}%"], textposition="outside", textfont=dict(color=TEXT),
               hovertemplate=f"Train: {tr}%<extra></extra>"),
        go.Bar(name="Test",  x=[""], y=[te], marker_color=BLUE,
               text=[f"{te}%"], textposition="outside", textfont=dict(color=TEXT),
               hovertemplate=f"Test: {te}%<extra></extra>"),
    ]

    gc     = RED if gap > 5 else GREEN
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