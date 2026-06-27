import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import load_and_preprocess

# load once and cache - avoids reprocessing on every import
_raw = load_and_preprocess()
train_X, test_X, train_y, test_y, mad_vals, num_cols = _raw

# Feature display names
FEATURE_DISPLAY_NAMES = {
    'bill_length_mm': 'Bill Length (mm)',
    'bill_depth_mm': 'Bill Depth (mm)',
    'flipper_length_mm': 'Flipper Length (mm)',
    'body_mass_g': 'Body Mass (g)',
}

# (species colours defined below with dark theme)


def compute_pdp(model, feature_name, model_type, n_grid=50, use_train=True):
    """
    Compute Partial Dependence Plot values for a single feature.

    PDP formula:
        f_s(x_s) = (1/n) * sum_i f(x_s, x_c^(i))

    where x_s is the feature of interest and x_c are complementary features.

    Args:
        model: trained sklearn model (DecisionTreeClassifier or LogisticRegression)
        feature_name: name of feature to analyze
        model_type: 'tree' or 'lr'
        n_grid: number of grid points
        use_train: whether to use training data (True) or test data

    Returns:
        dict with 'grid', 'pdp_values' (per class), 'ice_curves' (per sample per class)
    """
    X = train_X if use_train else test_X

    # Create grid of feature values
    feature_idx = list(X.columns).index(feature_name)
    feature_min = X[feature_name].min()
    feature_max = X[feature_name].max()
    grid = np.linspace(feature_min, feature_max, n_grid)

    class_names = list(model.classes_)
    n_classes = len(class_names)
    n_samples = len(X)

    # Store ICE curves: shape (n_samples, n_grid, n_classes)
    ice_curves = np.zeros((n_samples, n_grid, n_classes))

    # For each grid point
    for g_idx, grid_val in enumerate(grid):
        # Create modified dataset with feature set to grid value
        X_modified = X.copy()
        X_modified[feature_name] = grid_val

        # LR is now trained on raw features - no scaler needed
        proba = model.predict_proba(X_modified)  # shape (n_samples, n_classes)

        # Store for each sample
        ice_curves[:, g_idx, :] = proba

    # Compute PDP as mean of ICE curves
    pdp_values = ice_curves.mean(axis=0)  # shape (n_grid, n_classes)

    return {
        'grid': grid.tolist(),
        'pdp_values': {
            class_names[c]: pdp_values[:, c].tolist()
            for c in range(n_classes)
        },
        'ice_curves': {
            class_names[c]: ice_curves[:, :, c].tolist()
            for c in range(n_classes)
        },
        'feature_name': feature_name,
        'feature_display': FEATURE_DISPLAY_NAMES.get(feature_name, feature_name),
    }


def compute_ale(model, feature_name, model_type, n_intervals=40, use_train=True):
    """
    Compute Accumulated Local Effects for a single feature.

    ALE is computed as:
    1. Divide feature range into intervals (bins)
    2. For each interval, compute average difference in predictions at boundaries
    3. Accumulate effects and center around zero

    For partial derivatives:
    - Logistic Regression: Exact derivative via chain rule
    - Decision Tree: Finite difference approximation

    Args:
        model: trained sklearn model
        feature_name: name of feature to analyze
        model_type: 'tree' or 'lr'
        n_intervals: number of intervals/bins
        use_train: whether to use training data

    Returns:
        dict with 'grid', 'ale_values' (per class), 'derivative_type'
    """
    X = train_X if use_train else test_X
    feature_col = X[feature_name].values
    feature_idx = list(X.columns).index(feature_name)

    class_names = list(model.classes_)
    n_classes = len(class_names)

    # Create quantile-based bins (handles non-uniform distributions better)
    percentiles = np.linspace(0, 100, n_intervals + 1)
    bin_edges = np.percentile(feature_col, percentiles)

    # Remove duplicate edges (can happen with discrete-ish data)
    bin_edges = np.unique(bin_edges)
    n_bins = len(bin_edges) - 1

    if n_bins < 2:
        # Not enough unique values
        return {
            'grid': [float(feature_col.min()), float(feature_col.max())],
            'ale_values': {c: [0.0, 0.0] for c in class_names},
            'feature_name': feature_name,
            'feature_display': FEATURE_DISPLAY_NAMES.get(feature_name, feature_name),
            'derivative_type': 'exact' if model_type == 'lr' else 'finite_difference',
            'error': 'Not enough unique values for ALE computation',
        }

    # For each bin, collect samples and compute local effects
    bin_effects = np.zeros((n_bins, n_classes))
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        left_edge = bin_edges[i]
        right_edge = bin_edges[i + 1]

        # Find samples in this bin
        if i == n_bins - 1:
            in_bin = (feature_col >= left_edge) & (feature_col <= right_edge)
        else:
            in_bin = (feature_col >= left_edge) & (feature_col < right_edge)

        samples_in_bin = X[in_bin]
        n_in_bin = len(samples_in_bin)

        if n_in_bin == 0:
            continue

        bin_counts[i] = n_in_bin

        # Compute effect as f(right_edge) - f(left_edge) averaged over samples in bin
        if model_type == 'lr':
            # LR trained on raw features - exact finite difference, same as tree
            effects = compute_lr_local_effect_exact(
                model, samples_in_bin, feature_idx, left_edge, right_edge
            )
        else:
            # For decision tree, use finite difference
            effects = compute_tree_local_effect(
                model, samples_in_bin, feature_idx, left_edge, right_edge
            )

        bin_effects[i, :] = effects

    # Accumulate effects
    ale_values = np.zeros((n_bins + 1, n_classes))
    for i in range(n_bins):
        ale_values[i + 1, :] = ale_values[i, :] + bin_effects[i, :]

    # Center: subtract weighted mean
    # Weight by number of samples in each bin
    total_samples = bin_counts.sum()
    if total_samples > 0:
        # Compute weighted mean of ALE values at bin centers
        bin_centers_ale = (ale_values[:-1, :] + ale_values[1:, :]) / 2
        weights = bin_counts / total_samples
        weighted_mean = np.sum(bin_centers_ale * weights[:, np.newaxis], axis=0)
        ale_values = ale_values - weighted_mean

    return {
        'grid': bin_edges.tolist(),
        'ale_values': {
            class_names[c]: ale_values[:, c].tolist()
            for c in range(n_classes)
        },
        'feature_name': feature_name,
        'feature_display': FEATURE_DISPLAY_NAMES.get(feature_name, feature_name),
        'derivative_type': 'exact (analytical)' if model_type == 'lr' else 'finite difference (discretization)',
        'bin_counts': bin_counts.tolist(),
    }


def compute_lr_local_effect_exact(model, samples, feature_idx, left_val, right_val):
    """
    Compute local effect for logistic regression using finite differences at boundaries.
    LR is now trained on raw (unscaled) features, so no scaler transform needed.
    effect = mean over samples of [f(x_right) - f(x_left)]
    """
    n_samples = len(samples)
    n_classes = len(model.classes_)

    if n_samples == 0:
        return np.zeros(n_classes)

    feature_col = samples.columns[feature_idx]
    effects = np.zeros(n_classes)

    for i in range(n_samples):
        row = samples.iloc[[i]].copy()

        row[feature_col] = left_val
        proba_left = model.predict_proba(row)[0]

        row[feature_col] = right_val
        proba_right = model.predict_proba(row)[0]

        effects += (proba_right - proba_left)

    effects /= n_samples
    return effects


def compute_tree_local_effect(model, samples, feature_idx, left_val, right_val):
    """
    Compute local effect for decision tree using finite differences.

    Decision trees are piecewise constant, so the derivative is technically undefined
    at the split points and zero elsewhere. We approximate by:
    effect = f(x_right) - f(x_left)

    This is the finite difference approximation mentioned in the task.
    """
    n_samples = len(samples)
    n_classes = len(model.classes_)

    if n_samples == 0:
        return np.zeros(n_classes)

    feature_col = samples.columns[feature_idx]
    effects = np.zeros(n_classes)

    for i in range(n_samples):
        row = samples.iloc[[i]].copy()

        row[feature_col] = left_val
        proba_left = model.predict_proba(row)[0]

        row[feature_col] = right_val
        proba_right = model.predict_proba(row)[0]

        effects += (proba_right - proba_left)

    effects /= n_samples
    return effects


def compute_feature_importance_permutation(model, feature_name, model_type, n_repeats=10):
    """
    Compute permutation feature importance.
    Shuffle the feature values and measure decrease in accuracy.
    LR now uses raw features - no scaler needed.
    """
    from sklearn.metrics import accuracy_score

    X = test_X.copy()
    y = test_y

    # baseline accuracy - both model types predict on raw features
    baseline_preds = model.predict(X)
    baseline_acc = accuracy_score(y, baseline_preds)

    # permute the chosen feature and measure accuracy drop
    importances = []
    for _ in range(n_repeats):
        X_permuted = X.copy()
        X_permuted[feature_name] = np.random.permutation(X_permuted[feature_name].values)

        perm_preds = model.predict(X_permuted)
        perm_acc = accuracy_score(y, perm_preds)
        importances.append(baseline_acc - perm_acc)

    return {
        'mean_importance': float(np.mean(importances)),
        'std_importance': float(np.std(importances)),
        'baseline_accuracy': float(baseline_acc),
    }


# ── dark theme ────────────────────────────────────────────────────────────────
PLOT_BG  = "#1a1d2e"
PAPER_BG = "rgba(0,0,0,0)"
TEXT     = "#c9d1e0"
GRID     = "#252840"
AXIS     = "#353860"

# override species colours to match the rest of project 2
SPECIES_COLORS = {
    "Adelie":    "#4a7fe5",
    "Chinstrap": "#f5a623",
    "Gentoo":    "#3dba6e",
}


def _dark_layout(height=400, margin=None):
    m = margin or dict(l=55, r=25, t=50, b=55)
    return dict(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif", size=12),
        height=height,
        margin=m,
    )


def _axis_style(**extra):
    return dict(color=TEXT, tickfont=dict(color=TEXT), gridcolor=GRID,
                linecolor=AXIS, zerolinecolor=AXIS, **extra)


def _rug(feature_name, y_val, row=None, col=None):
    """Tiny tick marks at the bottom showing data distribution."""
    fv = train_X[feature_name].values
    kw = dict(row=row, col=col) if row else {}
    return go.Scatter(
        x=fv, y=[y_val] * len(fv),
        mode="markers",
        marker=dict(symbol="line-ns", size=7,
                    color="rgba(180,190,210,0.35)", line=dict(width=1)),
        showlegend=False, hoverinfo="skip",
    ), kw


# ── PDP (standalone) ──────────────────────────────────────────────────────────
def build_pdp_plot(pdp_data, show_ice=False, n_ice_samples=50):
    grid            = pdp_data["grid"]
    pdp_values      = pdp_data["pdp_values"]
    ice_curves      = pdp_data.get("ice_curves", {})
    feature_display = pdp_data["feature_display"]
    class_names     = list(pdp_values.keys())

    fig = go.Figure()

    # ICE curves (faint, behind PDP)
    if show_ice and ice_curves:
        for cls in class_names:
            curves  = ice_curves[cls]
            n       = min(n_ice_samples, len(curves))
            indices = np.random.choice(len(curves), n, replace=False)
            for idx in indices:
                fig.add_trace(go.Scatter(
                    x=grid, y=curves[idx], mode="lines",
                    line=dict(color=SPECIES_COLORS.get(cls, "#888"), width=0.6),
                    opacity=0.12, showlegend=False, hoverinfo="skip",
                ))

    # PDP main curves
    for cls in class_names:
        fig.add_trace(go.Scatter(
            x=grid, y=pdp_values[cls], mode="lines", name=cls,
            line=dict(color=SPECIES_COLORS.get(cls, "#888"), width=2.5),
            hovertemplate=f"<b>{cls}</b><br>{feature_display}: %{{x:.2f}}<br>P: %{{y:.3f}}<extra></extra>",
        ))

    # rug
    rug_trace, _ = _rug(pdp_data["feature_name"], -0.03)
    fig.add_trace(rug_trace)

    layout = _dark_layout(400)
    layout.update(
        title=dict(text=f"PDP — {feature_display}", font=dict(color=TEXT, size=14)),
        xaxis=_axis_style(title=feature_display),
        yaxis=_axis_style(title="Predicted Probability", range=[-0.07, 1.07]),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08,
                    font=dict(color=TEXT), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
    )
    fig.update_layout(**layout)
    return fig.to_json()


# ── ALE (standalone) ──────────────────────────────────────────────────────────
def build_ale_plot(ale_data):
    grid            = ale_data["grid"]
    ale_values      = ale_data["ale_values"]
    feature_display = ale_data["feature_display"]
    deriv_type      = ale_data.get("derivative_type", "")
    class_names     = list(ale_values.keys())

    fig = go.Figure()

    for cls in class_names:
        fig.add_trace(go.Scatter(
            x=grid, y=ale_values[cls], mode="lines", name=cls,
            line=dict(color=SPECIES_COLORS.get(cls, "#888"), width=2.5),
            hovertemplate=f"<b>{cls}</b><br>{feature_display}: %{{x:.2f}}<br>ALE: %{{y:.4f}}<extra></extra>",
        ))

    # zero reference line
    fig.add_hline(y=0, line_dash="dot", line_color="rgba(200,210,230,0.4)", line_width=1)

    # rug
    y_floor = min(min(v) for v in ale_values.values()) - 0.015
    rug_trace, _ = _rug(ale_data["feature_name"], y_floor)
    fig.add_trace(rug_trace)

    layout = _dark_layout(400)
    layout.update(
        title=dict(text=f"ALE — {feature_display}", font=dict(color=TEXT, size=14)),
        xaxis=_axis_style(title=feature_display),
        yaxis=_axis_style(title="ALE effect (centred)"),
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.08,
                    font=dict(color=TEXT), bgcolor="rgba(0,0,0,0)"),
        hovermode="x unified",
        annotations=[dict(
            text=f"<i>{deriv_type}</i>",
            xref="paper", yref="paper", x=1, y=-0.13,
            xanchor="right", showarrow=False,
            font=dict(size=10, color="rgba(180,190,210,0.6)"),
        )],
    )
    fig.update_layout(**layout)
    return fig.to_json()


# ── Combined PDP + ALE side by side ──────────────────────────────────────────
def build_combined_feature_effects_plot(model, feature_name, model_type, show_ice=False):
    pdp_data        = compute_pdp(model, feature_name, model_type)
    ale_data        = compute_ale(model, feature_name, model_type)
    feature_display = FEATURE_DISPLAY_NAMES.get(feature_name, feature_name)
    class_names     = list(pdp_data["pdp_values"].keys())
    deriv_type      = ale_data.get("derivative_type", "")

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=[
            f"Partial Dependence (PDP)",
            f"Accumulated Local Effects (ALE)",
        ],
        horizontal_spacing=0.12,
    )

    # ── PDP panel ────────────────────────────────────────────────────────────
    for cls in class_names:
        if show_ice and pdp_data.get("ice_curves"):
            curves  = pdp_data["ice_curves"][cls]
            n       = min(30, len(curves))
            indices = np.random.choice(len(curves), n, replace=False)
            for idx in indices:
                fig.add_trace(go.Scatter(
                    x=pdp_data["grid"], y=curves[idx], mode="lines",
                    line=dict(color=SPECIES_COLORS.get(cls, "#888"), width=0.5),
                    opacity=0.1, showlegend=False, hoverinfo="skip",
                ), row=1, col=1)

        fig.add_trace(go.Scatter(
            x=pdp_data["grid"], y=pdp_data["pdp_values"][cls],
            mode="lines", name=cls, legendgroup=cls,
            line=dict(color=SPECIES_COLORS.get(cls, "#888"), width=2.5),
            hovertemplate=f"<b>{cls}</b><br>P: %{{y:.3f}}<extra></extra>",
        ), row=1, col=1)

    rug_trace, _ = _rug(feature_name, -0.03)
    fig.add_trace(rug_trace, row=1, col=1)

    # ── ALE panel ────────────────────────────────────────────────────────────
    for cls in class_names:
        fig.add_trace(go.Scatter(
            x=ale_data["grid"], y=ale_data["ale_values"][cls],
            mode="lines", name=cls, legendgroup=cls, showlegend=False,
            line=dict(color=SPECIES_COLORS.get(cls, "#888"), width=2.5),
            hovertemplate=f"<b>{cls}</b><br>ALE: %{{y:.4f}}<extra></extra>",
        ), row=1, col=2)

    fig.add_hline(y=0, line_dash="dot",
                  line_color="rgba(200,210,230,0.4)", line_width=1,
                  row=1, col=2)

    y_floor = min(min(v) for v in ale_data["ale_values"].values()) - 0.015
    rug_ale, _ = _rug(feature_name, y_floor)
    fig.add_trace(rug_ale, row=1, col=2)

    # ── global layout ────────────────────────────────────────────────────────
    fig.update_layout(
        **_dark_layout(430, margin=dict(l=55, r=25, t=65, b=65)),
        hovermode="x unified",
        showlegend=True,
        legend=dict(orientation="h", x=0.5, xanchor="center", y=1.12,
                    font=dict(color=TEXT, size=11), bgcolor="rgba(0,0,0,0)"),
    )

    ax = _axis_style()
    fig.update_xaxes(title_text=feature_display, **ax)
    fig.update_yaxes(title_text="Probability", range=[-0.07, 1.07],
                     row=1, col=1, **ax)
    fig.update_yaxes(title_text="ALE effect (centred)",
                     row=1, col=2, **ax)

    # subplot title colours
    for ann in fig.layout.annotations:
        ann.font.color = TEXT
        ann.font.size  = 12

    # derivative note
    fig.add_annotation(
        text=f"<i>{deriv_type}</i>",
        xref="paper", yref="paper", x=1, y=-0.14,
        xanchor="right", showarrow=False,
        font=dict(size=10, color="rgba(180,190,210,0.55)"),
    )

    return fig.to_json()


# ── Permutation importance bar chart ─────────────────────────────────────────
def build_all_features_importance_plot(model, model_type):
    importances = []
    for feat in num_cols:
        imp = compute_feature_importance_permutation(model, feat, model_type, n_repeats=5)
        importances.append({
            "display":    FEATURE_DISPLAY_NAMES.get(feat, feat),
            "importance": imp["mean_importance"],
            "std":        imp["std_importance"],
        })

    importances.sort(key=lambda x: x["importance"], reverse=True)

    bar_colors = ["#3dba6e" if i["importance"] > 0 else "#e5604a" for i in importances]
    x_labels   = [i["display"]    for i in importances]
    y_vals     = [i["importance"] for i in importances]
    y_stds     = [i["std"]        for i in importances]

    fig = go.Figure(go.Bar(
        x=x_labels,
        y=y_vals,
        marker_color=bar_colors,
        marker_line=dict(width=0),
        error_y=dict(
            type="data",
            array=y_stds,
            color="#ffffff",
            thickness=2,
            width=8,
        ),
        hovertemplate=(
            "<b>%{x}</b><br>"
            "Importance: %{y:.4f}<br>"
            "\u00b1%{error_y.array:.4f}<extra></extra>"
        ),
    ))

    # annotations placed ABOVE the error bar tip so they are never hidden
    annotations = []
    for lbl, val, std in zip(x_labels, y_vals, y_stds):
        tip = val + std
        annotations.append(dict(
            x=lbl,
            y=tip,
            text=f"<b>{val:.3f}</b><br><span style='font-size:9px'>\u00b1{std:.3f}</span>",
            showarrow=False,
            yanchor="bottom",
            yshift=6,
            font=dict(color=TEXT, size=11),
            bgcolor="rgba(26,29,46,0.75)",
            borderpad=3,
        ))

    y_max  = max(v + s for v, s in zip(y_vals, y_stds))
    y_ceil = y_max * 1.4

    layout = _dark_layout(360, margin=dict(l=55, r=25, t=50, b=65))
    layout.update(
        title=dict(text="Permutation Feature Importance", font=dict(color=TEXT, size=13)),
        xaxis=_axis_style(title="Feature"),
        yaxis=_axis_style(title="Accuracy drop when shuffled", range=[0, y_ceil]),
        showlegend=False,
        annotations=annotations,
    )
    fig.update_layout(**layout)
    return fig.to_json()