import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import load_and_preprocess

# load once at module level and cache - same pattern as decision_tree.py / logistic_reg.py
_raw = load_and_preprocess()
train_X, test_X, train_y, test_y, mad_vals, num_cols = _raw

NUMERICAL_FEATURES = num_cols

BINARY_FEATURES = ['sex_Female', 'sex_Male']

ISLAND_FEATURES = ['island_Biscoe', 'island_Dream', 'island_Torgersen']

FEATURE_DISPLAY_NAMES = {
    'bill_length_mm': 'Bill Length (mm)',
    'bill_depth_mm': 'Bill Depth (mm)',
    'flipper_length_mm': 'Flipper Length (mm)',
    'body_mass_g': 'Body Mass (g)',
    'sex_Female': 'Sex: Female',
    'sex_Male': 'Sex: Male',
    'island_Biscoe': 'Island: Biscoe',
    'island_Dream': 'Island: Dream',
    'island_Torgersen': 'Island: Torgersen',
}

FEATURE_STATS = {}
for col in train_X.columns:
    FEATURE_STATS[col] = {
        'mean': train_X[col].mean(),
        'std': train_X[col].std(),
        'min': train_X[col].min(),
        'max': train_X[col].max(),
    }


def get_test_samples():
    samples = []
    for i in range(len(test_X)):
        row = test_X.iloc[i]
        species = test_y.iloc[i]

        summary = (
            f"#{i}: {species} | "
            f"Bill: {row['bill_length_mm']:.1f}mm, "
            f"Flipper: {row['flipper_length_mm']:.0f}mm, "
            f"Mass: {row['body_mass_g']:.0f}g"
        )

        samples.append({
            'index': i,
            'species': species,
            'summary': summary,
            'bill_length': round(row['bill_length_mm'], 1),
            'bill_depth': round(row['bill_depth_mm'], 1),
            'flipper_length': round(row['flipper_length_mm'], 0),
            'body_mass': round(row['body_mass_g'], 0),
        })

    return samples


def sample_around_point(x_original, n_samples=1000, noise_scale=0.3):
    samples = []
    x_array = x_original.values if isinstance(x_original, pd.Series) else x_original
    feature_names = list(train_X.columns)

    for _ in range(n_samples):
        new_point = x_array.copy().astype(float)

        for j, col in enumerate(feature_names):
            if col in NUMERICAL_FEATURES:
                std = FEATURE_STATS[col]['std']
                noise = np.random.normal(0, std * noise_scale)
                new_point[j] = new_point[j] + noise

                # Clip to valid range (with small buffer)
                min_val = FEATURE_STATS[col]['min'] * 0.9
                max_val = FEATURE_STATS[col]['max'] * 1.1
                new_point[j] = np.clip(new_point[j], min_val, max_val)

            elif col in BINARY_FEATURES:
                # Flip with probability proportional to noise_scale
                if np.random.random() < noise_scale * 0.5:
                    new_point[j] = 1.0 - new_point[j]

        # Handle island features specially - they're one-hot encoded
        # If we flip, choose a random different island
        if np.random.random() < noise_scale * 0.3:
            island_indices = [feature_names.index(f) for f in ISLAND_FEATURES]
            # Set all to 0, then pick one randomly
            for idx in island_indices:
                new_point[idx] = 0.0
            chosen_idx = np.random.choice(island_indices)
            new_point[chosen_idx] = 1.0

        samples.append(new_point)

    return np.array(samples)


def compute_mad_distance(x_original, x_counterfactual):
    feature_names = list(train_X.columns)
    x_orig = x_original.values if isinstance(x_original, pd.Series) else x_original

    total_distance = 0.0
    feature_distances = {}

    for j, col in enumerate(feature_names):
        diff = abs(x_orig[j] - x_counterfactual[j])

        if col in NUMERICAL_FEATURES:
            # MAD-weighted distance
            mad = mad_vals.get(col, 1.0)
            weighted_diff = diff / mad
            total_distance += weighted_diff
            feature_distances[col] = {
                'diff': diff,
                'weighted': weighted_diff,
                'original': x_orig[j],
                'counterfactual': x_counterfactual[j],
            }
        else:
            # Binary/categorical: penalty of 1 per change
            if diff > 0.5: 
                total_distance += 1.0
                feature_distances[col] = {
                    'diff': diff,
                    'weighted': 1.0,
                    'original': x_orig[j],
                    'counterfactual': x_counterfactual[j],
                }

    return total_distance, feature_distances


def generate_counterfactuals(model_info, sample_idx, target_class, model_type,
                              k=5, initial_n=2000, max_iterations=5):
    # LR no longer uses a scaler - it trains on raw features same as the tree
    model = model_info['model']
    x_original = test_X.iloc[sample_idx]
    original_species = test_y.iloc[sample_idx]
    feature_names = list(train_X.columns)

    # both model types now predict on the raw feature DataFrame
    x_for_pred = x_original.to_frame().T

    original_pred = model.predict(x_for_pred)[0]
    original_proba = model.predict_proba(x_for_pred)[0]

    # If target is same as original prediction, warn but continue
    if original_pred == target_class:
        same_as_original = True
    else:
        same_as_original = False

    all_counterfactuals = []
    noise_scale = 0.2

    for iteration in range(max_iterations):
        n_samples = initial_n * (iteration + 1)
        samples = sample_around_point(x_original, n_samples=n_samples, noise_scale=noise_scale)

        samples_df = pd.DataFrame(samples, columns=feature_names)
        # no scaler needed - LR and tree both trained on raw features
        samples_for_pred = samples_df

        predictions = model.predict(samples_for_pred)
        probabilities = model.predict_proba(samples_for_pred)

        target_mask = predictions == target_class
        target_samples = samples[target_mask]
        target_probas = probabilities[target_mask]

        if len(target_samples) > 0:
            for i in range(len(target_samples)):
                dist, feat_dists = compute_mad_distance(x_original, target_samples[i])
                all_counterfactuals.append({
                    'sample': target_samples[i],
                    'distance': dist,
                    'feature_distances': feat_dists,
                    'probability': target_probas[i],
                    'predicted_class': target_class,
                })

        if len(all_counterfactuals) >= k:
            break

        noise_scale += 0.15

    all_counterfactuals.sort(key=lambda x: x['distance'])
    top_k = all_counterfactuals[:k]

    class_names = list(model.classes_)
    class_idx = class_names.index(target_class)

    formatted_cfs = []
    for i, cf in enumerate(top_k):
        changes = []
        for col in feature_names:
            orig_val = x_original[col] if isinstance(x_original, pd.Series) else x_original[feature_names.index(col)]
            cf_val = cf['sample'][feature_names.index(col)]

            if col in NUMERICAL_FEATURES:
                if abs(orig_val - cf_val) > 0.01:
                    changes.append({
                        'feature': FEATURE_DISPLAY_NAMES.get(col, col),
                        'feature_key': col,
                        'original': round(float(orig_val), 2),
                        'counterfactual': round(float(cf_val), 2),
                        'change': round(float(cf_val - orig_val), 2),
                        'is_numerical': True,
                    })
            else:
                if abs(orig_val - cf_val) > 0.5:
                    changes.append({
                        'feature': FEATURE_DISPLAY_NAMES.get(col, col),
                        'feature_key': col,
                        'original': 'Yes' if orig_val > 0.5 else 'No',
                        'counterfactual': 'Yes' if cf_val > 0.5 else 'No',
                        'change': 'Changed',
                        'is_numerical': False,
                    })

        formatted_cfs.append({
            'rank': i + 1,
            'distance': round(cf['distance'], 3),
            'target_probability': round(float(cf['probability'][class_idx]) * 100, 1),
            'changes': changes,
            'all_probabilities': {
                class_names[j]: round(float(cf['probability'][j]) * 100, 1)
                for j in range(len(class_names))
            },
        })

    viz_json = build_counterfactual_visualization(
        x_original, formatted_cfs, original_species, original_pred,
        target_class, original_proba, class_names, feature_names
    )

    return {
        'original': {
            'index': sample_idx,
            'actual_species': original_species,
            'predicted_species': original_pred,
            'probabilities': {
                class_names[j]: round(float(original_proba[j]) * 100, 1)
                for j in range(len(class_names))
            },
            'features': {
                FEATURE_DISPLAY_NAMES.get(col, col): (
                    round(float(x_original[col]), 2) if col in NUMERICAL_FEATURES
                    else ('Yes' if x_original[col] > 0.5 else 'No')
                )
                for col in feature_names
            },
        },
        'target_class': target_class,
        'counterfactuals': formatted_cfs,
        'found_count': len(all_counterfactuals),
        'same_as_original': same_as_original,
        'visualization': viz_json,
    }


# ── dark theme constants ──────────────────────────────────────────────────────
PLOT_BG  = "#1a1d2e"
PAPER_BG = "rgba(0,0,0,0)"
TEXT     = "#c9d1e0"
GRID     = "#252840"

BLUE     = "#4a7fe5"   # original / Adelie
RED      = "#e5604a"   # counterfactual
GREEN    = "#3dba6e"   # positive change / Gentoo
ORANGE   = "#f5a623"   # Chinstrap

SPECIES_COLORS = {"Adelie": BLUE, "Chinstrap": ORANGE, "Gentoo": GREEN}

def _dark_layout(height=500):
    return dict(
        plot_bgcolor=PLOT_BG,
        paper_bgcolor=PAPER_BG,
        font=dict(color=TEXT, family="Inter, system-ui, sans-serif", size=12),
        height=height,
        margin=dict(l=50, r=30, t=60, b=50),
    )


def build_counterfactual_visualization(x_original, counterfactuals, original_species,
                                        original_pred, target_class, original_proba,
                                        class_names, feature_names):
    if not counterfactuals:
        fig = go.Figure()
        fig.add_annotation(
            text="No counterfactuals found. Try a different target class or sample.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color=TEXT)
        )
        fig.update_layout(**_dark_layout(300))
        return fig.to_json()

    best_cf = counterfactuals[0]

    # ── 3-panel layout: radar | changes | prob shift ──────────────────────────
    fig = make_subplots(
        rows=1, cols=3,
        specs=[[{"type": "polar"}, {"type": "xy"}, {"type": "xy"}]],
        subplot_titles=[
            "Feature Profile (Normalized)",
            "Changes Needed (Best CF)",
            "Probability Shift",
        ],
        horizontal_spacing=0.1,
    )

    # ── panel 1: radar ────────────────────────────────────────────────────────
    radar_features = NUMERICAL_FEATURES
    radar_labels   = [FEATURE_DISPLAY_NAMES[f] for f in radar_features]

    orig_vals, cf_vals = [], []
    for col in radar_features:
        orig  = float(x_original[col]) if isinstance(x_original, pd.Series) \
                else float(x_original[feature_names.index(col)])
        mn    = FEATURE_STATS[col]['min']
        mx    = FEATURE_STATS[col]['max']
        orig_norm = (orig - mn) / (mx - mn + 1e-9)
        orig_vals.append(round(orig_norm, 3))

        cf_change = next((c for c in best_cf['changes'] if c['feature_key'] == col), None)
        if cf_change:
            cf_norm = (cf_change['counterfactual'] - mn) / (mx - mn + 1e-9)
        else:
            cf_norm = orig_norm
        cf_vals.append(round(cf_norm, 3))

    # close the loop
    rl = radar_labels + [radar_labels[0]]
    ov = orig_vals   + [orig_vals[0]]
    cv = cf_vals     + [cf_vals[0]]

    fig.add_trace(go.Scatterpolar(
        r=ov, theta=rl, fill='toself',
        fillcolor='rgba(74,127,229,0.25)',
        line=dict(color=BLUE, width=2),
        name=f'Original ({original_pred})',
    ), row=1, col=1)

    fig.add_trace(go.Scatterpolar(
        r=cv, theta=rl, fill='toself',
        fillcolor='rgba(229,96,74,0.25)',
        line=dict(color=RED, width=2),
        name=f'Target ({target_class})',
    ), row=1, col=1)

    # ── panel 2: required changes bar ─────────────────────────────────────────
    # Normalise changes by feature range so all bars are comparable (0-1 scale)
    num_changes = [c for c in best_cf['changes'] if c['is_numerical']]
    cat_changes = [c for c in best_cf['changes'] if not c['is_numerical']]

    if num_changes:
        feat_labels, norm_changes, raw_labels, bar_colors = [], [], [], []
        for c in num_changes:
            col  = c['feature_key']
            rng  = FEATURE_STATS[col]['max'] - FEATURE_STATS[col]['min'] + 1e-9
            norm = c['change'] / rng          # normalised to [-1, 1]
            feat_labels.append(c['feature'].replace(' (mm)', '').replace(' (g)', ''))
            norm_changes.append(round(norm, 4))
            raw_labels.append(f"{'+' if c['change'] > 0 else ''}{c['change']:.1f}")
            bar_colors.append(GREEN if c['change'] > 0 else RED)

        fig.add_trace(go.Bar(
            x=feat_labels, y=norm_changes,
            marker_color=bar_colors,
            text=raw_labels,
            textposition='outside',
            textfont=dict(color=TEXT, size=10),
            name='Change needed',
            showlegend=False,
            hovertemplate="<b>%{x}</b><br>Raw change: %{text}<extra></extra>",
        ), row=1, col=2)

    # categorical changes as annotations
    if cat_changes and not num_changes:
        # show a text note if only categoricals changed
        fig.add_annotation(
            text="<br>".join([f"{c['feature']}: {c['original']} → {c['counterfactual']}"
                              for c in cat_changes]),
            xref="x2", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(color=TEXT, size=11),
        )

    # ── panel 3: probability shift ────────────────────────────────────────────
    orig_probs = [float(original_proba[i]) * 100 for i in range(len(class_names))]
    cf_probs   = [best_cf['all_probabilities'][c] for c in class_names]
    bar_x      = [c for c in class_names]

    fig.add_trace(go.Bar(
        x=bar_x, y=orig_probs,
        name='Original',
        marker_color=BLUE,
        opacity=0.85,
        text=[f"{p:.0f}%" for p in orig_probs],
        textposition='outside',
        textfont=dict(color=TEXT, size=10),
        offsetgroup=0,
        hovertemplate="<b>%{x}</b><br>Original: %{y:.1f}%<extra></extra>",
    ), row=1, col=3)

    fig.add_trace(go.Bar(
        x=bar_x, y=cf_probs,
        name='Counterfactual',
        marker_color=RED,
        opacity=0.85,
        text=[f"{p:.0f}%" for p in cf_probs],
        textposition='outside',
        textfont=dict(color=TEXT, size=10),
        offsetgroup=1,
        hovertemplate="<b>%{x}</b><br>Counterfactual: %{y:.1f}%<extra></extra>",
    ), row=1, col=3)

    # ── global layout ─────────────────────────────────────────────────────────
    fig.update_layout(
        **_dark_layout(420),
        barmode='group',
        showlegend=True,
        legend=dict(
            orientation='h', x=0.5, xanchor='center', y=-0.12,
            font=dict(color=TEXT, size=11), bgcolor='rgba(0,0,0,0)',
        ),
        polar=dict(
            bgcolor=PLOT_BG,
            radialaxis=dict(
                range=[0, 1], showticklabels=True,
                tickfont=dict(size=8, color=TEXT),
                gridcolor=GRID, linecolor=GRID,
            ),
            angularaxis=dict(
                tickfont=dict(size=9, color=TEXT),
                linecolor=GRID, gridcolor=GRID,
            ),
        ),
    )

    # style the xy axes
    axis_style = dict(color=TEXT, gridcolor=GRID, linecolor="#3a4060",
                      tickfont=dict(color=TEXT), zerolinecolor="#3a4060")

    fig.update_xaxes(tickangle=-20, **axis_style)
    fig.update_yaxes(**axis_style)
    fig.update_yaxes(title_text="Normalised change", row=1, col=2,
                     zeroline=True, zerolinewidth=1.5, zerolinecolor=TEXT)
    fig.update_yaxes(title_text="Probability (%)", range=[0, 115], row=1, col=3)

    # style subplot titles
    for ann in fig.layout.annotations:
        ann.font.color = TEXT
        ann.font.size  = 12

    return fig.to_json()


def build_counterfactual_table_viz(original, counterfactuals, target_class):
    if not counterfactuals:
        fig = go.Figure()
        fig.add_annotation(
            text="No counterfactuals found",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16, color=TEXT)
        )
        fig.update_layout(**_dark_layout(200))
        return fig.to_json()

    all_changed = sorted({
        c['feature'] for cf in counterfactuals
        for c in cf['changes'] if c['is_numerical']
    })

    if not all_changed:
        fig = go.Figure()
        fig.add_annotation(
            text="Only categorical features changed — see labels above",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=13, color=TEXT)
        )
        fig.update_layout(**_dark_layout(150))
        return fig.to_json()

    z_vals, hover_texts, y_labels = [], [], []

    for cf in counterfactuals:
        row, hover_row = [], []
        y_labels.append(f"CF #{cf['rank']}  dist={cf['distance']:.2f}  "
                        f"P({target_class})={cf['target_probability']}%")
        cd = {c['feature']: c for c in cf['changes'] if c['is_numerical']}

        for feat in all_changed:
            if feat in cd:
                ch   = cd[feat]['change']
                orig = cd[feat]['original']
                new  = cd[feat]['counterfactual']
                row.append(ch)
                hover_row.append(
                    f"<b>{feat}</b><br>Before: {orig}<br>After: {new}<br>"
                    f"Change: {'+' if ch > 0 else ''}{ch:.2f}"
                )
            else:
                row.append(0)
                hover_row.append(f"<b>{feat}</b><br>No change")

        z_vals.append(row)
        hover_texts.append(hover_row)

    # normalise each column to [-1,1] by its max abs so colours are comparable
    z_arr = np.array(z_vals, dtype=float)
    col_max = np.abs(z_arr).max(axis=0) + 1e-9
    z_norm  = z_arr / col_max

    text_labels = [
        [f"{v:+.1f}" if v != 0 else "—" for v in row]
        for row in z_vals
    ]

    fig = go.Figure(go.Heatmap(
        z=z_norm.tolist(),
        x=all_changed,
        y=y_labels,
        colorscale=[[0, "#c0392b"], [0.5, PLOT_BG], [1, "#2980b9"]],
        zmid=0, zmin=-1, zmax=1,
        text=text_labels,
        texttemplate="<b>%{text}</b>",
        textfont=dict(size=11, color="white"),
        hovertext=hover_texts,
        hoverinfo='text',
        showscale=True,
        colorbar=dict(
            title=dict(text="Direction", font=dict(color=TEXT)),
            tickfont=dict(color=TEXT),
            tickvals=[-1, 0, 1],
            ticktext=["Decrease", "No change", "Increase"],
            len=0.8,
        ),
    ))

    fig.update_layout(
        **_dark_layout(max(220, 70 * len(counterfactuals))),
        xaxis=dict(title="Feature", color=TEXT, tickfont=dict(color=TEXT, size=11),
                   linecolor="#3a4060"),
        yaxis=dict(title="", color=TEXT, tickfont=dict(color=TEXT, size=10),
                   linecolor="#3a4060"),
        margin=dict(l=260, r=100, t=30, b=60),
    )

    return fig.to_json()
