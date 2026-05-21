import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from .data import load_and_preprocess

train_X, test_X, train_y, test_y, mad_vals, num_cols = load_and_preprocess()

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
    from .logistic_reg import scaler, scaled_test_X

    model = model_info['model']
    x_original = test_X.iloc[sample_idx]
    original_species = test_y.iloc[sample_idx]
    feature_names = list(train_X.columns)

    # Get original prediction
    if model_type == 'lr':
        x_for_pred = scaled_test_X[sample_idx:sample_idx+1]
    else:
        x_for_pred = x_original.values.reshape(1, -1)

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

        if model_type == 'lr':
            samples_for_pred = scaler.transform(samples)
        else:
            samples_for_pred = samples

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


def build_counterfactual_visualization(x_original, counterfactuals, original_species,
                                        original_pred, target_class, original_proba,
                                        class_names, feature_names):
    if not counterfactuals:
        fig = go.Figure()
        fig.add_annotation(
            text="No counterfactuals found. Try a different target class or sample.",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16)
        )
        fig.update_layout(height=300)
        return fig.to_json()

    fig = make_subplots(
        rows=2, cols=2,
        specs=[
            [{"type": "polar"}, {"type": "xy"}],
            [{"type": "xy", "colspan": 2}, None]
        ],
        subplot_titles=(
            "Feature Comparison (Normalized)",
            "Required Changes",
            "Prediction Probability Shift"
        ),
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )

    best_cf = counterfactuals[0]

    # 1. Radar chart for numerical features
    radar_features = NUMERICAL_FEATURES
    radar_labels = [FEATURE_DISPLAY_NAMES[f] for f in radar_features]

    # Normalize values to 0-1 for radar
    orig_vals = []
    cf_vals = []
    for col in radar_features:
        orig = float(x_original[col]) if isinstance(x_original, pd.Series) else float(x_original[feature_names.index(col)])
        min_v = FEATURE_STATS[col]['min']
        max_v = FEATURE_STATS[col]['max']
        orig_norm = (orig - min_v) / (max_v - min_v + 1e-9)
        orig_vals.append(orig_norm)

        # Find CF value from changes
        cf_change = next((c for c in best_cf['changes'] if c['feature_key'] == col), None)
        if cf_change:
            cf_val = cf_change['counterfactual']
            cf_norm = (cf_val - min_v) / (max_v - min_v + 1e-9)
        else:
            cf_norm = orig_norm
        cf_vals.append(cf_norm)

    # Close the radar loop
    radar_labels_closed = radar_labels + [radar_labels[0]]
    orig_vals_closed = orig_vals + [orig_vals[0]]
    cf_vals_closed = cf_vals + [cf_vals[0]]

    fig.add_trace(
        go.Scatterpolar(
            r=orig_vals_closed,
            theta=radar_labels_closed,
            fill='toself',
            fillcolor='rgba(76, 114, 176, 0.3)',
            line=dict(color='#4C72B0', width=2),
            name=f'Original ({original_pred})',
        ),
        row=1, col=1
    )

    fig.add_trace(
        go.Scatterpolar(
            r=cf_vals_closed,
            theta=radar_labels_closed,
            fill='toself',
            fillcolor='rgba(196, 78, 82, 0.3)',
            line=dict(color='#C44E52', width=2),
            name=f'Counterfactual ({target_class})',
        ),
        row=1, col=1
    )

    # 2. Bar chart showing changes
    if best_cf['changes']:
        change_features = [c['feature'] for c in best_cf['changes'] if c['is_numerical']]
        change_values = [c['change'] for c in best_cf['changes'] if c['is_numerical']]

        colors = ['#55A868' if v > 0 else '#C44E52' for v in change_values]

        if change_features:
            fig.add_trace(
                go.Bar(
                    x=change_features,
                    y=change_values,
                    marker_color=colors,
                    name='Change Required',
                    text=[f"{'+' if v > 0 else ''}{v:.2f}" for v in change_values],
                    textposition='outside',
                    showlegend=False,
                ),
                row=1, col=2
            )

    # 3. Probability comparison
    x_positions = np.arange(len(class_names))
    width = 0.35

    orig_probs = [float(original_proba[i]) * 100 for i in range(len(class_names))]
    cf_probs = [best_cf['all_probabilities'][c] for c in class_names]

    fig.add_trace(
        go.Bar(
            x=[f"{c}" for c in class_names],
            y=orig_probs,
            name='Original',
            marker_color='#4C72B0',
            text=[f"{p:.1f}%" for p in orig_probs],
            textposition='outside',
            offsetgroup=0,
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Bar(
            x=[f"{c}" for c in class_names],
            y=cf_probs,
            name='Counterfactual',
            marker_color='#C44E52',
            text=[f"{p:.1f}%" for p in cf_probs],
            textposition='outside',
            offsetgroup=1,
        ),
        row=2, col=1
    )

    # Update layout
    fig.update_layout(
        height=600,
        showlegend=True,
        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='center', x=0.5),
        polar=dict(
            radialaxis=dict(range=[0, 1], showticklabels=False),
        ),
        barmode='group',
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    # Update axes
    fig.update_xaxes(title_text="Feature", row=1, col=2)
    fig.update_yaxes(title_text="Change in Value", row=1, col=2)
    fig.update_xaxes(title_text="Species", row=2, col=1)
    fig.update_yaxes(title_text="Probability (%)", range=[0, 110], row=2, col=1)

    return fig.to_json()


def build_counterfactual_table_viz(original, counterfactuals, target_class):
    if not counterfactuals:
        fig = go.Figure()
        fig.add_annotation(
            text="No counterfactuals found",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False, font=dict(size=16)
        )
        return fig.to_json()

    # Create a heatmap-style visualization of changes
    all_changed_features = set()
    for cf in counterfactuals:
        for change in cf['changes']:
            if change['is_numerical']:
                all_changed_features.add(change['feature'])

    all_changed_features = sorted(list(all_changed_features))

    if not all_changed_features:
        fig = go.Figure()
        fig.add_annotation(
            text="No numerical features changed",
            xref="paper", yref="paper", x=0.5, y=0.5,
            showarrow=False
        )
        return fig.to_json()

    z_values = []
    hover_texts = []
    y_labels = []

    for cf in counterfactuals:
        row = []
        hover_row = []
        y_labels.append(f"CF #{cf['rank']} (dist={cf['distance']:.2f})")

        change_dict = {c['feature']: c for c in cf['changes'] if c['is_numerical']}

        for feat in all_changed_features:
            if feat in change_dict:
                change = change_dict[feat]['change']
                orig = change_dict[feat]['original']
                new = change_dict[feat]['counterfactual']
                row.append(change)
                hover_row.append(f"{feat}<br>Original: {orig}<br>New: {new}<br>Change: {'+' if change > 0 else ''}{change}")
            else:
                row.append(0)
                hover_row.append(f"{feat}<br>No change")

        z_values.append(row)
        hover_texts.append(hover_row)

    fig = go.Figure(go.Heatmap(
        z=z_values,
        x=all_changed_features,
        y=y_labels,
        colorscale='RdBu',
        zmid=0,
        text=[[f"{v:+.1f}" if v != 0 else "-" for v in row] for row in z_values],
        texttemplate="%{text}",
        hovertext=hover_texts,
        hoverinfo='text',
        showscale=True,
        colorbar=dict(title='Change'),
    ))

    fig.update_layout(
        title=f"Feature Changes Required for {target_class} Prediction",
        xaxis_title="Feature",
        yaxis_title="Counterfactual",
        height=max(250, 80 * len(counterfactuals)),
        plot_bgcolor='white',
        paper_bgcolor='white',
    )

    return fig.to_json()
