import io
import base64
import numpy as np
import pandas as pd

# Use 'Agg' backend for headless environments and import brfore the render to avoid crashing 
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from django.shortcuts import render

 
# Helper: convert a matplotlib figure to a base64 string
def fig_to_base64(fig):
    """Saves a matplotlib figure to a buffer and returns base64 string."""
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100,
                facecolor='white', edgecolor='none')
    buffer.seek(0)
    image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close(fig)
    return image_base64

 
# Helper: decide if the target column is classification or regression
def detect_problem_type(target_series):
    """Simple heuristic to detect task type."""
    num_unique = target_series.nunique()
    num_total = len(target_series)
    # If 20 or fewer unique values or very low variety, treat as classification
    if num_unique <= 20 or (num_unique / num_total) < 0.05:
        return 'classification'
    return 'regression'

 
# Plot 1: Scatter Plot with Category Filtering
def create_scatter_plot(df, x_feature, y_feature, target_col, problem_type, selected_classes):
    fig, ax = plt.subplots(figsize=(5, 4))

    if problem_type == 'classification':
        # Get all classes for consistent color mapping across updates
        all_classes = sorted(df[target_col].unique())
        colormap = cm.get_cmap('tab10', len(all_classes))
        class_to_color = {str(cls): colormap(i) for i, cls in enumerate(all_classes)}

        # If selected_classes is empty, this loop won't run (empty plot)
        for cls_name in selected_classes:
            subset = df[df[target_col].astype(str) == cls_name]
            if not subset.empty:
                ax.scatter(
                    subset[x_feature], subset[y_feature],
                    label=cls_name, color=class_to_color.get(cls_name),
                    alpha=0.7, s=35, edgecolors='white', linewidths=0.3
                )
        if selected_classes:
            ax.legend(title=target_col, fontsize=8, title_fontsize=8)
    else:
        # Regression: uses a continuous color scale
        sc = ax.scatter(
            df[x_feature], df[y_feature],
            c=df[target_col], cmap='viridis',
            alpha=0.7, s=35, edgecolors='white', linewidths=0.3
        )
        fig.colorbar(sc, ax=ax, label=target_col)

    ax.set_xlabel(x_feature, fontsize=9)
    ax.set_ylabel(y_feature, fontsize=9)
    ax.set_title(f'{x_feature} vs {y_feature}', fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()

    return fig_to_base64(fig)

 
# Plot 2: Histogram with Category Filtering
def create_histogram(df, feature, target_col, problem_type, selected_classes):
    fig, ax = plt.subplots(figsize=(5, 4))

    if problem_type == 'classification':
        all_classes = sorted(df[target_col].unique())
        colormap = cm.get_cmap('tab10', len(all_classes))
        class_to_color = {str(cls): colormap(i) for i, cls in enumerate(all_classes)}

        for cls_name in selected_classes:
            subset = df[df[target_col].astype(str) == cls_name]
            if not subset.empty:
                ax.hist(
                    subset[feature], bins=20,
                    alpha=0.6, label=cls_name, color=class_to_color.get(cls_name)
                )
        if selected_classes:
            ax.legend(title=target_col, fontsize=8, title_fontsize=8)
    else:
        ax.hist(df[feature], bins=25, alpha=0.75, color='steelblue', edgecolor='white')

    ax.set_xlabel(feature, fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title(f'Distribution of {feature}', fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()

    return fig_to_base64(fig)

 
# Plot 3: Correlation heatmap
def create_correlation_heatmap(df, feature_cols):
    fig, ax = plt.subplots(figsize=(5, 4))
    corr = df[feature_cols].corr()
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')

    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(feature_cols, fontsize=7)

    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            val = corr.values[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)

    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title('Feature Correlation Matrix', fontsize=10, fontweight='bold')
    fig.tight_layout()

    return fig_to_base64(fig)

 
# Main view
def index(request):
    context = {
        "title": "Project 1 — Automated Machine Learning",
        "description": "An interface for a simple supervised learning setting.",
    }

    csv_text = None
    filename = None

    # Handle POST Request
    if request.method == 'POST':
        if request.FILES.get('csv_file'):
            csv_file = request.FILES['csv_file']
            csv_text = csv_file.read().decode('utf-8')
            filename = csv_file.name
            request.session['csv_data'] = csv_text
            request.session['csv_filename'] = filename
        elif request.session.get('csv_data'):
            csv_text = request.session['csv_data']
            filename = request.session.get('csv_filename', 'uploaded.csv')

    # Process CSV Data
    if csv_text:
        try:
            df = pd.read_csv(io.StringIO(csv_text))

            # Clean: Drop standard ID columns
            id_columns = [col for col in df.columns if col.strip().lower() in ('id', 'index', 'unnamed: 0')]
            df.drop(columns=id_columns, inplace=True, errors='ignore')

            # Column Detection
            target_col = df.columns[-1]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = [c for c in numeric_cols if c != target_col] if target_col in numeric_cols else numeric_cols
            problem_type = detect_problem_type(df[target_col])

            # Class Selection Logic (Dynamic from CSV)
            unique_classes = []
            selected_classes = []
            if problem_type == 'classification':
                unique_classes = sorted([str(c) for c in df[target_col].unique()])
                
                if request.FILES.get('csv_file'):
                    # Case 1: Brand new file uploaded -> Default to all selected
                    selected_classes = unique_classes
                else:
                    # Case 2: Update click -> Get checkboxes. 
                    # Note: If user unchecks everything, this correctly returns an empty list [].
                    selected_classes = request.POST.getlist('selected_classes')

            # Fetch Dropdown values
            scatter_x = request.POST.get('scatter_x', feature_cols[0])
            scatter_y = request.POST.get('scatter_y', feature_cols[1] if len(feature_cols) > 1 else feature_cols[0])
            hist_feature = request.POST.get('hist_feature', feature_cols[0])

            # Validation to avoid crashes if user switches files with different columns
            if scatter_x not in feature_cols: scatter_x = feature_cols[0]
            if scatter_y not in feature_cols: scatter_y = feature_cols[min(1, len(feature_cols)-1)]
            if hist_feature not in feature_cols: hist_feature = feature_cols[0]

            # Generate Plots
            scatter_plot = create_scatter_plot(df, scatter_x, scatter_y, target_col, problem_type, selected_classes)
            hist_plot = create_histogram(df, hist_feature, target_col, problem_type, selected_classes)
            corr_plot = create_correlation_heatmap(df, feature_cols)

            # Table Preview
            preview_html = df.head(10).to_html(classes='data-table', border=0, index=False)

            context.update({
                'scatter_plot': scatter_plot,
                'hist_plot': hist_plot,
                'corr_plot': corr_plot,
                'preview': preview_html,
                'feature_cols': feature_cols,
                'target_col': target_col,
                'problem_type': problem_type,
                'unique_classes': unique_classes,
                'selected_classes': selected_classes,
                'scatter_x': scatter_x,
                'scatter_y': scatter_y,
                'hist_feature': hist_feature,
                'n_rows': len(df),
                'n_cols': len(df.columns),
                'filename': filename,
            })

        except Exception as e:
            context['error'] = str(e)

    return render(request, 'project1/index.html', context)