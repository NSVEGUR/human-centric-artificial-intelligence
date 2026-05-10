import io
import base64
import math
from urllib import request
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from django.shortcuts import render, redirect

from .models import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    run_training,
    get_default_params,
    PROBLEM_EXPLANATIONS,
    MODEL_EXPLANATIONS,
)


# Helper function to convert matplotlib figure to base64 string
def fig_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100,
                facecolor='white', edgecolor='none')
    buffer.seek(0)
    image_data = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close(fig)
    return image_data


def detect_problem_type(target_series):
    if not pd.api.types.is_numeric_dtype(target_series):
        return 'classification'

    num_unique = target_series.nunique()
    num_total = len(target_series)

    if num_unique <= 10 and (num_unique / num_total) < 0.1:
        return 'classification'

    return 'regression'


def create_scatter_plot(df, x_feature, y_feature, target_col, problem_type, selected_classes):
    """Create a scatter plot comparing two features"""
    fig, ax = plt.subplots(figsize=(5, 4))
    
    if problem_type == 'classification':
        # Get all unique classes and create color mapping
        all_classes = sorted(df[target_col].unique())
        cmap = cm.get_cmap('tab10', len(all_classes))
        class_to_color = {}
        for i, c in enumerate(all_classes):
            class_to_color[str(c)] = cmap(i)
        
        # Plot each selected class
        for cls in selected_classes:
            subset = df[df[target_col].astype(str) == cls]
            if not subset.empty:
                ax.scatter(subset[x_feature], subset[y_feature],
                          label=cls, color=class_to_color.get(cls),
                          alpha=0.7, s=35, edgecolors='white', linewidths=0.3)
        
        if selected_classes:
            ax.legend(title=target_col, fontsize=8, title_fontsize=8)
    else:
        # For regression, use continuous color scale
        sc = ax.scatter(df[x_feature], df[y_feature], c=df[target_col],
                       cmap='viridis', alpha=0.7, s=35, 
                       edgecolors='white', linewidths=0.3)
        fig.colorbar(sc, ax=ax, label=target_col)
    
    ax.set_xlabel(x_feature, fontsize=9)
    ax.set_ylabel(y_feature, fontsize=9)
    ax.set_title(f'{x_feature} vs {y_feature}', fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    
    return fig_to_base64(fig)


def create_histogram(df, feature, target_col, problem_type, selected_classes):
    """Generate histogram for feature distribution"""
    fig, ax = plt.subplots(figsize=(5, 4))
    
    if problem_type == 'classification':
        all_classes = sorted(df[target_col].unique())
        cmap = cm.get_cmap('tab10', len(all_classes))
        
        # Create color mapping
        class_to_color = {}
        for i, c in enumerate(all_classes):
            class_to_color[str(c)] = cmap(i)
        
        # Plot histogram for each class
        for cls in selected_classes:
            subset = df[df[target_col].astype(str) == cls]
            if not subset.empty:
                ax.hist(subset[feature], bins=20, alpha=0.6, 
                       label=cls, color=class_to_color.get(cls))
        
        if selected_classes:
            ax.legend(title=target_col, fontsize=8, title_fontsize=8)
    else:
        # Simple histogram for regression
        ax.hist(df[feature], bins=25, alpha=0.75, 
               color='steelblue', edgecolor='white')
    
    ax.set_xlabel(feature, fontsize=9)
    ax.set_ylabel('Count', fontsize=9)
    ax.set_title(f'Distribution of {feature}', fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    
    return fig_to_base64(fig)


def create_correlation_heatmap(df, feature_cols):
    """Create correlation heatmap for numeric features"""
    fig, ax = plt.subplots(figsize=(5, 4))
    
    # Calculate correlation matrix
    corr = df[feature_cols].corr()
    
    # Display heatmap
    im = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    
    # Set ticks and labels
    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(feature_cols, fontsize=7)
    
    # Add correlation values as text
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            val = corr.values[i, j]
            # Use white text for high correlation, black for low
            if abs(val) > 0.6:
                color = 'white'
            else:
                color = 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', 
                   fontsize=8, color=color)
    
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title('Feature Correlation Matrix', fontsize=10, fontweight='bold')
    fig.tight_layout()
    
    return fig_to_base64(fig)


# Page size options for table pagination
PAGE_SIZE_OPTS = [10, 25, 50]


def build_table_context(df, post):
    """Build context dictionary for table display with pagination and filtering"""
    all_columns = list(df.columns)
    hidden_cols = post.getlist('hidden_cols')
    visible_cols = [c for c in all_columns if c not in hidden_cols]

    # Handle search functionality
    search_query = post.get('table_search', '').strip()
    if search_query:
        # Search across all visible columns
        mask = df[visible_cols].apply(
            lambda col: col.astype(str).str.contains(search_query, case=False, na=False)
        ).any(axis=1)
        df_view = df[mask].copy()
    else:
        df_view = df.copy()

    # Handle sorting
    sort_col = post.get('sort_col', '')
    sort_dir = post.get('sort_dir', 'asc')
    
    if sort_col and sort_col in df_view.columns:
        ascending = True if sort_dir == 'asc' else False
        df_view = df_view.sort_values(by=sort_col, ascending=ascending)
    
    # Toggle sort direction for next click
    if sort_dir == 'asc':
        next_sort_dir = 'desc'
    else:
        next_sort_dir = 'asc'

    # Handle page size
    try:
        page_size = int(post.get('page_size', 10))
        if page_size not in PAGE_SIZE_OPTS:
            page_size = 10
    except:
        page_size = 10

    # Calculate pagination
    total_rows = len(df_view)
    total_pages = math.ceil(total_rows / page_size)
    if total_pages < 1:
        total_pages = 1

    try:
        current_page = int(post.get('table_page', 1))
    except:
        current_page = 1
    
    # Make sure current page is valid
    if current_page < 1:
        current_page = 1
    if current_page > total_pages:
        current_page = total_pages

    # Get the rows for current page
    start = (current_page - 1) * page_size
    end = start + page_size
    df_page = df_view.iloc[start:end][visible_cols]

    # Build context dictionary
    context = {
        'table_rows': df_page.values.tolist(),
        'table_headers': visible_cols,
        'all_columns': all_columns,
        'hidden_cols': hidden_cols,
        'search_query': search_query,
        'sort_col': sort_col,
        'sort_dir': sort_dir,
        'next_sort_dir': next_sort_dir,
        'current_page': current_page,
        'total_pages': total_pages,
        'total_rows': total_rows,
        'page_size': page_size,
        'page_size_opts': PAGE_SIZE_OPTS,
        'page_range': _page_range(current_page, total_pages),
        'row_start': start + 1,
        'row_end': min(end, total_rows),
    }
    
    return context


def _page_range(current, total, window=2):
    """Generate page range for pagination with ellipsis"""
    # If total pages is small, show all
    if total <= 7:
        return list(range(1, total + 1))
    
    # Otherwise, show first, last, and pages around current
    pages = set([1, total])
    
    # Add pages around current page
    for p in range(max(1, current - window), min(total, current + window) + 1):
        pages.add(p)
    
    # Build result list with None for ellipsis
    result = []
    for p in sorted(pages):
        if result and p - result[-1] > 1:
            result.append(None)  # This will be rendered as ellipsis
        result.append(p)
    
    return result


def index(request):
    """Main view for the automated ML interface"""
    context = {
        'title': 'Project 1 — Automated Machine Learning',
        'description': 'An interface for a simple supervised learning setting.',
    }

    # Handle clear action - remove all session data
    if request.method == 'POST' and request.POST.get('action') == 'clear':
        session_keys = ['csv_data', 'csv_filename', 'scatter_plot',
                       'hist_plot', 'corr_plot', 'plot_meta']
        for key in session_keys:
            if key in request.session:
                del request.session[key]
        return redirect('project1:index')

    csv_text = None
    filename = None
    loaded_from_session = False

    # Figure out what action triggered this POST request
    action = request.POST.get('action', '')
    new_file = bool(request.FILES.get('csv_file'))
    update_plots = (action == 'update_plots')
    table_action = (action == 'table')
    train_action = (action == 'train')

    if request.method == 'POST':
        if new_file:
            # Handle new file upload
            csv_file = request.FILES['csv_file']
            csv_text = csv_file.read().decode('utf-8')
            filename = csv_file.name
            
            # Save to session
            request.session['csv_data'] = csv_text
            request.session['csv_filename'] = filename
            
            # Clear old plots from session
            plot_keys = ['scatter_plot', 'hist_plot', 'corr_plot', 'plot_meta']
            for key in plot_keys:
                if key in request.session:
                    del request.session[key]
                    
        elif 'csv_data' in request.session:
            # Load from session
            csv_text = request.session['csv_data']
            filename = request.session.get('csv_filename', 'uploaded.csv')
            loaded_from_session = not new_file and not update_plots

    if csv_text:
        try:
            # Parse CSV data
            df = pd.read_csv(io.StringIO(csv_text))
            
            # Remove ID columns if present
            id_cols = []
            for c in df.columns:
                if c.strip().lower() in ('id', 'index', 'unnamed: 0'):
                    id_cols.append(c)
            
            if id_cols:
                df = df.drop(columns=id_cols, errors='ignore')

            # Identify target and feature columns
            target_col = df.columns[-1]  # Last column is target
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            # Feature columns are numeric columns except target
            if target_col in numeric_cols:
                feature_cols = [c for c in numeric_cols if c != target_col]
            else:
                feature_cols = numeric_cols
            
            # Detect if classification or regression
            detected_problem_type = detect_problem_type(df[target_col])
            posted_problem_type = request.POST.get('problem_type', detected_problem_type)

            if posted_problem_type in ['classification', 'regression']:
                problem_type = posted_problem_type
            else:
                problem_type = detected_problem_type

            # Handle class selection for classification problems
            unique_classes = []
            selected_classes = []
            
            if problem_type == 'classification':
                unique_classes = sorted([str(c) for c in df[target_col].unique()])
                
                if new_file:
                    # Select all classes by default for new file
                    selected_classes = unique_classes
                else:
                    # Get selected classes from POST
                    selected_classes = request.POST.getlist('selected_classes')

            # Get plot axis selections from POST or use defaults
            if len(feature_cols) > 0:
                scatter_x = request.POST.get('scatter_x', feature_cols[0])
            else:
                scatter_x = feature_cols[0]
            
            if len(feature_cols) > 1:
                scatter_y = request.POST.get('scatter_y', feature_cols[1])
            else:
                scatter_y = feature_cols[0]
            
            hist_feature = request.POST.get('hist_feature', feature_cols[0])

            # Validate selections
            if scatter_x not in feature_cols:
                scatter_x = feature_cols[0]
            if scatter_y not in feature_cols:
                if len(feature_cols) > 1:
                    scatter_y = feature_cols[1]
                else:
                    scatter_y = feature_cols[0]
            if hist_feature not in feature_cols:
                hist_feature = feature_cols[0]

            # Decide whether to regenerate plots
            # Only regenerate on: new file upload, explicit update, or no cached plots
            # Table interactions should NOT trigger plot regeneration
            need_plots = False
            if not table_action:
                if new_file or update_plots:
                    need_plots = True
                elif 'scatter_plot' not in request.session:
                    need_plots = True

            if need_plots:
                # Generate new plots
                scatter_plot = create_scatter_plot(
                    df, scatter_x, scatter_y, target_col, problem_type, selected_classes)
                hist_plot = create_histogram(
                    df, hist_feature, target_col, problem_type, selected_classes)
                corr_plot = create_correlation_heatmap(df, feature_cols)
                
                # Save plots to session
                request.session['scatter_plot'] = scatter_plot
                request.session['hist_plot'] = hist_plot
                request.session['corr_plot'] = corr_plot
                request.session['plot_meta'] = {
                    'scatter_x': scatter_x,
                    'scatter_y': scatter_y,
                    'hist_feature': hist_feature,
                    'selected_classes': selected_classes,
                }
            else:
                # Load plots from session
                scatter_plot = request.session.get('scatter_plot', '')
                hist_plot = request.session.get('hist_plot', '')
                corr_plot = request.session.get('corr_plot', '')
                meta = request.session.get('plot_meta', {})
                
                # Restore plot selections from cache to keep dropdowns in sync
                if not update_plots and not new_file:
                    scatter_x = meta.get('scatter_x', scatter_x)
                    scatter_y = meta.get('scatter_y', scatter_y)
                    hist_feature = meta.get('hist_feature', hist_feature)
                    selected_classes = meta.get('selected_classes', selected_classes)

            # Get available models based on problem type
            if problem_type == 'classification':
                registry = CLASSIFICATION_MODELS
            else:
                registry = REGRESSION_MODELS
            
            available_models = []
            for key, cfg in registry.items():
                available_models.append((key, cfg['label']))

            # Build table context (always recomputed - it's fast)
            table_ctx = build_table_context(df, request.POST)

            # Update main context
            context.update({
                'scatter_plot': scatter_plot,
                'hist_plot': hist_plot,
                'corr_plot': corr_plot,
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
                'available_models': available_models,
                'loaded_from_session': loaded_from_session,
                'problem_explanation': PROBLEM_EXPLANATIONS.get(problem_type),
                'model_explanations': MODEL_EXPLANATIONS.get(problem_type),
            })
            
            # Merge table context
            for key, value in table_ctx.items():
                context[key] = value

            # Handle model training
            if train_action:
                model_key = request.POST.get('model_key', list(registry.keys())[0])
                test_size = float(request.POST.get('test_size', 0.2))

                # Validate test size
                if test_size < 0.1:
                    test_size = 0.1
                elif test_size > 0.5:
                    test_size = 0.5

                # Validate model key
                if model_key not in registry:
                    model_key = list(registry.keys())[0]

                # Get model hyperparameters
                model_params = get_default_params(model_key, problem_type)

                if model_key == 'KNN':
                    model_params['k'] = int(request.POST.get('k', 5))
                    model_params['metric'] = request.POST.get('metric', 'euclidean')

                elif model_key == 'DecisionTree':
                    model_params['max_depth'] = int(request.POST.get('max_depth', 5))
                    model_params['min_samples_leaf'] = int(request.POST.get('min_samples_leaf', 1))

                    if problem_type == 'classification':
                        model_params['criterion'] = request.POST.get('criterion', 'gini')
                    else:
                        model_params['criterion'] = request.POST.get('criterion', 'squared_error')

                elif model_key == 'LogisticRegression':
                    model_params['C'] = float(request.POST.get('C', 1))
                    model_params['penalty'] = request.POST.get('penalty', 'l2')

                elif model_key == 'Ridge':
                    model_params['alpha'] = float(request.POST.get('alpha', 1))

                training_results = run_training(
                    df,
                    feature_cols,
                    target_col,
                    problem_type,
                    model_key,
                    test_size,
                    model_params
                )

                context['training'] = training_results

        except Exception as e:
            # Handle any errors
            context['error'] = str(e)

    return render(request, 'project1/index.html', context)