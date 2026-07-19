import io
import math
import numpy as np
import pandas as pd


from django.shortcuts import render, redirect

from .models import (
    CLASSIFICATION_MODELS,
    REGRESSION_MODELS,
    run_training,
    make_boundary,
    get_default_params,
    PROBLEM_EXPLANATIONS,
    MODEL_EXPLANATIONS,
)


def detect_problem_type(target_series):
    if not pd.api.types.is_numeric_dtype(target_series):
        return 'classification'

    num_unique = target_series.nunique()
    num_total = len(target_series)

    if num_unique <= 10 and (num_unique / num_total) < 0.1:
        return 'classification'

    return 'regression'


def _col_as_list(series):
    out = []
    for v in series:
        if pd.isna(v):
            out.append(None)
        elif isinstance(v, (int, float, np.integer, np.floating)):
            out.append(round(float(v), 4))
        else:
            out.append(str(v))
    return out


def build_viz_payload(df, feature_cols, target_col, problem_type):
    """Pack everything the frontend needs to draw scatter / histogram /
    correlation heatmap itself. Sending the raw columns instead of images
    means every control updates instantly in the browser."""

    features = {}
    for c in feature_cols:
        features[c] = _col_as_list(df[c])

    if problem_type == 'classification':
        target = [str(v) for v in df[target_col]]
        classes = sorted(set(target))
    else:
        target = _col_as_list(df[target_col])
        classes = []

    # correlation matrix (pandas skips NaN by itself which is convenient)
    corr = df[feature_cols].corr()
    corr_z = []
    for row in corr.values:
        corr_z.append([None if pd.isna(v) else round(float(v), 3) for v in row])

    return {
        'features': features,
        'target': target,
        'classes': classes,
        'feature_cols': feature_cols,
        'target_col': target_col,
        'problem_type': problem_type,
        'corr': {'cols': feature_cols, 'z': corr_z},
    }


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
        # only the csv lives in the session now, plots are drawn client side
        session_keys = ['csv_data', 'csv_filename']
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

            # note: the old "filter categories" checkboxes are gone. Plotly's
            # legend does the same job for free (click = hide a class,
            # double click = show only that class) so keeping our own
            # checkbox state around was just duplicate bookkeeping

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

            # Build the data payload for the client side plots. This is cheap
            # so we just always do it (the old image caching stuff is gone)
            viz_payload = build_viz_payload(df, feature_cols, target_col, problem_type)

            # Compute dataset health info 
            class_distribution = {}
            target_stats = {}
            imbalance_flag = False
            if problem_type == 'classification':
                counts = df[target_col].value_counts()
                total = len(df)
                for cls, cnt in counts.items():
                    # pct as a string on purpose: if django localisation is on
                    # (german locale etc) a float 33.3 renders as "33,3" in the
                    # template which is invalid css and breaks the width bars
                    pct = str(round(100 * cnt / total, 1))
                    class_distribution[str(cls)] = {'count': int(cnt), 'pct': pct}
                imbalance_flag = any(float(v['pct']) < 10 for v in class_distribution.values())
            else:
                target_stats = {
                    'min':    round(float(df[target_col].min()), 3),
                    'max':    round(float(df[target_col].max()), 3),
                    'mean':   round(float(df[target_col].mean()), 3),
                    'median': round(float(df[target_col].median()), 3),
                }

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
                'has_data': True,
                'viz_json': viz_payload,  # json_script in the template handles the encoding
                'feature_cols': feature_cols,
                'target_col': target_col,
                'problem_type': problem_type,
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
                'class_distribution': class_distribution,
                'target_stats': target_stats,
                'imbalance_flag': imbalance_flag,
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

                # decision boundary on 2 user-picked features. defaults to the
                # first two if nothing was picked yet
                boundary_x = request.POST.get('boundary_x', feature_cols[0])
                boundary_y = request.POST.get('boundary_y',
                                              feature_cols[1] if len(feature_cols) > 1 else feature_cols[0])
                if boundary_x not in feature_cols:
                    boundary_x = feature_cols[0]
                if boundary_y not in feature_cols or boundary_y == boundary_x:
                    # fall back to any other feature thats not boundary_x
                    others = [c for c in feature_cols if c != boundary_x]
                    boundary_y = others[0] if others else boundary_x

                boundary = None
                if len(feature_cols) > 1:
                    boundary = make_boundary(
                        df, feature_cols, target_col, problem_type,
                        model_key, training_results['model_params'],
                        boundary_x, boundary_y, test_size
                    )

                context['training'] = training_results
                context['boundary_x'] = boundary_x
                context['boundary_y'] = boundary_y
                # curve + boundary go to the template as one blob,
                # json_script turns it into json for us
                context['training_json'] = {
                    'curve': training_results['curve'],
                    'boundary': boundary,
                }

        except Exception as e:
            # Handle any errors
            context['error'] = str(e)

    return render(request, 'project1/index.html', context)