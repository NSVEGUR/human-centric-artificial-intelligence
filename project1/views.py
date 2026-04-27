import io
import base64
import math
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from django.shortcuts import render, redirect

from .models import CLASSIFICATION_MODELS, REGRESSION_MODELS, run_training


# ── Helpers ───────────────────────────────────────────────────────────────────

def fig_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format='png', bbox_inches='tight', dpi=100,
                facecolor='white', edgecolor='none')
    buffer.seek(0)
    img = base64.b64encode(buffer.read()).decode('utf-8')
    buffer.close()
    plt.close(fig)
    return img


def detect_problem_type(target_series):
    num_unique = target_series.nunique()
    num_total  = len(target_series)
    if num_unique <= 20 or (num_unique / num_total) < 0.05:
        return 'classification'
    return 'regression'


# ── Plots ─────────────────────────────────────────────────────────────────────

def create_scatter_plot(df, x_feature, y_feature, target_col, problem_type, selected_classes):
    fig, ax = plt.subplots(figsize=(5, 4))
    if problem_type == 'classification':
        all_classes    = sorted(df[target_col].unique())
        cmap           = cm.get_cmap('tab10', len(all_classes))
        class_to_color = {str(c): cmap(i) for i, c in enumerate(all_classes)}
        for cls in selected_classes:
            subset = df[df[target_col].astype(str) == cls]
            if not subset.empty:
                ax.scatter(subset[x_feature], subset[y_feature],
                           label=cls, color=class_to_color.get(cls),
                           alpha=0.7, s=35, edgecolors='white', linewidths=0.3)
        if selected_classes:
            ax.legend(title=target_col, fontsize=8, title_fontsize=8)
    else:
        sc = ax.scatter(df[x_feature], df[y_feature], c=df[target_col],
                        cmap='viridis', alpha=0.7, s=35, edgecolors='white', linewidths=0.3)
        fig.colorbar(sc, ax=ax, label=target_col)
    ax.set_xlabel(x_feature, fontsize=9)
    ax.set_ylabel(y_feature, fontsize=9)
    ax.set_title(f'{x_feature} vs {y_feature}', fontsize=10, fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.3)
    fig.tight_layout()
    return fig_to_base64(fig)


def create_histogram(df, feature, target_col, problem_type, selected_classes):
    fig, ax = plt.subplots(figsize=(5, 4))
    if problem_type == 'classification':
        all_classes    = sorted(df[target_col].unique())
        cmap           = cm.get_cmap('tab10', len(all_classes))
        class_to_color = {str(c): cmap(i) for i, c in enumerate(all_classes)}
        for cls in selected_classes:
            subset = df[df[target_col].astype(str) == cls]
            if not subset.empty:
                ax.hist(subset[feature], bins=20,
                        alpha=0.6, label=cls, color=class_to_color.get(cls))
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


def create_correlation_heatmap(df, feature_cols):
    fig, ax = plt.subplots(figsize=(5, 4))
    corr = df[feature_cols].corr()
    im   = ax.imshow(corr.values, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(feature_cols)))
    ax.set_yticks(range(len(feature_cols)))
    ax.set_xticklabels(feature_cols, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(feature_cols, fontsize=7)
    for i in range(len(feature_cols)):
        for j in range(len(feature_cols)):
            val   = corr.values[i, j]
            color = 'white' if abs(val) > 0.6 else 'black'
            ax.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8, color=color)
    fig.colorbar(im, ax=ax, fraction=0.046)
    ax.set_title('Feature Correlation Matrix', fontsize=10, fontweight='bold')
    fig.tight_layout()
    return fig_to_base64(fig)


# ── Table helpers ─────────────────────────────────────────────────────────────

PAGE_SIZE_OPTS = [10, 25, 50]


def build_table_context(df, post):
    all_columns  = list(df.columns)
    hidden_cols  = post.getlist('hidden_cols')
    visible_cols = [c for c in all_columns if c not in hidden_cols]

    # Search
    search_query = (post.get('table_search') or '').strip()
    if search_query:
        mask = df[visible_cols].apply(
            lambda col: col.astype(str).str.contains(search_query, case=False, na=False)
        ).any(axis=1)
        df_view = df[mask].copy()
    else:
        df_view = df.copy()

    # Sort
    sort_col = post.get('sort_col', '') or ''
    sort_dir = post.get('sort_dir', 'asc') or 'asc'
    if sort_col and sort_col in df_view.columns:
        df_view = df_view.sort_values(by=sort_col, ascending=(sort_dir == 'asc'))
    next_sort_dir = 'desc' if sort_dir == 'asc' else 'asc'

    # Page size
    try:
        page_size = int(post.get('page_size', 10))
        if page_size not in PAGE_SIZE_OPTS:
            page_size = 10
    except (ValueError, TypeError):
        page_size = 10

    total_rows  = len(df_view)
    total_pages = max(1, math.ceil(total_rows / page_size))

    try:
        current_page = int(post.get('table_page', 1))
    except (ValueError, TypeError):
        current_page = 1
    current_page = max(1, min(current_page, total_pages))

    start   = (current_page - 1) * page_size
    end     = start + page_size
    df_page = df_view.iloc[start:end][visible_cols]

    return {
        'table_rows':     df_page.values.tolist(),
        'table_headers':  visible_cols,
        'all_columns':    all_columns,
        'hidden_cols':    hidden_cols,
        'search_query':   search_query,
        'sort_col':       sort_col,
        'sort_dir':       sort_dir,
        'next_sort_dir':  next_sort_dir,
        'current_page':   current_page,
        'total_pages':    total_pages,
        'total_rows':     total_rows,
        'page_size':      page_size,
        'page_size_opts': PAGE_SIZE_OPTS,
        'page_range':     _page_range(current_page, total_pages),
        'row_start':      start + 1,
        'row_end':        min(end, total_rows),
    }


def _page_range(current, total, window=2):
    if total <= 7:
        return list(range(1, total + 1))
    pages = set([1, total])
    for p in range(max(1, current - window), min(total, current + window) + 1):
        pages.add(p)
    result = []
    for p in sorted(pages):
        if result and p - result[-1] > 1:
            result.append(None)
        result.append(p)
    return result


# ── Main view ─────────────────────────────────────────────────────────────────

def index(request):
    context = {
        'title':       'Project 1 — Automated Machine Learning',
        'description': 'An interface for a simple supervised learning setting.',
    }

    # ── Clear ────────────────────────────────────────────────────────────────
    if request.method == 'POST' and request.POST.get('action') == 'clear':
        for key in ('csv_data', 'csv_filename', 'scatter_plot',
                    'hist_plot', 'corr_plot', 'plot_meta'):
            request.session.pop(key, None)
        return redirect('project1:index')

    csv_text            = None
    filename            = None
    loaded_from_session = False

    # ── Determine which action triggered this POST ────────────────────────────
    action         = request.POST.get('action', '')
    new_file       = bool(request.FILES.get('csv_file'))
    update_plots   = (action == 'update_plots')
    table_action   = (action == 'table')
    train_action   = (action == 'train')

    if request.method == 'POST':
        if new_file:
            csv_file = request.FILES['csv_file']
            csv_text = csv_file.read().decode('utf-8')
            filename = csv_file.name
            request.session['csv_data']     = csv_text
            request.session['csv_filename'] = filename
            # Invalidate cached plots when a new file is uploaded
            for key in ('scatter_plot', 'hist_plot', 'corr_plot', 'plot_meta'):
                request.session.pop(key, None)
        elif request.session.get('csv_data'):
            csv_text            = request.session['csv_data']
            filename            = request.session.get('csv_filename', 'uploaded.csv')
            loaded_from_session = not new_file and not update_plots

    if csv_text:
        try:
            df = pd.read_csv(io.StringIO(csv_text))
            id_cols = [c for c in df.columns
                       if c.strip().lower() in ('id', 'index', 'unnamed: 0')]
            df.drop(columns=id_cols, inplace=True, errors='ignore')

            target_col   = df.columns[-1]
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            feature_cols = ([c for c in numeric_cols if c != target_col]
                            if target_col in numeric_cols else numeric_cols)
            problem_type = detect_problem_type(df[target_col])

            # Classes
            unique_classes   = []
            selected_classes = []
            if problem_type == 'classification':
                unique_classes = sorted([str(c) for c in df[target_col].unique()])
                if new_file:
                    selected_classes = unique_classes
                else:
                    selected_classes = request.POST.getlist('selected_classes')

            # Plot axis selections
            scatter_x    = request.POST.get('scatter_x', feature_cols[0])
            scatter_y    = request.POST.get('scatter_y', feature_cols[1] if len(feature_cols) > 1 else feature_cols[0])
            hist_feature = request.POST.get('hist_feature', feature_cols[0])

            if scatter_x    not in feature_cols: scatter_x    = feature_cols[0]
            if scatter_y    not in feature_cols: scatter_y    = feature_cols[min(1, len(feature_cols) - 1)]
            if hist_feature not in feature_cols: hist_feature = feature_cols[0]

            # ── Regenerate plots only when needed ────────────────────────────
            # Regenerate if: new file, explicit update_plots, or no cached plots
            # Table actions NEVER regenerate — always read cached plots from session.
            # Regenerate only on new file upload or explicit Update Plots click.
            need_plots = (not table_action) and (new_file or update_plots or not request.session.get('scatter_plot'))

            if need_plots:
                scatter_plot = create_scatter_plot(
                    df, scatter_x, scatter_y, target_col, problem_type, selected_classes)
                hist_plot    = create_histogram(
                    df, hist_feature, target_col, problem_type, selected_classes)
                corr_plot    = create_correlation_heatmap(df, feature_cols)
                # Cache in session
                request.session['scatter_plot'] = scatter_plot
                request.session['hist_plot']    = hist_plot
                request.session['corr_plot']    = corr_plot
                request.session['plot_meta']    = {
                    'scatter_x':       scatter_x,
                    'scatter_y':       scatter_y,
                    'hist_feature':    hist_feature,
                    'selected_classes': selected_classes,
                }
            else:
                scatter_plot = request.session.get('scatter_plot', '')
                hist_plot    = request.session.get('hist_plot', '')
                corr_plot    = request.session.get('corr_plot', '')
                meta         = request.session.get('plot_meta', {})
                # Restore plot selections from cache so dropdowns stay in sync
                if not update_plots and not new_file:
                    scatter_x        = meta.get('scatter_x', scatter_x)
                    scatter_y        = meta.get('scatter_y', scatter_y)
                    hist_feature     = meta.get('hist_feature', hist_feature)
                    selected_classes = meta.get('selected_classes', selected_classes)

            registry         = CLASSIFICATION_MODELS if problem_type == 'classification' else REGRESSION_MODELS
            available_models = [(key, cfg['label']) for key, cfg in registry.items()]

            # Table (always recomputed — fast, no matplotlib)
            table_ctx = build_table_context(df, request.POST)

            context.update({
                'scatter_plot':        scatter_plot,
                'hist_plot':           hist_plot,
                'corr_plot':           corr_plot,
                'feature_cols':        feature_cols,
                'target_col':          target_col,
                'problem_type':        problem_type,
                'unique_classes':      unique_classes,
                'selected_classes':    selected_classes,
                'scatter_x':           scatter_x,
                'scatter_y':           scatter_y,
                'hist_feature':        hist_feature,
                'n_rows':              len(df),
                'n_cols':              len(df.columns),
                'filename':            filename,
                'available_models':    available_models,
                'loaded_from_session': loaded_from_session,
                **table_ctx,
            })

            if train_action:
                model_key = request.POST.get('model_key', list(registry.keys())[0])
                test_size = float(request.POST.get('test_size', 0.2))
                test_size = max(0.1, min(0.5, test_size))
                if model_key not in registry:
                    model_key = list(registry.keys())[0]
                context['training'] = run_training(
                    df, feature_cols, target_col, problem_type, model_key, test_size)

        except Exception as e:
            context['error'] = str(e)

    return render(request, 'project1/index.html', context)