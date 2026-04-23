import io
import base64
import numpy as np

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from django.db import models

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score


#Models section

CLASSIFICATION_MODELS = {
    'KNN': {
        'label': 'K-Nearest Neighbors',
        'param_label': 'Number of Neighbours (k)',
        'param_range': list(range(1, 21)),
        'build': lambda p: KNeighborsClassifier(n_neighbors=int(p)),
    },
    'DecisionTree': {
        'label': 'Decision Tree',
        'param_label': 'Max Depth',
        'param_range': list(range(1, 16)),
        'build': lambda p: DecisionTreeClassifier(max_depth=int(p), random_state=42),
    },
    'LogisticRegression': {
        'label': 'Logistic Regression',
        'param_label': 'Regularisation (C)',
        'param_range': [0.001, 0.01, 0.1, 1, 10, 100],
        'build': lambda p: LogisticRegression(C=float(p), max_iter=1000, random_state=42),
    },
}

REGRESSION_MODELS = {
    'KNN': {
        'label': 'K-Nearest Neighbors',
        'param_label': 'Number of Neighbours (k)',
        'param_range': list(range(1, 21)),
        'build': lambda p: KNeighborsRegressor(n_neighbors=int(p)),
    },
    'DecisionTree': {
        'label': 'Decision Tree',
        'param_label': 'Max Depth',
        'param_range': list(range(1, 16)),
        'build': lambda p: DecisionTreeRegressor(max_depth=int(p), random_state=42),
    },
    'Ridge': {
        'label': 'Ridge Regression',
        'param_label': 'Regularisation (alpha)',
        'param_range': [0.001, 0.01, 0.1, 1, 10, 100],
        'build': lambda p: Ridge(alpha=float(p)),
    },
}



#Helper function
def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100,
                facecolor='white', edgecolor='none')
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode('utf-8')
    buf.close()
    plt.close(fig)
    return b64


#Training function
def run_training(df, feature_cols, target_col, problem_type, model_key, test_size):
    
    X = df[feature_cols].values
    y = df[target_col].values

    if problem_type == 'classification':
        y = LabelEncoder().fit_transform(y.astype(str))

    X_scaled = StandardScaler().fit_transform(X)

    stratify = y if problem_type == 'classification' else None
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=test_size, random_state=42, stratify=stratify
    )

    registry    = CLASSIFICATION_MODELS if problem_type == 'classification' else REGRESSION_MODELS
    config      = registry[model_key]
    param_range = config['param_range']

    train_scores, test_scores = [], []
    for p in param_range:
        model = config['build'](p)
        model.fit(X_train, y_train)
        if problem_type == 'classification':
            train_scores.append(accuracy_score(y_train, model.predict(X_train)))
            test_scores.append( accuracy_score(y_test,  model.predict(X_test)))
        else:
            train_scores.append(r2_score(y_train, model.predict(X_train)))
            test_scores.append( r2_score(y_test,  model.predict(X_test)))

    best_idx    = int(np.argmax(test_scores))
    best_param  = param_range[best_idx]
    score_label = 'Accuracy' if problem_type == 'classification' else 'R² Score'

    # Training curve plot
    fig, ax = plt.subplots(figsize=(7, 4))
    x_pos    = range(len(param_range))
    x_labels = [str(p) for p in param_range]

    ax.plot(x_pos, train_scores, 'o-',  color='#275cb2', label='Train Score', linewidth=2, markersize=5)
    ax.plot(x_pos, test_scores,  's--', color='#e05c2d', label='Test Score',  linewidth=2, markersize=5)
    ax.axvline(best_idx, color='#27ae60', linestyle=':', linewidth=1.5,
               alpha=0.8, label=f'Best: {best_param}')
    ax.set_xticks(list(x_pos))
    ax.set_xticklabels(x_labels, rotation=45, fontsize=8)
    ax.set_xlabel(config['param_label'], fontsize=9)
    ax.set_ylabel(score_label, fontsize=9)
    ax.set_title(f'{config["label"]} — {score_label} vs {config["param_label"]}',
                 fontsize=10, fontweight='bold')
    ax.legend(fontsize=8)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_ylim(-0.05, 1.1)
    fig.tight_layout()

    return {
        'training_plot': _fig_to_base64(fig),
        'results_table': [
            {'param': p, 'train': round(tr, 4), 'test': round(te, 4)}
            for p, tr, te in zip(param_range, train_scores, test_scores)
        ],
        'best_param':  best_param,
        'best_test':   round(test_scores[best_idx],  4),
        'best_train':  round(train_scores[best_idx], 4),
        'param_label': config['param_label'],
        'model_label': config['label'],
        'score_label': score_label,
        'n_train':     len(X_train),
        'n_test':      len(X_test),
        'model_key':   model_key,
        'test_size':   test_size,
    }

