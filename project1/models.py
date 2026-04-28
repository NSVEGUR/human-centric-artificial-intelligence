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
from sklearn.linear_model import Ridge, LogisticRegression, LinearRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)

#Models section

CLASSIFICATION_MODELS = {
    'KNN': {
        'label': 'K-Nearest Neighbors',
        'param_label': 'Number of Neighbours (k)',
        'param_range': list(range(1, 21)),
        'build': lambda params: KNeighborsClassifier(
            n_neighbors=int(params.get('k', 5)),
            metric=params.get('metric', 'euclidean')
        ),
    },
    'DecisionTree': {
        'label': 'Decision Tree',
        'param_label': 'Max Depth',
        'param_range': list(range(1, 16)),
        'build': lambda params: DecisionTreeClassifier(
            max_depth=int(params.get('max_depth', 5)),
            min_samples_leaf=int(params.get('min_samples_leaf', 1)),
            criterion=params.get('criterion', 'gini'),
            random_state=42
        ),
    },
    'LogisticRegression': {
        'label': 'Logistic Regression',
        'param_label': 'Regularisation (C)',
        'param_range': [0.001, 0.01, 0.1, 1, 10, 100],
        'build': lambda params: LogisticRegression(
            C=float(params.get('C', 1)),
            penalty=params.get('penalty', 'l2'),
            solver='liblinear',
            max_iter=1000,
            random_state=42
        ),
    },
}


REGRESSION_MODELS = {
    'KNN': {
        'label': 'K-Nearest Neighbors Regressor',
        'param_label': 'Number of Neighbours (k)',
        'param_range': list(range(1, 21)),
        'build': lambda params: KNeighborsRegressor(
            n_neighbors=int(params.get('k', 5)),
            metric=params.get('metric', 'euclidean')
        ),
    },
    'DecisionTree': {
        'label': 'Decision Tree Regressor',
        'param_label': 'Max Depth',
        'param_range': list(range(1, 16)),
        'build': lambda params: DecisionTreeRegressor(
            max_depth=int(params.get('max_depth', 5)),
            min_samples_leaf=int(params.get('min_samples_leaf', 1)),
            criterion=params.get('criterion', 'squared_error'),
            random_state=42
        ),
    },
    'LinearRegression': {
        'label': 'Linear Regression',
        'param_label': 'No hyperparameter',
        'param_range': [1],
        'build': lambda params: LinearRegression(),
    },
    'Ridge': {
        'label': 'Ridge Regression',
        'param_label': 'Regularisation (alpha)',
        'param_range': [0.001, 0.01, 0.1, 1, 10, 100],
        'build': lambda params: Ridge(
            alpha=float(params.get('alpha', 1))
        ),
    },
}

PROBLEM_EXPLANATIONS = {
    'classification': {
        'title': 'Classification Problem',
        'description': (
            'Classification is used when the target value is a category or class. '
            'For example, flower type, pass/fail, yes/no, or disease/no disease. '
            'The model learns patterns from the input features and predicts which class a new sample belongs to.'
        ),
        'metrics': (
            'Accuracy shows overall correctness. Precision shows how reliable positive predictions are. '
            'Recall shows how many actual positives were found. F1 Score balances precision and recall.'
        )
    },
    'regression': {
        'title': 'Regression Problem',
        'description': (
            'Regression is used when the target value is a continuous number. '
            'For example, house price, temperature, salary, marks, or sales amount. '
            'The model learns relationships between features and predicts a numeric value.'
        ),
        'metrics': (
            'R² Score shows how well the model explains variation in the target. '
            'MAE is the average absolute error. MSE gives stronger penalty to large errors. '
            'RMSE is the error in the same unit as the target.'
        )
    }
}


MODEL_EXPLANATIONS = {
    'classification': {
        'KNN': {
            'description': 'KNN works by checking the closest data points and assigning the class based on what most of the neighbours belong to. The value of K decides how many neighbours are considered. If K is too small, the model may become sensitive to noise, while a larger K makes the prediction more stable but less detailed.'
        },
        'DecisionTree': {
            'description': 'A decision tree splits the data step by step based on feature values until it reaches a decision. It is easy to understand because it follows a rule-based structure. The depth of the tree controls how complex it becomes, and limiting it helps avoid overfitting.'
        },
        'LogisticRegression': {
            'description': 'Logistic regression predicts probabilities for different classes and assigns the most likely one. It is a simple and effective model, especially for linearly separable data. The regularisation parameter helps control overfitting by limiting how complex the model can become.'
        }
    },

    'regression': {
        'KNN': {
            'description': 'KNN regression predicts a value by averaging the values of nearby data points. It assumes that similar inputs will have similar outputs. The choice of K affects how smooth or sensitive the predictions are.'
        },
        'DecisionTree': {
            'description': 'A decision tree for regression divides the data into regions and predicts values based on the average within each region. It can capture non-linear patterns but may overfit if the tree becomes too deep.'
        },
        'LinearRegression': {
            'description': 'Linear regression models a straight-line relationship between features and the target variable. It is simple and works well when the relationship between variables is roughly linear.'
        },
        'Ridge': {
            'description': 'Ridge regression is similar to linear regression but includes a penalty term to prevent overfitting. It is useful when there are many correlated features and helps make the model more stable.'
        }
    }
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


#Default parameters for each model

def get_default_params(model_key, problem_type):
    if model_key == 'KNN':
        return {
            'k': 5,
            'metric': 'euclidean',
        }

    if model_key == 'DecisionTree':
        if problem_type == 'classification':
            criterion = 'gini'
        else:
            criterion = 'squared_error'

        return {
            'max_depth': 5,
            'min_samples_leaf': 1,
            'criterion': criterion,
        }

    if model_key == 'LogisticRegression':
        return {
            'C': 1,
            'penalty': 'l2',
        }

    if model_key == 'Ridge':
        return {'alpha': 1}

    return {}

#Training function
def run_training(df, feature_cols, target_col, problem_type, model_key, test_size, model_params=None):

    if model_params is None:
        model_params = get_default_params(model_key, problem_type)

    X = df[feature_cols].values
    y = df[target_col].values

    if problem_type == 'classification':
        y = LabelEncoder().fit_transform(y.astype(str))

    X_scaled = StandardScaler().fit_transform(X)

    stratify = y if problem_type == 'classification' and len(np.unique(y)) > 1 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify
    )

    registry = CLASSIFICATION_MODELS if problem_type == 'classification' else REGRESSION_MODELS
    config = registry[model_key]
    param_range = config['param_range']

    train_scores, test_scores = [], []

    for p in param_range:
        temp_params = model_params.copy()

        if model_key == 'KNN':
            temp_params['k'] = p
        elif model_key == 'DecisionTree':
            temp_params['max_depth'] = p
        elif model_key == 'LogisticRegression':
            temp_params['C'] = p
        elif model_key == 'Ridge':
            temp_params['alpha'] = p

        model = config['build'](temp_params)
        model.fit(X_train, y_train)

        y_train_pred = model.predict(X_train)
        y_test_pred = model.predict(X_test)

        if problem_type == 'classification':
            train_scores.append(accuracy_score(y_train, y_train_pred))
            test_scores.append(accuracy_score(y_test, y_test_pred))
        else:
            train_scores.append(r2_score(y_train, y_train_pred))
            test_scores.append(r2_score(y_test, y_test_pred))

    best_idx = int(np.argmax(test_scores))
    best_param = param_range[best_idx]

    overfit_gap = train_scores[best_idx] - test_scores[best_idx]

    if overfit_gap > 0.15:
        overfit_message = 'The model may be overfitting because the training score is much higher than the test score.'
    elif test_scores[best_idx] < 0.5:
        overfit_message = 'The model performance is weak. It may need better features, cleaner data, or different parameters.'
    else:
        overfit_message = 'The train and test scores look reasonably balanced for this dataset.'

        final_params = model_params.copy()

    if model_key == 'KNN':
        final_params['k'] = best_param
    elif model_key == 'DecisionTree':
        final_params['max_depth'] = best_param
    elif model_key == 'LogisticRegression':
        final_params['C'] = best_param
    elif model_key == 'Ridge':
        final_params['alpha'] = best_param

    final_model = config['build'](final_params)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    if problem_type == 'classification':
        metrics = {
            'Accuracy': round(accuracy_score(y_test, y_pred), 4),
            'Precision': round(precision_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            'Recall': round(recall_score(y_test, y_pred, average='weighted', zero_division=0), 4),
            'F1 Score': round(f1_score(y_test, y_pred, average='weighted', zero_division=0), 4),
        }
        score_label = 'Accuracy'
    else:
        mse = mean_squared_error(y_test, y_pred)
        metrics = {
            'R² Score': round(r2_score(y_test, y_pred), 4),
            'MAE': round(mean_absolute_error(y_test, y_pred), 4),
            'MSE': round(mse, 4),
            'RMSE': round(np.sqrt(mse), 4),
        }
        score_label = 'R² Score'

    fig, ax = plt.subplots(figsize=(7, 4))
    x_pos = range(len(param_range))
    x_labels = [str(p) for p in param_range]

    ax.plot(x_pos, train_scores, 'o-', color='#275cb2', label='Train Score', linewidth=2, markersize=5)
    ax.plot(x_pos, test_scores, 's--', color='#e05c2d', label='Test Score', linewidth=2, markersize=5)
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

    if problem_type == 'classification':
        ax.set_ylim(-0.05, 1.1)

    fig.tight_layout()

    return {
        'training_plot': _fig_to_base64(fig),
        'results_table': [
            {'param': p, 'train': round(tr, 4), 'test': round(te, 4)}
            for p, tr, te in zip(param_range, train_scores, test_scores)
        ],
        'metrics': metrics,
        'best_param': best_param,
        'best_test': round(test_scores[best_idx], 4),
        'best_train': round(train_scores[best_idx], 4),
        'param_label': config['param_label'],
        'model_label': config['label'],
        'score_label': score_label,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'model_key': model_key,
        'test_size': test_size,
        'model_params': final_params,
        'overfit_gap': round(overfit_gap, 4),
        'overfit_message': overfit_message,
    }

