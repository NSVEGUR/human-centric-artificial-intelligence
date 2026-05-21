# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Django web application for a Human-Centric AI course at TUHH. The Django project config lives in `pbl/` (settings, urls, wsgi/asgi). Package management uses `uv`.

## Commands

```bash
# Install dependencies
uv sync

# Run development server
uv run python manage.py runserver

# Run all tests
uv run python manage.py test

# Run tests for a specific app
uv run python manage.py test project1
uv run python manage.py test project2

# Apply migrations
uv run python manage.py migrate
```

## Architecture

Multi-app Django project. URL routing: `/` redirects to `/home/`, then `/project1/` and `/project2/` are the main assignment apps.

**project1/** — Automated ML interface. Users upload a CSV; the app auto-detects classification vs. regression (last column = target), trains sklearn models with hyperparameter sweeps, and returns matplotlib plots as base64 strings embedded in the HTML. Per-request state (CSV data, plots) is stored in Django sessions. `project1/models.py` is a pure ML module — not a Django model.

**project2/** — ML interpretability interface on the Palmer Penguins dataset. All models train at Django **import time** (module-level code in `decision_tree.py` and `logistic_reg.py`), so they are ready without per-request retraining. The view renders an initial page with Plotly chart JSON embedded via `{{ var|escapejs }}`, then all interactivity uses `fetch()` POST calls to JSON endpoints. All chart functions return `fig.to_json()`.

**project2 modules:**
- `data.py` — shared loading, preprocessing (one-hot encode island/sex, drop year), 80/20 split, MAD computation
- `decision_tree.py` — trains 29 trees (2–30 leaves); selects best via `score = (1 − test_acc) + λ × norm_leaves`
- `logistic_reg.py` — trains 11 LR models across C values; same objective using L1 norm (ω) as complexity; exports `scaler` and `scaled_test_X` used by other modules
- `counterfactuals.py` — samples around a test point with Gaussian/categorical noise, filters for target-class predictions, ranks by MAD-weighted L1 distance
- `feature_effects.py` — PDP, ALE, and ICE computed from scratch (no external library); ALE uses analytical derivatives for LR and finite differences for trees

## Key Design Decisions

- **Import-time training:** Changing module-level code in `decision_tree.py` or `logistic_reg.py` requires a server restart to take effect.
- **LR requires scaling:** `logistic_reg.py` exports `scaler`; `counterfactuals.py` and `feature_effects.py` import it. Decision tree uses raw (unscaled) features.
- **Lambda objective:** `(1 − test_acc) + λ × normalized_complexity`. At λ=0 the most accurate model wins; higher λ favors simpler models.
- **Static/template layout:** Global CSS at `static/style.css`; app-level CSS at `project2/static/project2/style.css`. Global base template at `templates/base.html`; app templates under `<app>/templates/<app>/`.
