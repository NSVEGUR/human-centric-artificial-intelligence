import json
import numpy as np
from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt

from .report import generate_report_pdf
from .classifier import get_classifier_stats, build_confusion_matrix_plot
from .experts import get_expert_stats, build_expert_accuracy_plot
from .deferral import get_deferral_stats, build_accuracy_coverage_plot, build_deferral_summary_plot
from .active_learning import get_al_stats, build_al_learning_curves_plot
from .data import LABEL_NAMES


def index(request):
    template = loader.get_template("project3/index.html")

    classifier_stats = get_classifier_stats()
    expert_stats = get_expert_stats()
    deferral_stats = get_deferral_stats()
    al_stats = get_al_stats()

    context = {
        # Task 1
        "test_acc": classifier_stats["test_acc"],
        "label_names": classifier_stats["label_names"],
        "confusion_matrix_json": build_confusion_matrix_plot(),
        # Task 2
        "expert_accuracy_json": build_expert_accuracy_plot(),
        "experts": expert_stats["experts"],
        # Task 3
        "deferral_stats": deferral_stats,
        "accuracy_coverage_json": build_accuracy_coverage_plot(),
        "deferral_summary_json": build_deferral_summary_plot(),
        # Task 4
        "al_stats": al_stats,
        "al_curves_json": build_al_learning_curves_plot(),
    }

    return HttpResponse(template.render(context, request))


@csrf_exempt
def classifier_stats(request):
    if request.method != "GET":
        return JsonResponse({"error": "only GET allowed"}, status=405)
    return JsonResponse(get_classifier_stats())


@csrf_exempt
def expert_stats(request):
    if request.method != "GET":
        return JsonResponse({"error": "only GET allowed"}, status=405)
    return JsonResponse(get_expert_stats())


@csrf_exempt
def deferral_stats_api(request):
    if request.method != "GET":
        return JsonResponse({"error": "only GET allowed"}, status=405)
    return JsonResponse(get_deferral_stats())


@csrf_exempt
def al_stats_api(request):
    if request.method != "GET":
        return JsonResponse({"error": "only GET allowed"}, status=405)
    return JsonResponse(get_al_stats())


# ── Task 5: Interactive Human Expert ─────────────────────────────────────────
from .active_learning import (
    pool_texts, pool_labels, pool_probas, pool_expert_preds,
    eval_probas, eval_expert_preds, eval_labels,
    _least_confidence, _margin, _entropy, _evaluate_deferral,
    N_POOL,
)
from .deferral import ai_only_acc, optimal_team_acc
from .experts import sports_per_class, tech_per_class

# how many labels before we show the user their expert profile. fewer than
# this and the per class estimates are basically noise
MIN_LABELS_FOR_PROFILE = 12

# measured per class accuracy of the simulated experts as fractions
_sports_frac = np.array([sports_per_class[LABEL_NAMES[k]] / 100.0 for k in range(4)])
_tech_frac = np.array([tech_per_class[LABEL_NAMES[k]] / 100.0 for k in range(4)])


def _compute_user_profile(labeled):
    """Estimate the user's per-class accuracy from the labels they gave.

    Same Laplace smoothing as the active learning loop so the numbers are
    directly comparable to what the AL strategies learn about the simulated
    experts.
    """
    correct_by_class = np.zeros(4)
    total_by_class = np.zeros(4)
    for item in labeled:
        t = item["true_label"]
        correct_by_class[t] += int(item["user_label"] == t)
        total_by_class[t] += 1

    prior_correct = np.ones(4) * 5.0
    prior_total = np.ones(4) * 10.0
    smoothed = (correct_by_class + prior_correct) / (total_by_class + prior_total)

    per_class = []
    for k in range(4):
        n = int(total_by_class[k])
        raw = round(100 * correct_by_class[k] / n, 1) if n > 0 else None
        per_class.append({
            "label": LABEL_NAMES[k],
            "n": n,
            "raw_acc": raw,
            "smoothed_acc": round(float(smoothed[k]) * 100, 1),
        })

    n_total = int(total_by_class.sum())
    overall = round(100 * float(correct_by_class.sum()) / n_total, 1) if n_total else None

    # specialty = class with the best smoothed accuracy that the user has
    # actually seen at least twice (otherwise its just the prior talking)
    seen = [k for k in range(4) if total_by_class[k] >= 2]
    if seen:
        specialty_idx = max(seen, key=lambda k: smoothed[k])
    else:
        specialty_idx = int(np.argmax(smoothed))

    # which simulated expert is the user most similar to (L1 distance
    # between the smoothed profile and each experts measured profile)
    d_sports = float(np.abs(smoothed - _sports_frac).sum())
    d_tech = float(np.abs(smoothed - _tech_frac).sum())
    similar_to = "Sports Expert" if d_sports <= d_tech else "Sci/Tech Expert"

    return {
        "per_class": per_class,
        "smoothed": smoothed,          # np array, internal use
        "overall_acc": overall,
        "specialty": LABEL_NAMES[specialty_idx],
        "specialty_acc": round(float(smoothed[specialty_idx]) * 100, 1),
        "similar_to": similar_to,
    }


def _evaluate_you_as_expert(smoothed_profile, seed=123):
    """Team accuracy with the *user* as the expert.

    We simulate the user on the held-out eval set with the exact same
    mechanism as the simulated experts (correct with probability p for
    their class, otherwise the classifier's second most likely class),
    then apply the same alpha=1 deferral rule as everywhere else. Fixed
    seed so the number doesnt jump around between page refreshes.
    """
    rng = np.random.default_rng(seed)
    user_preds = np.empty(len(eval_labels), dtype=int)
    for i, y in enumerate(eval_labels):
        y = int(y)
        if rng.random() < smoothed_profile[y]:
            user_preds[i] = y
        else:
            order = np.argsort(eval_probas[i])[::-1]
            user_preds[i] = int(next(c for c in order if c != y))

    p_user_correct = eval_probas @ smoothed_profile
    clf_uncertainty = 1.0 - eval_probas.max(axis=1)
    defer_mask = clf_uncertainty > (1.0 - p_user_correct)

    clf_preds = np.argmax(eval_probas, axis=1)
    team_preds = np.where(defer_mask, user_preds, clf_preds)

    from sklearn.metrics import accuracy_score
    team_acc = accuracy_score(eval_labels, team_preds) * 100
    deferral_rate = float(defer_mask.mean()) * 100
    return round(team_acc, 2), round(deferral_rate, 1)


def _build_profile_payload(labeled):
    """Everything the 'Your Expert Profile' section needs, or a locked
    placeholder if the user hasnt labeled enough articles yet."""
    n = len(labeled)
    if n < MIN_LABELS_FOR_PROFILE:
        return {
            "unlocked": False,
            "n_labeled": n,
            "labels_needed": MIN_LABELS_FOR_PROFILE - n,
            "min_labels": MIN_LABELS_FOR_PROFILE,
        }

    profile = _compute_user_profile(labeled)
    you_team_acc, you_deferral_rate = _evaluate_you_as_expert(profile["smoothed"])

    return {
        "unlocked": True,
        "n_labeled": n,
        "min_labels": MIN_LABELS_FOR_PROFILE,
        "per_class": profile["per_class"],
        "overall_acc": profile["overall_acc"],
        "specialty": profile["specialty"],
        "specialty_acc": profile["specialty_acc"],
        "similar_to": profile["similar_to"],
        "you_team_acc": you_team_acc,
        "you_deferral_rate": you_deferral_rate,
        "ai_only_acc": round(ai_only_acc, 2),
        "simulated_team_acc": round(optimal_team_acc, 2),
        # experts measured profiles for the comparison chart
        "sports_per_class": [round(float(v) * 100, 1) for v in _sports_frac],
        "tech_per_class": [round(float(v) * 100, 1) for v in _tech_frac],
        "label_names": LABEL_NAMES,
    }


def _get_next_query(strategy, queried_set, pool_probas):
    available = [i for i in range(N_POOL) if i not in queried_set]
    if not available:
        return None

    if strategy == 'random':
        return int(np.random.choice(available))

    avail_probas = pool_probas[available]
    if strategy == 'least_confidence':
        utils = _least_confidence(avail_probas)
    elif strategy == 'margin':
        utils = _margin(avail_probas)
    else:  # entropy (default)
        utils = _entropy(avail_probas)

    return available[int(np.argmax(utils))]


def _compute_current_team_acc(labeled):
    """Compute team accuracy from user-labeled instances so far."""
    if not labeled:
        return None

    correct_by_class = np.zeros(4)
    total_by_class = np.zeros(4)
    prior_correct = np.ones(4) * 5.0
    prior_total = np.ones(4) * 10.0

    for item in labeled:
        true_label = item["true_label"]
        user_label = item["user_label"]
        correct_by_class[true_label] += int(user_label == true_label)
        total_by_class[true_label] += 1

    est_per_class_acc = (correct_by_class + prior_correct) / (total_by_class + prior_total)
    team_acc = _evaluate_deferral(est_per_class_acc, eval_probas, eval_expert_preds, eval_labels)
    return round(team_acc, 2)


def human_label(request):
    template = loader.get_template("project3/human_label.html")

    # Initialise session state
    if 'al_labeled' not in request.session:
        request.session['al_labeled'] = []
    if 'al_strategy' not in request.session:
        request.session['al_strategy'] = 'entropy'
    if 'al_queried' not in request.session:
        request.session['al_queried'] = []

    labeled = request.session['al_labeled']
    strategy = request.session['al_strategy']
    queried_set = set(request.session['al_queried'])

    next_idx = _get_next_query(strategy, queried_set, pool_probas)

    current_article = None
    if next_idx is not None:
        clf_proba = pool_probas[next_idx].tolist()
        clf_pred = int(np.argmax(clf_proba))
        current_article = {
            "idx": int(next_idx),
            "text": pool_texts[next_idx][:600],
            "clf_pred": LABEL_NAMES[clf_pred],
            "clf_confidence": round(max(clf_proba) * 100, 1),
        }

    team_acc = _compute_current_team_acc(labeled)

    context = {
        "label_names": LABEL_NAMES,
        "label_names_json": json.dumps(LABEL_NAMES),
        "current_article": current_article,
        "n_labeled": len(labeled),
        "strategy": strategy,
        "team_acc": team_acc,
        "labeled_history": list(reversed(labeled[-5:])),
        "profile_json": json.dumps(_build_profile_payload(labeled)),
    }

    return HttpResponse(template.render(context, request))


@csrf_exempt
def human_label_submit(request):
    if request.method != "POST":
        return JsonResponse({"error": "only POST allowed"}, status=405)

    try:
        body = json.loads(request.body)
        article_idx = int(body["idx"])
        user_label = int(body["label"])
    except (KeyError, ValueError, json.JSONDecodeError):
        return JsonResponse({"error": "invalid payload"}, status=400)

    if article_idx < 0 or article_idx >= N_POOL:
        return JsonResponse({"error": "invalid idx"}, status=400)
    if user_label < 0 or user_label >= 4:
        return JsonResponse({"error": "invalid label"}, status=400)

    labeled = request.session.get('al_labeled', [])
    queried = request.session.get('al_queried', [])
    strategy = request.session.get('al_strategy', 'entropy')

    true_label = int(pool_labels[article_idx])
    labeled.append({
        "idx": article_idx,
        "true_label": true_label,
        "user_label": user_label,
        "user_label_name": LABEL_NAMES[user_label],
        "true_label_name": LABEL_NAMES[true_label],
        "correct": user_label == true_label,
        "text_snippet": pool_texts[article_idx][:120],
    })
    queried.append(article_idx)

    request.session['al_labeled'] = labeled
    request.session['al_queried'] = queried
    request.session.modified = True

    team_acc = _compute_current_team_acc(labeled)

    # Next query
    queried_set = set(queried)
    next_idx = _get_next_query(strategy, queried_set, pool_probas)

    next_article = None
    if next_idx is not None:
        clf_proba = pool_probas[next_idx].tolist()
        clf_pred = int(np.argmax(clf_proba))
        next_article = {
            "idx": int(next_idx),
            "text": pool_texts[next_idx][:600],
            "clf_pred": LABEL_NAMES[clf_pred],
            "clf_confidence": round(max(clf_proba) * 100, 1),
        }

    return JsonResponse({
        "submitted": True,
        "correct": user_label == true_label,
        "true_label_name": LABEL_NAMES[true_label],
        "n_labeled": len(labeled),
        "team_acc": team_acc,
        "next_article": next_article,
        "profile": _build_profile_payload(labeled),
    })


@csrf_exempt
def human_label_strategy(request):
    if request.method != "POST":
        return JsonResponse({"error": "only POST allowed"}, status=405)

    try:
        body = json.loads(request.body)
        strategy = body["strategy"]
    except (KeyError, json.JSONDecodeError):
        return JsonResponse({"error": "invalid payload"}, status=400)

    if strategy not in ('random', 'least_confidence', 'margin', 'entropy'):
        return JsonResponse({"error": "unknown strategy"}, status=400)

    request.session['al_strategy'] = strategy
    request.session.modified = True
    return JsonResponse({"strategy": strategy})


def download_report(request):
    pdf_bytes = generate_report_pdf()
    response = HttpResponse(pdf_bytes, content_type='application/pdf')
    response['Content-Disposition'] = 'attachment; filename="project3_report.pdf"'
    return response


@csrf_exempt
def human_label_reset(request):
    if request.method != "POST":
        return JsonResponse({"error": "only POST allowed"}, status=405)

    request.session['al_labeled'] = []
    request.session['al_queried'] = []
    request.session.modified = True
    return JsonResponse({"reset": True})
