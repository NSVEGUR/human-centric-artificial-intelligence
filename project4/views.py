import json
import uuid

from django.http import HttpResponse, JsonResponse
from django.shortcuts import render, redirect
from django.urls import reverse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt

from . import data
from .models import Participant, Trial, QuestionnaireResponse, FinalResponse
from .preference_model import fit_w, evaluate_pairs
from .report import generate_report_pdf
from .session_plan import build_plan
from .study_config import (
    PAIRWISE_ELICIT_N, RANKING_ELICIT_N, RANKING_SIZE, VALIDATION_N,
    DESIGN_LABELS, STAGE_URL_NAME,
)

FEATURE_DIM = data.FEATURE_MATRIX.shape[1]


# ── landing / report ─────────────────────────────────────────────────────

def landing(request):
    return render(request, "project4/landing.html")


def download_report(request):
    pdf_bytes = generate_report_pdf()
    response = HttpResponse(pdf_bytes, content_type="application/pdf")
    response["Content-Disposition"] = 'attachment; filename="project4_report.pdf"'
    return response


# ── session / state-machine helpers ──────────────────────────────────────

def _current_url(request):
    stage = request.session.get("p4_stage")
    if not stage:
        return reverse("project4:consent")
    return reverse(STAGE_URL_NAME[stage])


def _require_participant(request):
    """Redirect to consent if this browser hasn't started a session, or to
    wherever the participant actually is if they try to jump ahead/back."""
    pid = request.session.get("p4_participant_id")
    if not pid:
        return redirect("project4:consent")
    return None


def _guard_stage(request, expected_stage):
    redirect_resp = _require_participant(request)
    if redirect_resp:
        return redirect_resp
    if request.session.get("p4_stage") != expected_stage:
        return redirect(_current_url(request))
    return None


def _advance(request):
    stage = request.session["p4_stage"]
    idx = request.session["p4_condition_idx"]
    if stage == "background":
        stage = "instructions"
    elif stage == "instructions":
        stage = "trials"
    elif stage == "trials":
        stage = "questionnaire"
    elif stage == "questionnaire":
        if idx == 0:
            idx = 1
            stage = "instructions"
        else:
            stage = "validation"
    elif stage == "validation":
        stage = "final"
    elif stage == "final":
        stage = "done"
    request.session["p4_stage"] = stage
    request.session["p4_condition_idx"] = idx
    request.session.modified = True


def _current_design(request):
    order = request.session["p4_order"]
    return order[request.session["p4_condition_idx"]]


# ── consent ───────────────────────────────────────────────────────────────

def consent(request):
    if request.session.get("p4_participant_id"):
        return redirect(_current_url(request))

    if request.method == "POST":
        if request.POST.get("agree") != "on":
            return render(request, "project4/consent.html", {"error": "Please check the consent box to continue."})

        # balanced counterbalancing of condition order
        order = ["pairwise", "ranking"] if Participant.objects.count() % 2 == 0 else ["ranking", "pairwise"]
        participant = Participant.objects.create(
            condition_order="pairwise_first" if order[0] == "pairwise" else "ranking_first",
            consented_at=timezone.now(),
            prolific_id=request.POST.get("prolific_id", "").strip(),
        )

        request.session["p4_participant_id"] = str(participant.id)
        request.session["p4_order"] = order
        request.session["p4_condition_idx"] = 0
        request.session["p4_stage"] = "background"
        request.session["p4_plan"] = build_plan()
        request.session.modified = True
        return redirect(_current_url(request))

    return render(request, "project4/consent.html")


# ── background survey (once, right after consent) ───────────────────────

BACKGROUND_FIELDS = ["age_bracket", "movie_frequency", "recsys_familiarity"]


def background(request):
    resp = _guard_stage(request, "background")
    if resp:
        return resp

    pid = request.session["p4_participant_id"]

    if request.method == "POST":
        values = {f: request.POST.get(f, "") for f in BACKGROUND_FIELDS}
        if any(not v for v in values.values()):
            return render(request, "project4/background.html", {"error": "Please answer all questions."})
        Participant.objects.filter(id=pid).update(**values)
        _advance(request)
        return redirect(_current_url(request))

    return render(request, "project4/background.html")


# ── instructions ─────────────────────────────────────────────────────────

def instructions(request):
    resp = _guard_stage(request, "instructions")
    if resp:
        return resp

    design = _current_design(request)
    idx = request.session["p4_condition_idx"]

    if request.method == "POST":
        _advance(request)
        return redirect(_current_url(request))

    return render(request, "project4/instructions.html", {
        "design": design,
        "design_label": DESIGN_LABELS[design],
        "condition_number": idx + 1,
        "pairwise_n": PAIRWISE_ELICIT_N,
        "ranking_n": RANKING_ELICIT_N,
        "ranking_size": RANKING_SIZE,
    })


# ── trials (pairwise & ranking elicitation, AJAX loop) ──────────────────

def _plan_for(request, design):
    return request.session["p4_plan"][design]


def _elicitation_spec(request, design):
    """Return (phase, trial_index, spec_dict) for the next trial to show,
    or None if this design's elicitation is complete. Authoritative source
    of truth is the DB row count, so a page refresh always resumes correctly."""
    pid = request.session["p4_participant_id"]
    plan = _plan_for(request, design)

    n_practice_done = Trial.objects.filter(participant_id=pid, design=design, phase="practice").count()
    if n_practice_done < 1:
        return "practice", 0, plan["practice"]

    n_elicit_done = Trial.objects.filter(participant_id=pid, design=design, phase="elicitation").count()
    elicitation = plan["elicitation"]
    if n_elicit_done < len(elicitation):
        return "elicitation", n_elicit_done, elicitation[n_elicit_done]

    return None


def _trial_payload(design, phase, trial_index, spec, total_elicit):
    movies = data.get_movies(spec["movies"])
    payload = {
        "design": design,
        "phase": phase,
        "trial_index": trial_index,
        "total": total_elicit,
        "movies": movies,
    }
    if phase == "practice":
        payload["target_id"] = spec["target"]
    return payload


def trials(request):
    resp = _guard_stage(request, "trials")
    if resp:
        return resp

    design = _current_design(request)
    spec = _elicitation_spec(request, design)
    if spec is None:
        _advance(request)
        return redirect(_current_url(request))

    phase, trial_index, trial_spec = spec
    total = len(_plan_for(request, design)["elicitation"])
    payload = _trial_payload(design, phase, trial_index, trial_spec, total)

    template = "project4/trial_pairwise.html" if design == "pairwise" else "project4/trial_ranking.html"
    return render(request, template, {
        "design": design,
        "design_label": DESIGN_LABELS[design],
        "trial": payload,
        "total": total,
    })


@csrf_exempt
def submit_trial(request):
    resp = _guard_stage(request, "trials")
    if resp:
        return JsonResponse({"error": "not in trial stage"}, status=409)
    if request.method != "POST":
        return JsonResponse({"error": "only POST allowed"}, status=405)

    design = _current_design(request)
    spec = _elicitation_spec(request, design)
    if spec is None:
        return JsonResponse({"error": "no trial pending"}, status=409)
    phase, trial_index, trial_spec = spec
    movie_ids = trial_spec["movies"]

    try:
        body = json.loads(request.body)
        response_time_ms = body.get("response_time_ms")
    except json.JSONDecodeError:
        return JsonResponse({"error": "invalid payload"}, status=400)

    pid = request.session["p4_participant_id"]

    if design == "pairwise":
        chosen = body.get("chosen_id")
        if chosen not in movie_ids:
            return JsonResponse({"error": "chosen_id not in this trial"}, status=400)
        response = chosen
    else:
        order = body.get("order")
        if not isinstance(order, list) or sorted(order) != sorted(movie_ids):
            return JsonResponse({"error": "order must be a permutation of the shown movies"}, status=400)
        response = order

    is_correct = None
    if phase == "practice":
        target = trial_spec.get("target")
        is_correct = (response == target) if design == "pairwise" else (response[0] == target)

    Trial.objects.create(
        participant_id=pid, design=design, phase=phase, trial_index=trial_index,
        movie_ids=movie_ids, response=response, response_time_ms=response_time_ms,
        is_correct=is_correct,
    )

    next_spec = _elicitation_spec(request, design)
    if next_spec is None:
        _advance(request)
        return JsonResponse({"ok": True, "done": True, "redirect": _current_url(request)})

    next_phase, next_idx, next_trial_spec = next_spec
    total = len(_plan_for(request, design)["elicitation"])
    payload = _trial_payload(design, next_phase, next_idx, next_trial_spec, total)
    return JsonResponse({"ok": True, "done": False, "next": payload})


# ── questionnaire ────────────────────────────────────────────────────────

LIKERT_FIELDS = ["ease_of_use", "cognitive_load", "enjoyment", "trust"]


def questionnaire(request):
    resp = _guard_stage(request, "questionnaire")
    if resp:
        return resp

    design = _current_design(request)
    pid = request.session["p4_participant_id"]

    if request.method == "POST":
        try:
            values = {f: int(request.POST[f]) for f in LIKERT_FIELDS}
        except (KeyError, ValueError):
            return render(request, "project4/questionnaire.html", {
                "design": design, "design_label": DESIGN_LABELS[design],
                "error": "Please answer all questions.",
            })
        if any(not (1 <= v <= 7) for v in values.values()):
            return render(request, "project4/questionnaire.html", {
                "design": design, "design_label": DESIGN_LABELS[design],
                "error": "Please answer all questions.",
            })

        QuestionnaireResponse.objects.update_or_create(
            participant_id=pid, design=design,
            defaults={**values, "free_text": request.POST.get("free_text", "").strip()},
        )
        _advance(request)
        return redirect(_current_url(request))

    return render(request, "project4/questionnaire.html", {
        "design": design,
        "design_label": DESIGN_LABELS[design],
    })


# ── shared held-out validation block ────────────────────────────────────

def _validation_spec(request):
    pid = request.session["p4_participant_id"]
    plan = request.session["p4_plan"]["validation"]
    n_done = Trial.objects.filter(participant_id=pid, design="pairwise", phase="validation").count()
    if n_done < len(plan):
        return n_done, plan[n_done]
    return None


def validation(request):
    resp = _guard_stage(request, "validation")
    if resp:
        return resp

    spec = _validation_spec(request)
    if spec is None:
        _advance(request)
        return redirect(_current_url(request))

    trial_index, trial_spec = spec
    total = len(request.session["p4_plan"]["validation"])
    payload = _trial_payload("pairwise", "validation", trial_index, trial_spec, total)
    return render(request, "project4/trial_validation.html", {
        "trial": payload,
        "total": total,
    })


@csrf_exempt
def submit_validation(request):
    resp = _guard_stage(request, "validation")
    if resp:
        return JsonResponse({"error": "not in validation stage"}, status=409)
    if request.method != "POST":
        return JsonResponse({"error": "only POST allowed"}, status=405)

    spec = _validation_spec(request)
    if spec is None:
        return JsonResponse({"error": "no trial pending"}, status=409)
    trial_index, trial_spec = spec
    movie_ids = trial_spec["movies"]

    try:
        body = json.loads(request.body)
        chosen = body.get("chosen_id")
        response_time_ms = body.get("response_time_ms")
    except json.JSONDecodeError:
        return JsonResponse({"error": "invalid payload"}, status=400)
    if chosen not in movie_ids:
        return JsonResponse({"error": "chosen_id not in this trial"}, status=400)

    pid = request.session["p4_participant_id"]
    Trial.objects.create(
        participant_id=pid, design="pairwise", phase="validation", trial_index=trial_index,
        movie_ids=movie_ids, response=chosen, response_time_ms=response_time_ms,
    )

    next_spec = _validation_spec(request)
    if next_spec is None:
        _advance(request)
        return JsonResponse({"ok": True, "done": True, "redirect": _current_url(request)})

    next_idx, next_trial_spec = next_spec
    total = len(request.session["p4_plan"]["validation"])
    payload = _trial_payload("pairwise", "validation", next_idx, next_trial_spec, total)
    return JsonResponse({"ok": True, "done": False, "next": payload})


# ── fitting w and scoring the two designs against the validation set ────

def _fit_and_score(pid):
    X = data.FEATURE_MATRIX

    pairwise_trials = Trial.objects.filter(participant_id=pid, design="pairwise", phase="elicitation")
    pairs = [(t.response, [m for m in t.movie_ids if m != t.response][0]) for t in pairwise_trials]

    ranking_trials = Trial.objects.filter(participant_id=pid, design="ranking", phase="elicitation")
    rankings = [t.response for t in ranking_trials]

    validation_trials = Trial.objects.filter(participant_id=pid, design="pairwise", phase="validation")
    gt_pairs = [(t.response, [m for m in t.movie_ids if m != t.response][0]) for t in validation_trials]

    results = {}
    if pairs:
        w_pairwise = fit_w(X, FEATURE_DIM, pairs=pairs)
        acc, n = evaluate_pairs(w_pairwise, X, gt_pairs)
        results["pairwise"] = acc
    if rankings:
        w_ranking = fit_w(X, FEATURE_DIM, rankings=rankings)
        acc, n = evaluate_pairs(w_ranking, X, gt_pairs)
        results["ranking"] = acc
    return results


# ── final forced-choice comparison ──────────────────────────────────────

def final(request):
    resp = _guard_stage(request, "final")
    if resp:
        return resp

    pid = request.session["p4_participant_id"]

    if request.method == "POST":
        preferred = request.POST.get("preferred_design")
        if preferred not in ("pairwise", "ranking"):
            return render(request, "project4/final.html", {"error": "Please choose one option."})

        scores = _fit_and_score(pid)
        participant = Participant.objects.get(id=pid)
        code = "P4-" + uuid.uuid4().hex[:8].upper()
        participant.completed_at = timezone.now()
        participant.completion_code = code
        participant.save()

        FinalResponse.objects.update_or_create(
            participant_id=pid,
            defaults={
                "preferred_design": preferred,
                "preferred_reason": request.POST.get("preferred_reason", "").strip(),
                "pairwise_validation_accuracy": scores.get("pairwise"),
                "ranking_validation_accuracy": scores.get("ranking"),
            },
        )
        _advance(request)
        return redirect(_current_url(request))

    return render(request, "project4/final.html")


def done(request):
    resp = _guard_stage(request, "done")
    if resp:
        return resp

    pid = request.session["p4_participant_id"]
    participant = Participant.objects.get(id=pid)
    final_response = getattr(participant, "final_response", None)

    return render(request, "project4/done.html", {
        "completion_code": participant.completion_code,
        "final_response": final_response,
    })
