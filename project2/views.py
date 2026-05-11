import json
from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.http import HttpResponse, JsonResponse, Http404
from django.views.decorators.csrf import csrf_exempt

from .decision_tree import (
    get_best_tree, build_tree_plotly,
    build_tradeoff_plot, build_confusion_plot, build_gap_plot
)
from .logistic_reg import (
    get_best_lr, build_lr_plotly,
    build_tradeoff_plot as lr_tradeoff,
    build_confusion_plot as lr_confusion,
    build_gap_plot as lr_gap
)


def index(request):
    template = loader.get_template("project2/index.html")

    # default to best tree at lambda=0 (most accurate one)
    default_tree = get_best_tree(0.0)

    # build all 4 charts upfront for the initial render
    # these are json strings - template uses escapejs to pass them to JS safely
    context = {
        "initial_plot":      build_tree_plotly(default_tree),
        "initial_tradeoff":  build_tradeoff_plot(default_tree),
        "initial_confusion": build_confusion_plot(default_tree),
        "initial_gap":       build_gap_plot(default_tree),
        "initial_accuracy":  round(default_tree["test_acc"] * 100, 2),
        "initial_complexity": default_tree["n_leaves"],
    }

    return HttpResponse(template.render(context, request))


@csrf_exempt
def update_model(request):
    # called via fetch() every time the user moves the slider or switches model
    # returns updated chart data as json

    if request.method != "POST":
        return JsonResponse({"error": "only POST allowed"}, status=405)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "bad json"}, status=400)

    model_type     = body.get("model_type", "tree")
    lam            = float(body.get("lambda_val", 0.0))
    sort_by        = body.get("sort_by", "magnitude")
    filter_class   = body.get("selected_class", None)
    filter_feature = body.get("selected_feature", None)

    if model_type == "tree":
        picked = get_best_tree(lam)

        return JsonResponse({
            "plotly_json":    build_tree_plotly(picked),
            "tradeoff_json":  build_tradeoff_plot(picked),
            "confusion_json": build_confusion_plot(picked),
            "gap_json":       build_gap_plot(picked),
            "accuracy":       round(picked["test_acc"] * 100, 2),
            "complexity":     int(picked["n_leaves"]),
            "complexity_label": "Leaves",
        })

    elif model_type == "lr":
        picked = get_best_lr(lam)

        return JsonResponse({
            "plotly_json":    build_lr_plotly(picked, sort_by=sort_by, filter_class=filter_class, filter_feature=filter_feature),
            "tradeoff_json":  lr_tradeoff(picked),
            "confusion_json": lr_confusion(picked),
            "gap_json":       lr_gap(picked),
            "accuracy":       round(picked["test_acc"] * 100, 2),
            "complexity":     round(picked["omega"], 4),
            "complexity_label": "‖w‖₁",
        })

    return JsonResponse({"error": "unknown model type"}, status=400)