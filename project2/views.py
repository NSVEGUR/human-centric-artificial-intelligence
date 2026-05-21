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
from .counterfactuals import (
    get_test_samples, generate_counterfactuals,
    build_counterfactual_table_viz
)
from .feature_effects import (
    compute_pdp, compute_ale,
    build_pdp_plot, build_ale_plot,
    build_combined_feature_effects_plot,
    build_all_features_importance_plot,
    num_cols as numerical_features
)


def index(request):
    template = loader.get_template("project2/index.html")

    # default to best tree at lambda=0 (most accurate one)
    default_tree = get_best_tree(0.0)

    if default_tree is None:
        return JsonResponse({"error": "no model found"}, status=400)

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
        
        if picked is None:
            return JsonResponse({"error": "no model found"}, status=400)

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
        
        if picked is None:
            return JsonResponse({"error": "no model found"}, status=400)

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


@csrf_exempt
def get_samples(request):
    """
    Return list of test samples for the counterfactual dropdown.
    """
    if request.method != "GET":
        return JsonResponse({"error": "only GET allowed"}, status=405)

    samples = get_test_samples()
    return JsonResponse({"samples": samples})


@csrf_exempt
def counterfactuals(request):
    """
    Generate counterfactual explanations for a selected sample.

    Expects POST with:
    - sample_idx: index of the test sample
    - target_class: desired prediction class
    - model_type: 'tree' or 'lr'
    - lambda_val: regularization parameter
    - k: number of counterfactuals to return (optional, default 5)
    """
    if request.method != "POST":
        return JsonResponse({"error": "only POST allowed"}, status=405)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "bad json"}, status=400)

    sample_idx = int(body.get("sample_idx", 0))
    target_class = body.get("target_class", "Adelie")
    model_type = body.get("model_type", "tree")
    lam = float(body.get("lambda_val", 0.0))
    k = int(body.get("k", 5))

    # Get the selected model
    if model_type == "tree":
        model_info = get_best_tree(lam)
    else:
        model_info = get_best_lr(lam)

    # Generate counterfactuals
    try:
        result = generate_counterfactuals(
            model_info=model_info,
            sample_idx=sample_idx,
            target_class=target_class,
            model_type=model_type,
            k=k
        )

        # Also build the table visualization
        table_viz = build_counterfactual_table_viz(
            result['original'],
            result['counterfactuals'],
            target_class
        )
        result['table_visualization'] = table_viz

        return JsonResponse(result)

    except Exception as e:
        import traceback
        return JsonResponse({
            "error": str(e),
            "traceback": traceback.format_exc()
        }, status=500)


@csrf_exempt
def feature_effects(request):
    """
    Compute PDP and ALE for a selected feature.

    Expects POST with:
    - feature: name of the numerical feature
    - model_type: 'tree' or 'lr'
    - lambda_val: regularization parameter
    - show_ice: whether to include ICE curves (optional)
    """
    if request.method != "POST":
        return JsonResponse({"error": "only POST allowed"}, status=405)

    try:
        body = json.loads(request.body)
    except json.JSONDecodeError:
        return JsonResponse({"error": "bad json"}, status=400)

    feature = body.get("feature", "bill_length_mm")
    model_type = body.get("model_type", "tree")
    lam = float(body.get("lambda_val", 0.0))
    show_ice = body.get("show_ice", False)

    # Validate feature
    if feature not in numerical_features:
        return JsonResponse({
            "error": f"Invalid feature. Must be one of: {numerical_features}"
        }, status=400)

    # Get the selected model
    if model_type == "tree":
        model_info = get_best_tree(lam)
    else:
        model_info = get_best_lr(lam)

    if model_info is None:
        return JsonResponse({"error": "Failed to load model"}, status=500)

    model = model_info['model']

    try:
        # Compute PDP and ALE
        pdp_data = compute_pdp(model, feature, model_type)
        ale_data = compute_ale(model, feature, model_type)

        # Build visualizations
        pdp_plot = build_pdp_plot(pdp_data, show_ice=show_ice)
        ale_plot = build_ale_plot(ale_data)
        combined_plot = build_combined_feature_effects_plot(model, feature, model_type, show_ice=show_ice)

        # Feature importance
        importance_plot = build_all_features_importance_plot(model, model_type)

        return JsonResponse({
            "feature": feature,
            "model_type": model_type,
            "derivative_type": ale_data.get('derivative_type', 'unknown'),
            "pdp_plot": pdp_plot,
            "ale_plot": ale_plot,
            "combined_plot": combined_plot,
            "importance_plot": importance_plot,
        })

    except Exception as e:
        import traceback
        return JsonResponse({
            "error": str(e),
            "traceback": traceback.format_exc()
        }, status=500)


@csrf_exempt
def get_numerical_features(request):
    """
    Return list of numerical features for the feature effects dropdown.
    """
    if request.method != "GET":
        return JsonResponse({"error": "only GET allowed"}, status=405)

    features = [
        {"key": "bill_length_mm", "display": "Bill Length (mm)"},
        {"key": "bill_depth_mm", "display": "Bill Depth (mm)"},
        {"key": "flipper_length_mm", "display": "Flipper Length (mm)"},
        {"key": "body_mass_g", "display": "Body Mass (g)"},
    ]
    return JsonResponse({"features": features})