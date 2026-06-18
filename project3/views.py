import json
from django.http import HttpResponse, JsonResponse
from django.template import loader
from django.views.decorators.csrf import csrf_exempt

from .classifier import get_classifier_stats, build_confusion_matrix_plot
from .experts import get_expert_stats, build_expert_accuracy_plot


def index(request):
    template = loader.get_template("project3/index.html")

    classifier_stats = get_classifier_stats()
    expert_stats = get_expert_stats()

    context = {
        "test_acc": classifier_stats["test_acc"],
        "label_names": classifier_stats["label_names"],
        "confusion_matrix_json": build_confusion_matrix_plot(),
        "expert_accuracy_json": build_expert_accuracy_plot(),
        "experts": expert_stats["experts"],
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