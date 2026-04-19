from django.shortcuts import render


def index(request):
    context = {
        "title": "Project 1 — Automated Machine Learning",
        "description": (
            "an interface for a simple supervised learning setting"
        ),
    }
    return render(request, "project1/index.html", context)
