from django.urls import path
from . import views

app_name = "project4"

urlpatterns = [
    path("", views.landing, name="landing"),
    path("report.pdf", views.download_report, name="report"),

    path("study/consent/", views.consent, name="consent"),
    path("study/background/", views.background, name="background"),
    path("study/instructions/", views.instructions, name="instructions"),
    path("study/trials/", views.trials, name="trials"),
    path("study/trials/submit/", views.submit_trial, name="submit_trial"),
    path("study/questionnaire/", views.questionnaire, name="questionnaire"),
    path("study/validation/", views.validation, name="validation"),
    path("study/validation/submit/", views.submit_validation, name="submit_validation"),
    path("study/final/", views.final, name="final"),
    path("study/done/", views.done, name="done"),
]
