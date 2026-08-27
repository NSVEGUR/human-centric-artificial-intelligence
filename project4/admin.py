import csv

from django.contrib import admin
from django.http import HttpResponse

from .models import Participant, Trial, QuestionnaireResponse, FinalResponse


def export_trials_csv(modeladmin, request, queryset):
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = 'attachment; filename="project4_trials.csv"'
    writer = csv.writer(response)
    writer.writerow(["participant_id", "design", "phase", "trial_index", "movie_ids", "response", "response_time_ms", "is_correct", "created_at"])
    for t in queryset:
        writer.writerow([t.participant_id, t.design, t.phase, t.trial_index, t.movie_ids, t.response, t.response_time_ms, t.is_correct, t.created_at])
    return response


export_trials_csv.short_description = "Export selected trials as CSV"


@admin.register(Participant)
class ParticipantAdmin(admin.ModelAdmin):
    list_display = (
        "id", "condition_order", "created_at", "consented_at", "completed_at", "completion_code",
        "age_bracket", "movie_frequency", "recsys_familiarity",
    )
    list_filter = ("condition_order", "age_bracket", "movie_frequency", "recsys_familiarity")


@admin.register(Trial)
class TrialAdmin(admin.ModelAdmin):
    list_display = ("participant", "design", "phase", "trial_index", "response_time_ms", "is_correct", "created_at")
    list_filter = ("design", "phase", "is_correct")
    actions = [export_trials_csv]


@admin.register(QuestionnaireResponse)
class QuestionnaireResponseAdmin(admin.ModelAdmin):
    list_display = ("participant", "design", "ease_of_use", "cognitive_load", "enjoyment", "trust")
    list_filter = ("design",)


@admin.register(FinalResponse)
class FinalResponseAdmin(admin.ModelAdmin):
    list_display = ("participant", "preferred_design", "pairwise_validation_accuracy", "ranking_validation_accuracy")
