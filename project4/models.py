import uuid

from django.db import models


class Participant(models.Model):
    """One respondent of the preference-elicitation study.

    A real deployment of this interface (see report, Task 3) would persist
    exactly these rows; this model is what lets the study actually be run
    without further modification, not just previewed.
    """

    ORDER_CHOICES = [
        ("pairwise_first", "Pairwise comparisons first"),
        ("ranking_first", "Ranking lists first"),
    ]
    AGE_CHOICES = [
        ("18-24", "18-24"), ("25-34", "25-34"), ("35-44", "35-44"),
        ("45-54", "45-54"), ("55+", "55+"), ("na", "Prefer not to say"),
    ]
    FREQUENCY_CHOICES = [
        ("daily", "Daily"), ("weekly", "A few times a week"),
        ("monthly", "A few times a month"), ("rarely", "Rarely"),
    ]
    FAMILIARITY_CHOICES = [
        ("none", "Never used one"), ("some", "Used one occasionally"),
        ("regular", "Use one regularly (e.g. Netflix/IMDb recommendations)"),
    ]

    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    created_at = models.DateTimeField(auto_now_add=True)
    condition_order = models.CharField(max_length=20, choices=ORDER_CHOICES)
    consented_at = models.DateTimeField(null=True, blank=True)
    prolific_id = models.CharField(max_length=100, blank=True)
    completed_at = models.DateTimeField(null=True, blank=True)
    completion_code = models.CharField(max_length=12, blank=True)

    # Brief background survey (Lecture 7, "6 steps": general + task-relevant
    # info), collected once, right after consent — used only to describe the
    # cohort (Task 3, analysis plan) and never joined back to trial data.
    age_bracket = models.CharField(max_length=10, choices=AGE_CHOICES, blank=True)
    movie_frequency = models.CharField(max_length=10, choices=FREQUENCY_CHOICES, blank=True)
    recsys_familiarity = models.CharField(max_length=10, choices=FAMILIARITY_CHOICES, blank=True)

    def __str__(self):
        return f"Participant {self.id} ({self.condition_order})"


class Trial(models.Model):
    DESIGN_CHOICES = [("pairwise", "Pairwise"), ("ranking", "Ranking")]
    PHASE_CHOICES = [
        ("practice", "Practice / attention check"),
        ("elicitation", "Elicitation"),
        ("validation", "Held-out validation"),
    ]

    participant = models.ForeignKey(Participant, on_delete=models.CASCADE, related_name="trials")
    design = models.CharField(max_length=10, choices=DESIGN_CHOICES)
    phase = models.CharField(max_length=15, choices=PHASE_CHOICES)
    trial_index = models.IntegerField()
    movie_ids = models.JSONField()          # items shown, in presentation order
    response = models.JSONField()           # pairwise: chosen movie id; ranking: ordered movie-id list
    response_time_ms = models.IntegerField(null=True, blank=True)
    is_correct = models.BooleanField(
        null=True, blank=True,
        help_text="Only meaningful for phase='practice': did the response match the "
                   "instructed target? Used as the pre-registered attention-check "
                   "exclusion criterion (report, section 3.5).",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["participant_id", "created_at"]

    def __str__(self):
        return f"{self.design}/{self.phase} #{self.trial_index} for {self.participant_id}"


class QuestionnaireResponse(models.Model):
    DESIGN_CHOICES = Trial.DESIGN_CHOICES

    participant = models.ForeignKey(Participant, on_delete=models.CASCADE, related_name="questionnaires")
    design = models.CharField(max_length=10, choices=DESIGN_CHOICES)
    ease_of_use = models.IntegerField()      # 1-7 Likert
    cognitive_load = models.IntegerField()   # 1-7 Likert
    enjoyment = models.IntegerField()        # 1-7 Likert
    trust = models.IntegerField()            # 1-7 Likert, "system understands my taste"
    free_text = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("participant", "design")

    def __str__(self):
        return f"Questionnaire({self.design}) for {self.participant_id}"


class FinalResponse(models.Model):
    participant = models.OneToOneField(Participant, on_delete=models.CASCADE, related_name="final_response")
    preferred_design = models.CharField(max_length=10, choices=Trial.DESIGN_CHOICES)
    preferred_reason = models.TextField(blank=True)
    pairwise_validation_accuracy = models.FloatField(null=True, blank=True)
    ranking_validation_accuracy = models.FloatField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"FinalResponse for {self.participant_id}"
