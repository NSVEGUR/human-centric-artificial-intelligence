from django.test import TestCase, Client

from project4.models import Participant, Trial, QuestionnaireResponse, FinalResponse


class TestLandingAndReport(TestCase):
    def test_landing_200(self):
        response = self.client.get("/project4/")
        self.assertEqual(response.status_code, 200)

    def test_report_is_pdf(self):
        response = self.client.get("/project4/report.pdf")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "application/pdf")
        self.assertGreater(len(response.content), 1000)


class TestGuards(TestCase):
    def test_trials_without_consent_redirects_to_consent(self):
        response = self.client.get("/project4/study/trials/")
        self.assertRedirects(response, "/project4/study/consent/")

    def test_questionnaire_without_consent_redirects_to_consent(self):
        response = self.client.get("/project4/study/questionnaire/")
        self.assertRedirects(response, "/project4/study/consent/")


class TestFullStudyFlow(TestCase):
    """Drives one participant through the entire state machine, matching
    the manual walkthrough used to validate the implementation."""

    def setUp(self):
        self.client = Client()

    def _consent(self):
        response = self.client.get("/project4/study/consent/")
        csrf = response.context["csrf_token"]
        response = self.client.post("/project4/study/consent/", {"agree": "on", "csrfmiddlewaretoken": csrf})
        self.assertRedirects(response, "/project4/study/background/")
        self._submit_background()

    def _submit_background(self):
        response = self.client.get("/project4/study/background/")
        csrf = response.context["csrf_token"]
        response = self.client.post("/project4/study/background/", {
            "csrfmiddlewaretoken": csrf, "age_bracket": "25-34",
            "movie_frequency": "weekly", "recsys_familiarity": "regular",
        })
        self.assertRedirects(response, "/project4/study/instructions/")

    def _advance_instructions(self):
        response = self.client.get("/project4/study/instructions/")
        csrf = response.context["csrf_token"]
        response = self.client.post("/project4/study/instructions/", {"csrfmiddlewaretoken": csrf})
        self.assertRedirects(response, "/project4/study/trials/")

    def _run_trials(self, design, response_builder):
        n = 0
        while True:
            response = self.client.get("/project4/study/trials/")
            self.assertEqual(response.status_code, 200)
            trial = response.context["trial"]
            body = response_builder(trial)
            result = self.client.post(
                "/project4/study/trials/submit/", body, content_type="application/json"
            ).json()
            n += 1
            if result["done"]:
                self.assertEqual(result["redirect"], "/project4/study/questionnaire/")
                break
        return n

    def _submit_questionnaire(self):
        response = self.client.get("/project4/study/questionnaire/")
        csrf = response.context["csrf_token"]
        response = self.client.post("/project4/study/questionnaire/", {
            "csrfmiddlewaretoken": csrf, "ease_of_use": 5, "cognitive_load": 4,
            "enjoyment": 5, "trust": 4, "free_text": "",
        })
        self.assertEqual(response.status_code, 302)

    def test_end_to_end(self):
        self._consent()
        participant = Participant.objects.get()
        self.assertIn(participant.condition_order, ("pairwise_first", "ranking_first"))

        first_design = "pairwise" if participant.condition_order == "pairwise_first" else "ranking"
        second_design = "ranking" if first_design == "pairwise" else "pairwise"

        # condition 1
        self._advance_instructions()
        n1 = self._run_trials(first_design, self._pairwise_body if first_design == "pairwise" else self._ranking_body)
        self._submit_questionnaire()

        # condition 2
        self._advance_instructions()
        n2 = self._run_trials(second_design, self._pairwise_body if second_design == "pairwise" else self._ranking_body)
        self._submit_questionnaire()

        self.assertGreater(n1, 0)
        self.assertGreater(n2, 0)

        # validation block
        response = self.client.get("/project4/study/validation/")
        self.assertEqual(response.status_code, 200)
        while True:
            response = self.client.get("/project4/study/validation/")
            trial = response.context["trial"]
            chosen = trial["movies"][0]["id"]
            result = self.client.post(
                "/project4/study/validation/submit/",
                {"chosen_id": chosen, "response_time_ms": 300},
                content_type="application/json",
            ).json()
            if result["done"]:
                self.assertEqual(result["redirect"], "/project4/study/final/")
                break

        # final
        response = self.client.get("/project4/study/final/")
        csrf = response.context["csrf_token"]
        response = self.client.post("/project4/study/final/", {
            "csrfmiddlewaretoken": csrf, "preferred_design": "pairwise", "preferred_reason": "quick",
        })
        self.assertRedirects(response, "/project4/study/done/")

        response = self.client.get("/project4/study/done/")
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "P4-")

        participant.refresh_from_db()
        self.assertIsNotNone(participant.completed_at)
        self.assertTrue(participant.completion_code.startswith("P4-"))

        self.assertEqual(QuestionnaireResponse.objects.filter(participant=participant).count(), 2)
        self.assertEqual(FinalResponse.objects.filter(participant=participant).count(), 1)
        self.assertGreater(Trial.objects.filter(participant=participant).count(), 0)

        self.assertEqual(participant.age_bracket, "25-34")
        self.assertEqual(participant.movie_frequency, "weekly")
        self.assertEqual(participant.recsys_familiarity, "regular")

    @staticmethod
    def _pairwise_body(trial):
        return {"chosen_id": trial["movies"][0]["id"], "response_time_ms": 500}

    @staticmethod
    def _ranking_body(trial):
        return {"order": [m["id"] for m in trial["movies"]], "response_time_ms": 900}


class TestAttentionCheck(TestCase):
    """Practice trials are the pre-registered attention check (report, section
    3.5): the response is scored against the instructed target and stored on
    the Trial row so it can be filtered on directly in the CSV export."""

    def setUp(self):
        self.client = Client()
        response = self.client.get("/project4/study/consent/")
        csrf = response.context["csrf_token"]
        response = self.client.post("/project4/study/consent/", {"agree": "on", "csrfmiddlewaretoken": csrf})
        response = self.client.get("/project4/study/background/")
        csrf = response.context["csrf_token"]
        self.client.post("/project4/study/background/", {
            "csrfmiddlewaretoken": csrf, "age_bracket": "25-34",
            "movie_frequency": "weekly", "recsys_familiarity": "regular",
        })
        response = self.client.get("/project4/study/instructions/")
        csrf = response.context["csrf_token"]
        self.client.post("/project4/study/instructions/", {"csrfmiddlewaretoken": csrf})

    def test_pairwise_or_ranking_practice_scored_correctly(self):
        response = self.client.get("/project4/study/trials/")
        trial = response.context["trial"]
        self.assertEqual(trial["phase"], "practice")

        if trial["design"] == "pairwise":
            correct_body = {"chosen_id": trial["target_id"], "response_time_ms": 500}
        else:
            other_ids = [m["id"] for m in trial["movies"] if m["id"] != trial["target_id"]]
            correct_body = {"order": [trial["target_id"]] + other_ids, "response_time_ms": 900}

        self.client.post("/project4/study/trials/submit/", correct_body, content_type="application/json")
        practice_trial = Trial.objects.get(phase="practice")
        self.assertTrue(practice_trial.is_correct)


class TestCounterbalancing(TestCase):
    def test_order_alternates(self):
        orders = []
        for _ in range(4):
            client = Client()
            response = client.get("/project4/study/consent/")
            csrf = response.context["csrf_token"]
            client.post("/project4/study/consent/", {"agree": "on", "csrfmiddlewaretoken": csrf})
        orders = list(Participant.objects.order_by("created_at").values_list("condition_order", flat=True))
        self.assertEqual(orders, [
            "pairwise_first", "ranking_first", "pairwise_first", "ranking_first",
        ])
