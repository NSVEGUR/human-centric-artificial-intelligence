import json

import numpy as np
from django.test import TestCase

URL_INDEX = '/project3/'
URL_CLASSIFIER_STATS = '/project3/classifier-stats/'
URL_EXPERT_STATS = '/project3/expert-stats/'
URL_DEFERRAL_STATS = '/project3/deferral-stats/'
URL_AL_STATS = '/project3/al-stats/'
URL_REPORT = '/project3/report.pdf'
URL_HUMAN_LABEL = '/project3/human-label/'
URL_HUMAN_LABEL_SUBMIT = '/project3/human-label/submit/'
URL_HUMAN_LABEL_STRATEGY = '/project3/human-label/strategy/'
URL_HUMAN_LABEL_RESET = '/project3/human-label/reset/'


class TestProject3Index(TestCase):
    """Task 1-4: the landing page must render and expose the stats every
    task's section is built from."""

    def test_get_200(self):
        response = self.client.get(URL_INDEX)
        self.assertEqual(response.status_code, 200)

    def test_context_has_all_task_stats(self):
        response = self.client.get(URL_INDEX)
        # Task 1
        self.assertIn('test_acc', response.context)
        self.assertIn('label_names', response.context)
        self.assertIn('confusion_matrix_json', response.context)
        # Task 2
        self.assertIn('experts', response.context)
        self.assertEqual(len(response.context['experts']), 2)
        # Task 3 - must report more than a single accuracy number: coverage
        # and deferral rate characterise the *quality* of the deferral
        # decisions, not just team accuracy.
        deferral_stats = response.context['deferral_stats']
        self.assertIn('optimal', deferral_stats)
        self.assertIn('team_acc', deferral_stats['optimal'])
        self.assertIn('coverage', deferral_stats['optimal'])
        self.assertIn('deferral_rate', deferral_stats['optimal'])
        self.assertIn('baselines', deferral_stats)
        self.assertIn('coverages', deferral_stats)
        self.assertIn('team_accuracies', deferral_stats)
        # Task 4
        al_stats = response.context['al_stats']
        self.assertIn('strategies', al_stats)
        self.assertIn('oracle_acc', al_stats)

    def test_page_contains_plotly_data(self):
        response = self.client.get(URL_INDEX)
        self.assertIn(b'plotly', response.content.lower())

    def test_report_download_link_present(self):
        response = self.client.get(URL_INDEX)
        self.assertContains(response, '/project3/report.pdf')


class TestProject3JsonEndpoints(TestCase):
    def test_classifier_stats_get(self):
        response = self.client.get(URL_CLASSIFIER_STATS)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('test_acc', data)
        self.assertIn('conf_matrix', data)

    def test_classifier_stats_post_rejected(self):
        response = self.client.post(URL_CLASSIFIER_STATS)
        self.assertEqual(response.status_code, 405)

    def test_expert_stats_get(self):
        response = self.client.get(URL_EXPERT_STATS)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(len(data['experts']), 2)
        for expert in data['experts']:
            self.assertIn('per_class_acc', expert)
            self.assertIn('strong_class', expert)

    def test_deferral_stats_get(self):
        response = self.client.get(URL_DEFERRAL_STATS)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('optimal', data)
        self.assertIn('baselines', data)

    def test_al_stats_get(self):
        response = self.client.get(URL_AL_STATS)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('strategies', data)
        self.assertIn('curves', data)
        self.assertIn('oracle_acc', data)

    def test_al_stats_post_rejected(self):
        response = self.client.post(URL_AL_STATS)
        self.assertEqual(response.status_code, 405)


class TestProject3Report(TestCase):
    """The PDF report must be genuinely downloadable and non-trivial, per
    the assignment preamble ("the report must be accessible from the
    project interface, for instance with a download button")."""

    def test_report_is_pdf(self):
        response = self.client.get(URL_REPORT)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/pdf')
        self.assertGreater(len(response.content), 1000)

    def test_report_is_downloadable_attachment(self):
        response = self.client.get(URL_REPORT)
        self.assertIn('attachment', response['Content-Disposition'])
        self.assertIn('.pdf', response['Content-Disposition'])

    def test_report_starts_with_pdf_magic_bytes(self):
        response = self.client.get(URL_REPORT)
        self.assertTrue(response.content.startswith(b'%PDF'))


class TestDeferralLogic(TestCase):
    """Task 3: the Bayes-optimal deferral rule (defer iff classifier
    uncertainty exceeds expert expected error) must actually respond to its
    two inputs. We exercise the real production function used by both the
    Task 4 active-learning loop and the Task 5 human-labeling flow,
    `_evaluate_deferral`, on tiny synthetic examples rather than the full
    trained pipeline.
    """

    @staticmethod
    def _team_accuracy(est_per_class_acc, probas, expert_preds, labels):
        from project3.active_learning import _evaluate_deferral
        return _evaluate_deferral(
            np.array(est_per_class_acc),
            np.array(probas),
            np.array(expert_preds),
            np.array(labels),
        )

    def test_defer_decision_flips_with_classifier_confidence(self):
        # Expert competence held perfectly constant across both cases: a
        # uniform per-class accuracy vector always yields the same expert
        # error (0.3) no matter how probability mass is distributed, since
        # the weights sum to 1. Only the classifier's confidence differs.
        est_uniform = [0.7, 0.7, 0.7, 0.7]
        true_label = [0]
        expert_pred = [0]  # expert is correct on this instance

        # Classifier is confident but WRONG (predicts class 1).
        proba_high_conf = [[0.05, 0.85, 0.05, 0.05]]
        acc_high_conf = self._team_accuracy(est_uniform, proba_high_conf, expert_pred, true_label)

        # Same wrong prediction, but much less confident about it.
        proba_low_conf = [[0.25, 0.35, 0.2, 0.2]]
        acc_low_conf = self._team_accuracy(est_uniform, proba_low_conf, expert_pred, true_label)

        # High confidence -> trust the (wrong) classifier -> team is wrong.
        self.assertEqual(acc_high_conf, 0.0)
        # Low confidence -> defer to the (correct) expert -> team is right.
        self.assertEqual(acc_low_conf, 100.0)

    def test_defer_decision_flips_with_expert_accuracy(self):
        # Classifier confidence held constant across both cases (always
        # wrong, predicting class 0 with uncertainty 0.6). Only the
        # expert's estimated per-class competence differs.
        proba_fixed = [[0.4, 0.3, 0.15, 0.15]]
        true_label = [1]
        expert_pred = [1]  # expert is correct on this instance

        est_good_expert = [0.5, 0.95, 0.5, 0.5]
        acc_good_expert = self._team_accuracy(est_good_expert, proba_fixed, expert_pred, true_label)

        est_bad_expert = [0.5, 0.05, 0.5, 0.5]
        acc_bad_expert = self._team_accuracy(est_bad_expert, proba_fixed, expert_pred, true_label)

        # High expert competence -> defer -> team is right.
        self.assertEqual(acc_good_expert, 100.0)
        # Low expert competence -> trust the classifier instead -> team is wrong.
        self.assertEqual(acc_bad_expert, 0.0)

    def test_alpha_sweep_coverage_is_monotonic(self):
        # From deferral.py: larger alpha requires more expert error to
        # trigger a defer, so coverage (fraction handled by AI) should be
        # non-decreasing in alpha.
        from project3.deferral import coverages
        self.assertTrue(all(
            coverages[i] <= coverages[i + 1] + 1e-9 for i in range(len(coverages) - 1)
        ))

    def test_optimal_team_beats_or_matches_ai_alone(self):
        # Task 3 requirement: the human-AI team should match/beat the
        # Task 1 baseline.
        from project3.deferral import ai_only_acc, optimal_team_acc
        self.assertGreaterEqual(optimal_team_acc, ai_only_acc)


class TestHumanLabelFlow(TestCase):
    """Task 5: a human plays the role of the expert through the interactive
    interface. Drives the actual AJAX endpoints with the Django test
    client, mirroring what human_label.html's fetch() calls do."""

    def test_page_loads_and_initialises_session(self):
        response = self.client.get(URL_HUMAN_LABEL)
        self.assertEqual(response.status_code, 200)
        session = self.client.session
        self.assertEqual(session['al_labeled'], [])
        self.assertEqual(session['al_strategy'], 'entropy')

    def test_submit_label_is_recorded_and_reflected(self):
        response = self.client.get(URL_HUMAN_LABEL)
        current = response.context['current_article']
        self.assertIsNotNone(current)
        idx = current['idx']

        # Get the true label so we can submit a *correct* answer and check
        # the "correct" flag flows through end to end.
        from project3.active_learning import pool_labels
        true_label = int(pool_labels[idx])

        submit_response = self.client.post(
            URL_HUMAN_LABEL_SUBMIT,
            data=json.dumps({'idx': idx, 'label': true_label}),
            content_type='application/json',
        )
        self.assertEqual(submit_response.status_code, 200)
        data = submit_response.json()
        self.assertTrue(data['submitted'])
        self.assertTrue(data['correct'])
        self.assertEqual(data['n_labeled'], 1)

        # Recorded in the session.
        session = self.client.session
        self.assertEqual(len(session['al_labeled']), 1)
        self.assertEqual(session['al_labeled'][0]['idx'], idx)
        self.assertEqual(session['al_queried'], [idx])

        # Reflected in the live team-accuracy estimate on a follow-up GET.
        page_response = self.client.get(URL_HUMAN_LABEL)
        self.assertEqual(page_response.context['n_labeled'], 1)
        self.assertIsNotNone(page_response.context['team_acc'])

    def test_submit_advances_to_a_different_article(self):
        response = self.client.get(URL_HUMAN_LABEL)
        first_idx = response.context['current_article']['idx']

        submit_response = self.client.post(
            URL_HUMAN_LABEL_SUBMIT,
            data=json.dumps({'idx': first_idx, 'label': 0}),
            content_type='application/json',
        )
        data = submit_response.json()
        self.assertIsNotNone(data['next_article'])
        self.assertNotEqual(data['next_article']['idx'], first_idx)

    def test_submit_invalid_payload_rejected(self):
        response = self.client.post(
            URL_HUMAN_LABEL_SUBMIT,
            data=json.dumps({'idx': 'not-an-int', 'label': 0}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)

    def test_submit_out_of_range_label_rejected(self):
        self.client.get(URL_HUMAN_LABEL)
        response = self.client.post(
            URL_HUMAN_LABEL_SUBMIT,
            data=json.dumps({'idx': 0, 'label': 99}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)

    def test_submit_get_rejected(self):
        response = self.client.get(URL_HUMAN_LABEL_SUBMIT)
        self.assertEqual(response.status_code, 405)

    def test_change_strategy(self):
        self.client.get(URL_HUMAN_LABEL)
        response = self.client.post(
            URL_HUMAN_LABEL_STRATEGY,
            data=json.dumps({'strategy': 'random'}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()['strategy'], 'random')
        self.assertEqual(self.client.session['al_strategy'], 'random')

    def test_change_strategy_rejects_unknown(self):
        response = self.client.post(
            URL_HUMAN_LABEL_STRATEGY,
            data=json.dumps({'strategy': 'bogus'}),
            content_type='application/json',
        )
        self.assertEqual(response.status_code, 400)

    def test_reset_clears_session(self):
        response = self.client.get(URL_HUMAN_LABEL)
        idx = response.context['current_article']['idx']
        self.client.post(
            URL_HUMAN_LABEL_SUBMIT,
            data=json.dumps({'idx': idx, 'label': 0}),
            content_type='application/json',
        )
        self.assertEqual(len(self.client.session['al_labeled']), 1)

        reset_response = self.client.post(URL_HUMAN_LABEL_RESET)
        self.assertEqual(reset_response.status_code, 200)
        self.assertEqual(self.client.session['al_labeled'], [])
        self.assertEqual(self.client.session['al_queried'], [])

    def test_reset_get_rejected(self):
        response = self.client.get(URL_HUMAN_LABEL_RESET)
        self.assertEqual(response.status_code, 405)

    def test_profile_unlocks_after_minimum_labels(self):
        response = self.client.get(URL_HUMAN_LABEL)
        from project3.views import MIN_LABELS_FOR_PROFILE

        last_data = None
        for _ in range(MIN_LABELS_FOR_PROFILE):
            current = response.context['current_article'] if response.context.get('current_article') else None
            if current is None:
                break
            idx = current['idx']
            submit_response = self.client.post(
                URL_HUMAN_LABEL_SUBMIT,
                data=json.dumps({'idx': idx, 'label': 0}),
                content_type='application/json',
            )
            last_data = submit_response.json()
            response = self.client.get(URL_HUMAN_LABEL)

        self.assertIsNotNone(last_data)
        self.assertTrue(last_data['profile']['unlocked'])
        self.assertIn('you_team_acc', last_data['profile'])
        self.assertIn('specialty', last_data['profile'])
