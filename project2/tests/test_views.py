import json
from django.test import TestCase

URL_INDEX = '/project2/'
URL_UPDATE = '/project2/update-model/'
URL_SAMPLES = '/project2/get-samples/'
URL_FEATURES = '/project2/get-features/'
URL_CF = '/project2/counterfactuals/'
URL_FE = '/project2/feature-effects/'


class TestProject2Index(TestCase):
    def test_get_200(self):
        response = self.client.get(URL_INDEX)
        self.assertEqual(response.status_code, 200)

    def test_initial_context_has_chart_data(self):
        response = self.client.get(URL_INDEX)
        self.assertIn('initial_accuracy', response.context)
        self.assertIn('initial_complexity', response.context)
        self.assertIn('initial_plot', response.context)

    def test_page_contains_plotly_data(self):
        response = self.client.get(URL_INDEX)
        self.assertIn(b'plotly', response.content.lower())


class TestProject2UpdateModel(TestCase):
    def _post(self, body):
        return self.client.post(
            URL_UPDATE,
            data=json.dumps(body),
            content_type='application/json',
        )

    def test_tree_default_lambda(self):
        response = self._post({'model_type': 'tree', 'lambda_val': 0.0})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('plotly_json', data)
        self.assertIn('accuracy', data)
        self.assertIn('complexity', data)
        self.assertEqual(data['complexity_label'], 'Leaves')

    def test_tree_tradeoff_and_confusion_included(self):
        response = self._post({'model_type': 'tree', 'lambda_val': 0.0})
        data = response.json()
        self.assertIn('tradeoff_json', data)
        self.assertIn('confusion_json', data)
        self.assertIn('gap_json', data)

    def test_lr_model(self):
        response = self._post({'model_type': 'lr', 'lambda_val': 0.0})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('plotly_json', data)
        self.assertIn('accuracy', data)
        self.assertIn('‖w‖₁', data['complexity_label'])

    def test_higher_lambda_gives_simpler_tree(self):
        r0 = self._post({'model_type': 'tree', 'lambda_val': 0.0})
        r10 = self._post({'model_type': 'tree', 'lambda_val': 10.0})
        self.assertEqual(r0.status_code, 200)
        self.assertEqual(r10.status_code, 200)
        # higher λ penalises complexity → fewer or equal leaves
        self.assertLessEqual(r10.json()['complexity'], r0.json()['complexity'])

    def test_get_method_rejected(self):
        response = self.client.get(URL_UPDATE)
        self.assertEqual(response.status_code, 405)

    def test_bad_json_returns_400(self):
        response = self.client.post(URL_UPDATE, data='not json', content_type='application/json')
        self.assertEqual(response.status_code, 400)

    def test_unknown_model_type_returns_400(self):
        response = self._post({'model_type': 'svm', 'lambda_val': 0.0})
        self.assertEqual(response.status_code, 400)


class TestProject2GetSamples(TestCase):
    def test_returns_non_empty_samples_list(self):
        response = self.client.get(URL_SAMPLES)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('samples', data)
        self.assertGreater(len(data['samples']), 0)

    def test_post_rejected(self):
        response = self.client.post(URL_SAMPLES, data='{}', content_type='application/json')
        self.assertEqual(response.status_code, 405)


class TestProject2GetFeatures(TestCase):
    def test_returns_numerical_features(self):
        response = self.client.get(URL_FEATURES)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('features', data)
        keys = [f['key'] for f in data['features']]
        self.assertIn('bill_length_mm', keys)
        self.assertIn('flipper_length_mm', keys)

    def test_post_rejected(self):
        response = self.client.post(URL_FEATURES, data='{}', content_type='application/json')
        self.assertEqual(response.status_code, 405)


class TestProject2Counterfactuals(TestCase):
    def _post(self, body):
        return self.client.post(
            URL_CF,
            data=json.dumps(body),
            content_type='application/json',
        )

    def test_tree_counterfactuals_structure(self):
        response = self._post({
            'sample_idx': 0,
            'target_class': 'Gentoo',
            'model_type': 'tree',
            'lambda_val': 0.0,
            'k': 3,
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertNotIn('error', data)
        self.assertIn('counterfactuals', data)
        self.assertIn('original', data)
        self.assertIn('table_visualization', data)

    def test_lr_counterfactuals_no_error(self):
        response = self._post({
            'sample_idx': 0,
            'target_class': 'Gentoo',
            'model_type': 'lr',
            'lambda_val': 0.0,
            'k': 3,
        })
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('error', response.json())

    def test_get_rejected(self):
        response = self.client.get(URL_CF)
        self.assertEqual(response.status_code, 405)

    def test_bad_json_returns_400(self):
        response = self.client.post(URL_CF, data='not json', content_type='application/json')
        self.assertEqual(response.status_code, 400)


class TestProject2FeatureEffects(TestCase):
    def _post(self, body):
        return self.client.post(
            URL_FE,
            data=json.dumps(body),
            content_type='application/json',
        )

    def test_tree_pdp_ale_importance(self):
        response = self._post({
            'feature': 'bill_length_mm',
            'model_type': 'tree',
            'lambda_val': 0.0,
            'show_ice': False,
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('pdp_plot', data)
        self.assertIn('ale_plot', data)
        self.assertIn('importance_plot', data)
        self.assertIn('derivative_type', data)

    def test_lr_derivative_type_present(self):
        response = self._post({
            'feature': 'bill_depth_mm',
            'model_type': 'lr',
            'lambda_val': 0.0,
            'show_ice': False,
        })
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn('derivative_type', data)

    def test_ice_curves_no_error(self):
        response = self._post({
            'feature': 'flipper_length_mm',
            'model_type': 'tree',
            'lambda_val': 0.0,
            'show_ice': True,
        })
        self.assertEqual(response.status_code, 200)
        self.assertNotIn('error', response.json())

    def test_all_numerical_features_work(self):
        for feature in ('bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g'):
            with self.subTest(feature=feature):
                response = self._post({
                    'feature': feature,
                    'model_type': 'tree',
                    'lambda_val': 0.0,
                })
                self.assertEqual(response.status_code, 200)

    def test_invalid_feature_returns_400(self):
        response = self._post({
            'feature': 'not_a_feature',
            'model_type': 'tree',
            'lambda_val': 0.0,
        })
        self.assertEqual(response.status_code, 400)

    def test_get_rejected(self):
        response = self.client.get(URL_FE)
        self.assertEqual(response.status_code, 405)

    def test_bad_json_returns_400(self):
        response = self.client.post(URL_FE, data='not json', content_type='application/json')
        self.assertEqual(response.status_code, 400)
