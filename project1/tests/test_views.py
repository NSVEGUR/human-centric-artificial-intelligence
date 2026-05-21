from django.test import TestCase
from django.core.files.uploadedfile import SimpleUploadedFile

URL = '/project1/'


def _clf_csv(n_per_class=20):
    """Classification CSV: 2 linearly-separable classes, 4 numeric features."""
    rows = ['f1,f2,f3,f4,label']
    for i in range(n_per_class):
        rows.append(f'{1.0 + i * 0.1:.1f},{2.0 + i * 0.1:.1f},{3.0 + i * 0.1:.1f},{4.0 + i * 0.1:.1f},A')
    for i in range(n_per_class):
        rows.append(f'{5.0 + i * 0.1:.1f},{6.0 + i * 0.1:.1f},{7.0 + i * 0.1:.1f},{8.0 + i * 0.1:.1f},B')
    return '\n'.join(rows).encode()


def _reg_csv(n=30):
    """Regression CSV: 3 numeric features, continuous numeric target."""
    rows = ['x1,x2,x3,target']
    for i in range(n):
        rows.append(f'{1.0 + i:.1f},{2.0 + i:.1f},{3.0 + i:.1f},{10.5 + i * 5.1:.2f}')
    return '\n'.join(rows).encode()


class TestProject1Index(TestCase):
    def test_get_200(self):
        response = self.client.get(URL)
        self.assertEqual(response.status_code, 200)

    def test_get_contains_upload_form(self):
        response = self.client.get(URL)
        self.assertContains(response, 'csv_file')


class TestProject1Upload(TestCase):
    def test_shows_filename_after_upload(self):
        f = SimpleUploadedFile('iris.csv', _clf_csv(), content_type='text/csv')
        response = self.client.post(URL, {'csv_file': f})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['filename'], 'iris.csv')

    def test_detects_classification(self):
        f = SimpleUploadedFile('clf.csv', _clf_csv(), content_type='text/csv')
        response = self.client.post(URL, {'csv_file': f})
        self.assertEqual(response.context['problem_type'], 'classification')
        self.assertGreater(len(response.context['feature_cols']), 0)

    def test_detects_regression(self):
        f = SimpleUploadedFile('reg.csv', _reg_csv(), content_type='text/csv')
        response = self.client.post(URL, {'csv_file': f})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['problem_type'], 'regression')

    def test_generates_plots_on_upload(self):
        f = SimpleUploadedFile('clf.csv', _clf_csv(), content_type='text/csv')
        response = self.client.post(URL, {'csv_file': f})
        self.assertTrue(response.context.get('scatter_plot'))
        self.assertTrue(response.context.get('hist_plot'))
        self.assertTrue(response.context.get('corr_plot'))


class _BaseWithCSV(TestCase):
    """Uploads a classification CSV in setUp so subsequent tests start with session data."""
    def setUp(self):
        f = SimpleUploadedFile('test.csv', _clf_csv(), content_type='text/csv')
        self.client.post(URL, {'csv_file': f})


class TestProject1Table(_BaseWithCSV):
    def test_returns_rows_and_headers(self):
        response = self.client.post(URL, {'action': 'table', 'page_size': '10'})
        self.assertEqual(response.status_code, 200)
        self.assertIn('table_headers', response.context)
        self.assertIn('table_rows', response.context)
        self.assertIn('total_rows', response.context)

    def test_pagination_respects_page_size(self):
        # PAGE_SIZE_OPTS = [10, 25, 50]; invalid values fall back to 10
        response = self.client.post(URL, {'action': 'table', 'page_size': '25', 'table_page': '1'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['page_size'], 25)
        self.assertLessEqual(len(response.context['table_rows']), 25)

    def test_search_filters_rows(self):
        response = self.client.post(URL, {'action': 'table', 'table_search': 'A'})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.context['search_query'], 'A')
        for row in response.context['table_rows']:
            self.assertTrue(any('A' in str(cell) for cell in row))

    def test_session_restore_flag(self):
        # table action with existing session data sets loaded_from_session=True
        response = self.client.post(URL, {'action': 'table'})
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.context.get('loaded_from_session'))


class TestProject1TrainClassification(_BaseWithCSV):
    def test_knn_returns_training_results(self):
        response = self.client.post(URL, {
            'action': 'train',
            'model_key': 'KNN',
            'test_size': '0.2',
            'k': '3',
            'metric': 'euclidean',
        })
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.context.get('error'))
        training = response.context.get('training')
        self.assertIsNotNone(training)
        self.assertIn('Accuracy', training['metrics'])
        self.assertIn('training_plot', training)
        self.assertIn('best_test', training)
        self.assertIn('results_table', training)

    def test_decision_tree_returns_training_results(self):
        response = self.client.post(URL, {
            'action': 'train',
            'model_key': 'DecisionTree',
            'test_size': '0.2',
            'max_depth': '3',
            'min_samples_leaf': '1',
        })
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context.get('training'))

    def test_logistic_regression_returns_training_results(self):
        response = self.client.post(URL, {
            'action': 'train',
            'model_key': 'LogisticRegression',
            'test_size': '0.2',
            'C': '1.0',
            'penalty': 'l2',
        })
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context.get('training'))


class TestProject1TrainRegression(TestCase):
    def setUp(self):
        f = SimpleUploadedFile('reg.csv', _reg_csv(), content_type='text/csv')
        self.client.post(URL, {'csv_file': f})

    def test_ridge_returns_r2_score(self):
        response = self.client.post(URL, {
            'action': 'train',
            'model_key': 'Ridge',
            'test_size': '0.2',
            'alpha': '1.0',
        })
        self.assertEqual(response.status_code, 200)
        training = response.context.get('training')
        self.assertIsNotNone(training)
        self.assertIn('R² Score', training['metrics'])

    def test_linear_regression_returns_training_results(self):
        response = self.client.post(URL, {
            'action': 'train',
            'model_key': 'LinearRegression',
            'test_size': '0.2',
        })
        self.assertEqual(response.status_code, 200)
        self.assertIsNotNone(response.context.get('training'))


class TestProject1Clear(_BaseWithCSV):
    def test_clear_redirects(self):
        response = self.client.post(URL, {'action': 'clear'})
        self.assertRedirects(response, URL)

    def test_clear_empties_session(self):
        self.client.post(URL, {'action': 'clear'})
        response = self.client.get(URL)
        self.assertEqual(response.status_code, 200)
        self.assertIsNone(response.context.get('filename'))
