from unittest.mock import patch

from pages.model_eval_dashboard import run_eval


def test_run_eval_calls_gemini_main():
    with patch('pages.model_eval_dashboard.gemini_main') as mock_main:
        run_eval('dataset.csv', 'proj', 'results.csv')
        mock_main.assert_called_once_with('dataset.csv', 'proj', 'results.csv')
