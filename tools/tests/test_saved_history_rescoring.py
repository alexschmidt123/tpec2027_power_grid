"""Independent checks of archived-history likelihood and chunked bounds."""
import unittest
import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from scipy.special import logsumexp
from tools.rescore_saved_histories import history_likelihood, information_bounds, score_history, load_completed_results


class LinearCarryObserver:
    def initial_state(self, count):
        return np.zeros((count, 1))

    def propagate(self, theta, state, duration):
        final = state + theta * duration[:, None]
        return SimpleNamespace(observations=final, terminal_state=final)


class TestSavedHistoryScoring(unittest.TestCase):
    def test_carried_history_matches_manual_gaussian_likelihood(self):
        theta = np.array([[0.5], [1.0]])
        values, state = history_likelihood(LinearCarryObserver(), theta, [0.2, 0.4],
                                          np.array([[0.12], [0.29]]), 0.1)
        predictions = np.array([[0.1, 0.3], [0.2, 0.6]])
        expected = -0.5 * np.sum(((np.array([0.12, 0.29])-predictions)/0.1)**2, axis=1)
        np.testing.assert_allclose(values, expected, atol=1e-12)
        np.testing.assert_allclose(state[:, 0], [0.3, 0.6])

    def test_bounds_match_direct_density_ratios(self):
        likelihood = np.array([0.2, 0.3, 0.1, 0.4])
        lower, upper = information_bounds(np.log(likelihood[0]),
                                         logsumexp(np.log(likelihood[1:])), 3)
        self.assertAlmostEqual(lower, np.log(0.2/np.mean(likelihood)))
        self.assertAlmostEqual(upper, np.log(0.2/np.mean(likelihood[1:])))

    def test_scores_remain_finite_when_densities_underflow(self):
        lower, upper = information_bounds(-1000., -2000., 10000)
        self.assertTrue(np.isfinite([lower, upper]).all())
        self.assertLessEqual(lower, np.log(10001))

    def test_chunk_partition_does_not_change_samples_or_scores(self):
        metadata = {'prior_lower':[0.1], 'prior_upper':[1.],
                    'settings':{'noise_sigma':0.1, 'T':2, 'min_duration_separation':0.01}}
        row = {'true_MK':[0.5], 'duration_sequence_s':[0.2,0.4],
               'observations_rocof_hz_s':[[0.12],[0.29]], 'true_terminal_state':[0.3],
               'evaluation_seed':1001, 'system':7, 'method':'dad', 'terminal_spce_nats':0.}
        a = score_history(LinearCarryObserver(), metadata, row, [7, 29], 3)
        b = score_history(LinearCarryObserver(), metadata, row, [7, 29], 29)
        for left, right in zip(a['scores'], b['scores']):
            self.assertAlmostEqual(left['spce_nats'], right['spce_nats'], places=12)
            self.assertAlmostEqual(left['snmc_nats'], right['snmc_nats'], places=12)


    def test_resume_rejects_duplicates_and_wrong_levels(self):
        row = {'T':5, 'method':'dad', 'evaluation_seed':1001, 'system':0,
               'training_seed':101, 'scores':[{'L':1024, 'spce_nats':5., 'snmc_nats':6.}]}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'scores.jsonl'
            path.write_text(json.dumps(row)+'\n')
            self.assertEqual(len(load_completed_results(path, [1024])), 1)
            with self.assertRaises(ValueError):
                load_completed_results(path, [10000])
            path.write_text((json.dumps(row)+'\n')*2)
            with self.assertRaises(ValueError):
                load_completed_results(path, [1024])


if __name__ == '__main__':
    unittest.main()
