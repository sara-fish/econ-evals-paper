import unittest

from econ_evals.experiments.scheduling.stable_matching_environment import (
    get_blocking_pairs,
    select_blocking_pair,
)
from econ_evals.experiments.scheduling.generate_preferences import (
    generate_preferences,
)
import numpy as np


class TestPreferenceGeneration(unittest.TestCase):
    def test_generate_preferences1(self):
        # Both sides have fixed priorities over the other
        worker_ids = ["W1", "W2", "W3", "W4", "W5"]
        task_ids = ["T1", "T2", "T3", "T4", "T5"]

        my_random = np.random.RandomState(0)

        worker_prefs, task_prefs = generate_preferences(
            worker_ids=worker_ids,
            task_ids=task_ids,
            score_gap_worker=None,
            score_gap_task=None,
            my_random=my_random,
        )

        worker_prefs_l, task_prefs_l = (
            list(worker_prefs.values()),
            list(task_prefs.values()),
        )

        # Workers and tasks have same prefs
        self.assertTrue(
            all(worker_pref == worker_prefs_l[0] for worker_pref in worker_prefs_l)
        )
        self.assertTrue(all(task_pref == task_prefs_l[0] for task_pref in task_prefs_l))

        # If we rerun with same randomness, we get same prefs
        my_random = np.random.RandomState(0)
        worker_prefs2, task_prefs2 = generate_preferences(
            worker_ids=worker_ids,
            task_ids=task_ids,
            score_gap_worker=None,
            score_gap_task=None,
            my_random=my_random,
        )

        self.assertEqual(worker_prefs, worker_prefs2)
        self.assertEqual(task_prefs, task_prefs2)

    def test_generate_preferences2(self):
        # Tasks have fixed priorities over workers, workers have random prefs
        worker_ids = ["W1", "W2", "W3", "W4", "W5"]
        task_ids = ["T1", "T2", "T3", "T4", "T5"]

        my_random = np.random.RandomState(0)

        worker_prefs, task_prefs = generate_preferences(
            worker_ids=worker_ids,
            task_ids=task_ids,
            score_gap_worker=None,
            score_gap_task=1,
            my_random=my_random,
        )

        worker_prefs_l, task_prefs_l = (
            list(worker_prefs.values()),
            list(task_prefs.values()),
        )

        # Tasks have same prefs
        self.assertTrue(all(task_pref == task_prefs_l[0] for task_pref in task_prefs_l))
        # Workers don't
        self.assertFalse(
            all(worker_pref == worker_prefs_l[0] for worker_pref in worker_prefs_l)
        )

        # If we rerun with same randomness, we get same prefs

        my_random = np.random.RandomState(0)

        worker_prefs2, task_prefs2 = generate_preferences(
            worker_ids=worker_ids,
            task_ids=task_ids,
            score_gap_worker=None,
            score_gap_task=1,
            my_random=my_random,
        )

        self.assertEqual(worker_prefs, worker_prefs2)
        self.assertEqual(task_prefs, task_prefs2)

    def test_generate_preferences3(self):
        # Both sides have fixed priorities over the other
        worker_ids = ["W1", "W2", "W3", "W4", "W5"]
        task_ids = ["T1", "T2", "T3", "T4", "T5"]

        my_random = np.random.RandomState(0)

        worker_prefs, task_prefs = generate_preferences(
            worker_ids=worker_ids,
            task_ids=task_ids,
            score_gap_worker=1e9,
            score_gap_task=1e9,
            my_random=my_random,
        )

        worker_prefs_l, task_prefs_l = (
            list(worker_prefs.values()),
            list(task_prefs.values()),
        )

        # Workers and tasks have same prefs (public score is high enough )
        self.assertTrue(
            all(worker_pref == worker_prefs_l[0] for worker_pref in worker_prefs_l)
        )
        self.assertTrue(all(task_pref == task_prefs_l[0] for task_pref in task_prefs_l))

        # If we rerun with same randomness, we get same prefs
        my_random = np.random.RandomState(0)
        worker_prefs2, task_prefs2 = generate_preferences(
            worker_ids=worker_ids,
            task_ids=task_ids,
            score_gap_worker=1e9,
            score_gap_task=1e9,
            my_random=my_random,
        )

        self.assertEqual(worker_prefs, worker_prefs2)
        self.assertEqual(task_prefs, task_prefs2)


class TestMatching(unittest.TestCase):
    def test_get_blocking_pairs1(self):
        matching = {"A": "1", "B": "2", "C": "3"}
        worker_prefs = {
            "A": ["1", "2", "3"],
            "B": ["2", "1", "3"],
            "C": ["3", "2", "1"],
        }
        # all workers gets their first choice, no blocking pairs
        task_prefs = {"1": ["A", "B", "C"], "2": ["A", "B", "C"], "3": ["A", "B", "C"]}
        blocking_pairs = get_blocking_pairs(matching, worker_prefs, task_prefs)
        self.assertEqual(blocking_pairs, [])
        with self.assertRaises(AssertionError):
            select_blocking_pair(
                blocking_pairs=blocking_pairs,
                num_blocking_pairs=1,
                worker_prefs=worker_prefs,
                task_prefs=task_prefs,
                blocking_pair_selection_method="random",
                my_random=np.random.RandomState(0),
            )

    def test_get_blocking_pairs2(self):
        matching = {"A": "2", "B": "3", "C": "1"}
        worker_prefs = {
            "A": ["1", "2", "3"],
            "B": ["2", "3", "1"],
            "C": ["3", "1", "2"],
        }
        # A is favored worker, so they should get 1, not C
        task_prefs = {"1": ["A", "B", "C"], "2": ["A", "B", "C"], "3": ["A", "B", "C"]}
        blocking_pairs = get_blocking_pairs(matching, worker_prefs, task_prefs)
        self.assertEqual(blocking_pairs, [("A", "1")])
        self.assertEqual(
            select_blocking_pair(
                blocking_pairs=blocking_pairs,
                num_blocking_pairs=1,
                worker_prefs=worker_prefs,
                task_prefs=task_prefs,
                blocking_pair_selection_method="random",
                my_random=np.random.RandomState(0),
            ),
            [("A", "1")],
        )

    def test_get_blocking_pairs3(self):
        matching = {"A": "1", "B": "2", "C": "3", "D": "4"}
        worker_prefs = {
            "A": ["4", "3", "2", "1"],
            "B": ["4", "3", "2", "1"],
            "C": ["1", "2", "4", "3"],  # note! swap 4,3
            "D": ["1", "2", "3", "4"],
        }
        task_prefs = {
            "1": ["A", "B", "C", "D"],
            "2": ["A", "B", "C", "D"],
            "3": ["A", "B", "C", "D"],
            "4": ["A", "B", "C", "D"],
        }
        blocking_pairs = get_blocking_pairs(
            matching=matching, worker_prefs=worker_prefs, task_prefs=task_prefs
        )
        self.assertEqual(
            set(blocking_pairs),
            {("A", "4"), ("A", "3"), ("A", "2"), ("B", "4"), ("B", "3"), ("C", "4")},
        )

    def test_get_blocking_pairs_random_cache1(self):
        my_random = np.random.RandomState(0)

        matching = {"A": "1", "B": "2", "C": "3", "D": "4"}
        worker_prefs = {
            "A": ["4", "3", "2", "1"],
            "B": ["4", "3", "2", "1"],
            "C": ["1", "2", "4", "3"],  # note! swap 4,3
            "D": ["1", "2", "3", "4"],
        }
        task_prefs = {
            "1": ["A", "B", "C", "D"],
            "2": ["A", "B", "C", "D"],
            "3": ["A", "B", "C", "D"],
            "4": ["A", "B", "C", "D"],
        }
        blocking_pairs = get_blocking_pairs(
            matching=matching, worker_prefs=worker_prefs, task_prefs=task_prefs
        )

        for num_blocking_pairs in [1, 5, 10]:
            blocking_pair1 = select_blocking_pair(
                blocking_pairs=blocking_pairs,
                num_blocking_pairs=num_blocking_pairs,
                worker_prefs=worker_prefs,
                task_prefs=task_prefs,
                blocking_pair_selection_method="random_cache",
                my_random=my_random,
                matching=matching,
                matching_attempts=[],
                blocking_pair_results=[],
            )

            blocking_pair2 = select_blocking_pair(
                blocking_pairs=blocking_pairs,
                num_blocking_pairs=num_blocking_pairs,
                worker_prefs=worker_prefs,
                task_prefs=task_prefs,
                blocking_pair_selection_method="random_cache",
                my_random=my_random,
                matching=matching,
                matching_attempts=[matching],
                blocking_pair_results=[blocking_pair1],
            )

            self.assertEqual(blocking_pair1, blocking_pair2)

            # But, if we use random, they might be different

            blocking_pairs_not_cached = []
            for _ in range(10):
                blocking_pair = select_blocking_pair(
                    blocking_pairs=blocking_pairs,
                    num_blocking_pairs=num_blocking_pairs,
                    worker_prefs=worker_prefs,
                    task_prefs=task_prefs,
                    blocking_pair_selection_method="random",
                    my_random=my_random,
                    matching_attempts=[matching],
                    blocking_pair_results=[blocking_pair1],
                )
                blocking_pairs_not_cached.append(blocking_pair)

            if num_blocking_pairs < len(blocking_pairs):
                self.assertTrue(
                    not all(
                        blocking_pair == blocking_pair2
                        for blocking_pair in blocking_pairs_not_cached
                    )
                )
