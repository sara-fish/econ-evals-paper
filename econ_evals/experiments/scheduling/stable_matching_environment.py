from typing import Optional
import numpy as np


def is_valid_matching(
    matching: dict[str, str], worker_ids: list[str], task_ids: list[str]
) -> tuple[bool, str]:
    """
    Given a matching (dict mapping worker to task), check if it is a valid 1-1 bijection between worker_ids and task_ids

    Return:
    - True/False (true if matching is valid, false otherwise)
    - if False, string explaining why invalid (that the LLM will read). If True, empty string
    """
    if not isinstance(matching, dict):
        return False, "Assignment must be a dictionary mapping workers to tasks"
    unmatched_workers = set(worker_ids).difference(set(matching.keys()))
    unmatched_tasks = set(task_ids).difference(set(matching.values()))
    unknown_workers = set(matching.keys()).difference(set(worker_ids))
    unknown_tasks = set(matching.values()).difference(set(task_ids))
    if unmatched_workers:
        return False, "Assignment doesn't include workers: " + str(unmatched_workers)
    if unmatched_tasks:
        return False, "Assignment doesn't include tasks: " + str(unmatched_tasks)
    if unknown_workers:
        return False, "Assignment includes invalid workers: " + str(unknown_workers)
    if unknown_tasks:
        return False, "Assignment includes invalid tasks: " + str(unknown_tasks)
    return True, ""


def get_blocking_pairs(
    matching: dict[str, str],
    worker_prefs: dict[str, list[str]],
    task_prefs: dict[str, list[str]],
) -> list[tuple[str, str]]:
    """
    Given a matching, and prefs of worker and task, return a list of blocking pairs
    """
    # validate input
    assert len(matching) == len(worker_prefs) == len(task_prefs)
    assert set(matching.keys()) == set(worker_prefs.keys())
    assert set(matching.values()) == set(task_prefs.keys())
    # find blocking pairs
    blocking_pairs = []
    for worker in worker_prefs:
        for task in task_prefs:
            # Check if (worker, task) is a blocking pair
            if _is_blocking_pair(worker, task, matching, worker_prefs, task_prefs):
                blocking_pairs.append((worker, task))
    return blocking_pairs


def _is_blocking_pair(
    worker: str,
    task: str,
    matching: dict[str, str],
    worker_prefs: dict[str, list[str]],
    task_prefs: dict[str, list[str]],
) -> bool:
    """
    Given worker and task and matching, return whether (worker, task) is blocking pair
    """
    current_task_for_worker = matching[worker]
    current_worker_for_task = {v: k for k, v in matching.items()}[task]

    # Worker would prefer task to their current match
    worker_prefs_task = worker_prefs[worker].index(task) < worker_prefs[worker].index(
        current_task_for_worker
    )

    # Task would prefer worker to their current match
    task_prefs_worker = task_prefs[task].index(worker) < task_prefs[task].index(
        current_worker_for_task
    )
    return worker_prefs_task and task_prefs_worker


def select_blocking_pair(
    blocking_pairs: list[tuple[str, str]],
    num_blocking_pairs: int,
    worker_prefs: dict[str, list[str]],
    task_prefs: dict[str, list[str]],
    blocking_pair_selection_method: str,
    my_random: np.random.RandomState,
    matching: Optional[dict[str, str]] = None,
    matching_attempts: Optional[list[dict[str, str]]] = None,
    blocking_pair_results: Optional[list[list[tuple[str, str]]]] = None,
) -> tuple[str, str]:
    """
    Given a list of blocking pairs, select the one(s) to give to the agent as feedback

    num_blocking_pairs: Number of blocking pairs to return (might return less if they're fewer blocking pairs)

    Options for blocking_pair_selection_method:
    - random: Select a random blocking pair
    - random_cache: Select a random blocking pair, but if that exact matching was seen before,
        select the same blocking pair as in the history
    """
    assert len(blocking_pairs) >= 1

    if blocking_pair_selection_method == "random_cache":
        assert (
            matching is not None
            and matching_attempts is not None
            and blocking_pair_results is not None
        )

    if blocking_pair_selection_method == "random":
        if num_blocking_pairs >= len(blocking_pairs):
            selected_blocking_pairs = blocking_pairs
        else:
            selected_blocking_pairs_idx = my_random.choice(
                range(len(blocking_pairs)), num_blocking_pairs, replace=False
            )
            selected_blocking_pairs = [
                blocking_pairs[i] for i in selected_blocking_pairs_idx
            ]

    elif blocking_pair_selection_method == "random_cache":
        # Check if current matching has been seen before, and what was returned
        for matching_attempt, blocking_pair_result in zip(
            matching_attempts, blocking_pair_results
        ):
            if matching == matching_attempt:
                selected_blocking_pairs = blocking_pair_result
                break
        else:
            selected_blocking_pairs = select_blocking_pair(
                blocking_pairs=blocking_pairs,
                num_blocking_pairs=num_blocking_pairs,
                worker_prefs=worker_prefs,
                task_prefs=task_prefs,
                blocking_pair_selection_method="random",
                matching=matching,
                matching_attempts=matching_attempts,
                blocking_pair_results=blocking_pair_results,
                my_random=my_random,
            )

    else:
        raise NotImplementedError

    assert 1 <= len(selected_blocking_pairs) <= num_blocking_pairs

    return selected_blocking_pairs
