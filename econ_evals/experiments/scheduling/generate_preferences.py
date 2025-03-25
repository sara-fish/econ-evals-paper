from typing import Optional
import numpy as np
import math


def _generate_agent_prefs(
    score_gap: Optional[float],
    scored_agent_ids: list[str],
    scoring_agent_ids: list[str],
    my_random: np.random.RandomState,
) -> dict[str, list[str]]:
    """
    Generate preferences of scoring agents over scored agents, where the scores
    are generated based on the score_gap parameter.
    """

    if score_gap is None:
        # score_gap = infty => all agents agree on how to rank each other

        scored_agent_ids_permuted = list(scored_agent_ids).copy()
        my_random.shuffle(scored_agent_ids_permuted)

        prefs = {
            scoring_agent_id: scored_agent_ids_permuted.copy()
            for scoring_agent_id in scoring_agent_ids
        }
    else:
        # score_gap = bounded => generate public scores and sample

        ## Generate scores ##
        n = len(scored_agent_ids)
        scores = [1 * (score_gap / 1) ** (i / (n - 1)) for i in range(n)]
        assert math.isclose(max(scores), score_gap)
        my_random.shuffle(scores)

        ## Generate prefs of scoring agents over scored agents ##

        scoring_agent_latents = {
            scoring_agent_id: my_random.exponential(scale=scores)
            for scoring_agent_id in scoring_agent_ids
        }
        # this is a trick from probability to avoid resampling, see https://arxiv.org/abs/2009.05124
        # Map each scoring agent to samples from Exp(task score) for each task score.

        prefs = {
            scoring_agent_id: sorted(
                scored_agent_ids,
                key=lambda scored_agent_id: scoring_agent_latents[scoring_agent_id][
                    scored_agent_ids.index(scored_agent_id)
                ],
                reverse=True,
            )
            for scoring_agent_id in scoring_agent_ids
        }

    return prefs


def generate_preferences(
    worker_ids: list[str],
    task_ids: list[str],
    my_random: np.random.RandomState,
    score_gap_worker: Optional[float] = None,
    score_gap_task: Optional[float] = None,
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    """
    Generate preferences for workers and tasks.

    Args:
    worker_ids: list of worker ids
    task_ids: list of task ids
    my_random: np.random.RandomState
    score_gap_worker: float >= 1 or None.
        If float >= 1, generate public scores for workers, evenly spaced between 1 and score_gap_worker
        Tasks's prefs over workers generated randomly proportionally to scores (favorite worker is selected w.p. proportional to score, and so on)
        (So 1 = tasks have uniform random prefs over workers)
        If None, then generate some fixed ranking of workers, that all tasks have in common (equiv to score_gap_worker = infty)
    score_gap_task: float >= 1 or None.
        Analagous to score_gap_worker.

    For more on the public scores model, see e.g. https://arxiv.org/abs/2009.05124

    Returns: worker_prefs, task_prefs
    """

    assert score_gap_worker is None or score_gap_worker >= 1
    assert score_gap_task is None or score_gap_task >= 1
    assert len(worker_ids) == len(task_ids)

    worker_prefs = _generate_agent_prefs(
        score_gap=score_gap_task,
        scored_agent_ids=task_ids,
        scoring_agent_ids=worker_ids,
        my_random=my_random,
    )

    task_prefs = _generate_agent_prefs(
        score_gap=score_gap_worker,
        scored_agent_ids=worker_ids,
        scoring_agent_ids=task_ids,
        my_random=my_random,
    )

    return worker_prefs, task_prefs
