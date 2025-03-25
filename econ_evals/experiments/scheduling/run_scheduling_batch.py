from econ_evals.experiments.scheduling.run_scheduling_experiment import (
    run,
    SchedulingArgs,
)
from tqdm import tqdm

from econ_evals.utils.helper_functions import get_time_string
from econ_evals.utils.llm_tools import ALL_MODELS

import argparse

PREFERENCE_GENERATION_PARAMS = [
    {"score_gap_worker": 1, "score_gap_task": 1},  # uniform iid
    {
        "score_gap_worker": 3,
        "score_gap_task": 3,
    },  # both have some correlation in prefs
    {
        "score_gap_worker": None,
        "score_gap_task": 3,
    },  # tasks agree on priorities of worker, workers have some correlation in prefs
    {
        "score_gap_worker": None,
        "score_gap_task": 1,
    },  # tasks agree on priorities of worker, workers have iid prefs
]

SCORE_GAP_WORKERS_REPLICATION = sum(
    [
        [config["score_gap_worker"] for _ in range(3)]
        for config in PREFERENCE_GENERATION_PARAMS
    ],
    start=[],
)

SCORE_GAP_TASKS_REPLICATION = sum(
    [
        [config["score_gap_task"] for _ in range(3)]
        for config in PREFERENCE_GENERATION_PARAMS
    ],
    start=[],
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=ALL_MODELS, required=True)
    parser.add_argument(
        "--difficulty", type=str, choices=["Basic", "Medium", "Hard"], required=True
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[], required=True)
    args = parser.parse_args()

    # Global params same for all runs
    model = args.model
    num_attempts = 100  # num periods
    prompt_type = "v1"
    final_prompt_type = "final_attempt_v1"
    blocking_pair_selection_method = "random_cache"
    verbose = False
    difficulty = args.difficulty

    difficulty_to_num_workers = {"Basic": 10, "Medium": 20, "Hard": 50}
    difficulty_to_num_blocking_pairs = {"Basic": 1, "Medium": 2, "Hard": 5}
    seeds = args.seeds

    # Params for preference generation

    score_gap_workers = []
    score_gap_tasks = []

    for seed in seeds:
        # we did it this weird way, but basically seeds 0-11 are balanced in the 4 types of prefs
        score_gap_workers.append(
            SCORE_GAP_WORKERS_REPLICATION[seed % len(SCORE_GAP_WORKERS_REPLICATION)]
        )
        score_gap_tasks.append(
            SCORE_GAP_TASKS_REPLICATION[seed % len(SCORE_GAP_TASKS_REPLICATION)]
        )

    # Now, generate the args for each run
    difficulty_to_args_list = {}
    args_list = []
    for (
        score_gap_worker,
        score_gap_task,
        seed,
    ) in zip(
        score_gap_workers,
        score_gap_tasks,
        seeds,
    ):
        args = SchedulingArgs(
            num_attempts=num_attempts,
            prompt_type=prompt_type,
            final_prompt_type=final_prompt_type,
            num_workers=difficulty_to_num_workers[difficulty],
            num_blocking_pairs=difficulty_to_num_blocking_pairs[difficulty],
            score_gap_worker=score_gap_worker,
            score_gap_task=score_gap_task,
            seed=seed,
            model=model,
            verbose=verbose,
            blocking_pair_selection_method=blocking_pair_selection_method,
        )
        args_list.append(args)

    # Now, run the experiments
    print(f"Staged to run of {num_attempts} periods at {difficulty} {model}")
    log_subdirname = f"{get_time_string()}__{difficulty}__{model}"
    for args in tqdm(args_list):
        run(args, log_subdirname=log_subdirname)
