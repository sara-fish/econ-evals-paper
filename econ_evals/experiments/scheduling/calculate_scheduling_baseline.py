import pandas as pd
from ast import literal_eval
import numpy as np
from scipy.stats import bootstrap
import os
from econ_evals.utils.helper_functions import get_base_dir_path, get_time_string
from econ_evals.experiments.scheduling.stable_matching_environment import (
    get_blocking_pairs,
)

from tqdm import tqdm


def num_blocking_pairs_in_expectation(
    worker_ids, tasks_ids, worker_prefs, task_prefs, seed=0, num_trials=100
):
    num_blocking_pairs_list = []
    rng = np.random.RandomState(seed)
    for _ in range(num_trials):
        permuted_task_ids = rng.permutation(tasks_ids)
        random_matching = dict(zip(worker_ids, permuted_task_ids))
        num_blocking_pairs_list.append(
            len(get_blocking_pairs(random_matching, worker_prefs, task_prefs))
        )
    res = bootstrap(
        (np.array(num_blocking_pairs_list),),
        np.mean,
        confidence_level=0.95,
        random_state=rng,
    )
    return (
        float(np.mean(num_blocking_pairs_list)),
        float(res.confidence_interval.low),
        float(res.confidence_interval.high),
    )


def run(num_trials: int):
    log_subdirnames = os.listdir(get_base_dir_path() / "experiments/scheduling/logs/")
    log_subdirname_to_dirnames = {
        log_subdirname: os.listdir(
            get_base_dir_path() / "experiments/scheduling/logs/" / log_subdirname
        )
        for log_subdirname in log_subdirnames
        if not log_subdirname.startswith(".")
    }

    difficulty_to_seed_to_baseline_data = {"Basic": {}, "Medium": {}, "Hard": {}}
    for log_subdirname in tqdm(log_subdirnames):
        # Skip hidden directories
        if log_subdirname.startswith("."):
            continue
        dirnames = log_subdirname_to_dirnames[log_subdirname]
        short_log_subdirname = "__".join(log_subdirname.split("__")[1:])
        for dirname in tqdm(dirnames):
            if dirname.startswith("."):
                continue
            global_params = pd.read_csv(
                get_base_dir_path()
                / "experiments/scheduling/logs/"
                / f"{log_subdirname}/{dirname}/global_params.csv"
            ).T.to_dict()[0]

            # Figure out if we've seen this one before
            difficulty = short_log_subdirname.split("__")[0]
            seed = global_params["seed"]

            if seed in difficulty_to_seed_to_baseline_data[difficulty]:
                continue

            # Otherwise, compute the baseline

            # Calculate baseline, which is how well a naive solution would do (matching in order)
            worker_prefs = literal_eval(global_params["worker_prefs"])
            task_prefs = literal_eval(global_params["task_prefs"])
            worker_ids = literal_eval(global_params["worker_ids"])
            task_ids = literal_eval(global_params["task_ids"])
            (
                baseline_num_blocking_pairs,
                confidence_interval_low,
                confident_interval_high,
            ) = num_blocking_pairs_in_expectation(
                worker_ids, task_ids, worker_prefs, task_prefs, num_trials=num_trials
            )

            difficulty_to_seed_to_baseline_data[difficulty][seed] = {
                "baseline_num_blocking_pairs": baseline_num_blocking_pairs,
                "confidence_interval_low": confidence_interval_low,
                "confidence_interval_high": confident_interval_high,
            }

    # Save dict to txt file to be reloaded
    with open(
        get_base_dir_path()
        / f"plots/{get_time_string()}__{num_trials}__scheduling_baseline_data.txt",
        "w",
    ) as f:
        f.write(str(difficulty_to_seed_to_baseline_data))


if __name__ == "__main__":
    num_trials = 10000
    run(num_trials)
