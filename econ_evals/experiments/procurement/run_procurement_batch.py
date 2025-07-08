from tqdm import tqdm

from econ_evals.utils.helper_functions import get_time_string

from econ_evals.experiments.procurement.run_procurement_experiment import (
    ProcurementArgs,
    run,
)

from econ_evals.utils.llm_tools import ALL_MODELS

import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=ALL_MODELS, required=True)
    parser.add_argument(
        "--prompt_type",
        type=str,
        default="v1",
        choices=["v1", "v1_o3", "v1_known_horizon", "v1_graded_best"],
    )
    parser.add_argument(
        "--difficulty", type=str, choices=["Basic", "Medium", "Hard"], required=True
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[], required=True)
    args = parser.parse_args()

    model = args.model
    prompt_type = args.prompt_type
    difficulty = args.difficulty
    seeds = args.seeds

    num_attempts = 100
    agg_type = "prod"  # min is implemented but we never use it
    verbose = False

    global_params = {
        "model": model,
        "num_attempts": num_attempts,
        "prompt_type": prompt_type,
        "agg_type": agg_type,
        "verbose": verbose,
    }

    basic_params = {
        "num_inputs": 3,
        "num_alternatives_per_input": 4,
        "num_entries": 12,
        "NUM_ITEMS_PER_ENTRY_P": 0.8,
        "QUANTITY_PER_ITEM_P": 0.5,
        "OFFER_QTY_IN_SAMPLE_BUNDLE_P": 0.5,
        "MIN_EFFECTIVENESS": 1,
        "MAX_EFFECTIVENESS": 3,
    }

    medium_params = {
        "num_inputs": 5,
        "num_alternatives_per_input": 6,
        "num_entries": 30,
        "NUM_ITEMS_PER_ENTRY_P": 0.5,
        "QUANTITY_PER_ITEM_P": 0.2,
        "OFFER_QTY_IN_SAMPLE_BUNDLE_P": 0.2,
        "MIN_EFFECTIVENESS": 1,
        "MAX_EFFECTIVENESS": 5,
    }

    # If using Gurobi free version, highest you can do is about num_inputs=5, num_alternatives=8
    # To run Hard, need academic license
    hard_params = {
        "num_inputs": 10,
        "num_alternatives_per_input": 10,
        "num_entries": 100,
        "NUM_ITEMS_PER_ENTRY_P": 0.1,
        "QUANTITY_PER_ITEM_P": 0.1,
        "OFFER_QTY_IN_SAMPLE_BUNDLE_P": 0.1,
        "MIN_EFFECTIVENESS": 1,
        "MAX_EFFECTIVENESS": 20,
    }

    difficulty_to_params = {
        "Basic": basic_params,
        "Medium": medium_params,
        "Hard": hard_params,
    }

    seeds_str = "_".join(map(str, seeds))

    print(
        f"Staged to run {num_attempts} attempts at difficulty {difficulty} using {model}, seeds ={seeds_str}"
    )
    log_subdirname = f"{get_time_string()}__{difficulty}__{model}__{seeds_str}"
    difficulty_params = difficulty_to_params[difficulty]
    for seed in tqdm(seeds):
        args = ProcurementArgs(
            **global_params,
            **difficulty_params,
            seed=seed,
        )
        run(
            args,
            log_subdirname=log_subdirname,
        )
