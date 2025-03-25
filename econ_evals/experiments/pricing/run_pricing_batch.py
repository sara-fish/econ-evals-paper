from econ_evals.experiments.pricing.run_pricing_experiment import (
    run,
)

from econ_evals.utils.helper_functions import get_time_string
from tqdm import tqdm
import numpy as np

from econ_evals.utils.llm_tools import ALL_MODELS


from econ_evals.experiments.pricing.generate_instance import (
    generate_instance,
)

import argparse

DIFFCULTY_TO_NUM_PRODUCTS = {
    "Basic": 1,
    "Medium": 4,
    "Hard": 10,
}

ENV_TYPES = ["linear_shifts", "periodic_shifts"]


if __name__ == "__main__":
    # args to accept: model name, difficulty, and list of seeds (each integers)

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=ALL_MODELS, required=True)
    parser.add_argument(
        "--difficulty", type=str, choices=["Basic", "Medium", "Hard"], required=True
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[], required=True)
    args = parser.parse_args()

    num_periods = 100
    sigma = 0.5
    mu = 1
    start_multiplier = 100
    group_idx_p = 0.2
    group_idx_cutoff_proportion = 0.25

    ## STUFF TO CHANGE
    model = args.model
    difficulty = args.difficulty
    seeds = args.seeds
    ##

    args_to_run = []

    for seed in seeds:
        env_type = ENV_TYPES[seed % 2]
        my_random = np.random.RandomState(seed)

        num_products = DIFFCULTY_TO_NUM_PRODUCTS[difficulty]
        noise_param = 0

        product_ids = [f"Product_{i}" for i in range(1, num_products + 1)]

        ## Generate params
        starting_alphas = my_random.uniform(1, 10, size=num_products).tolist()
        cost_list = my_random.uniform(1, 2, size=num_products).tolist()
        a_list = my_random.uniform(2, 3, size=num_products).tolist()
        ##

        costs = {product_id: cost for product_id, cost in zip(product_ids, cost_list)}
        a_tuple = tuple(a_list)

        period_length = my_random.randint(10, 20)

        args = generate_instance(
            num_attempts=num_periods,
            seed=seed,
            model=model,
            env_type=env_type,
            num_products=num_products,
            noise_param=noise_param,
            sigma=sigma,
            mu=mu,
            start_multiplier=start_multiplier,
            group_idx_p=group_idx_p,
            group_idx_cutoff_proportion=group_idx_cutoff_proportion,
            product_ids=product_ids,
            costs=costs,
            a_tuple=a_tuple,
            starting_alphas=starting_alphas,
            period_length=period_length,
            my_random=my_random,
        )

        args_to_run.append(args)

    seeds_str = "_".join(map(str, seeds))

    sub_dirname = f"{get_time_string()}__pricing__{model}__{seeds_str}__{difficulty}"

    for args in tqdm(args_to_run, desc="Run"):
        print("Running pricing experiment with args:")
        for key, value in vars(args).items():
            print(f"{key}: {value}")

        run(args, difficulty=difficulty, log_subdirname=sub_dirname)
