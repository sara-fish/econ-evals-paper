from tqdm import tqdm
from econ_evals.experiments.efficiency_vs_equality.run_efficiency_vs_equality_experiment import (
    EfficiencyFairnessArgs,
    run,
)

from econ_evals.utils.llm_tools import ALL_MODELS
import argparse


from itertools import product

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=ALL_MODELS, required=True)
    parser.add_argument(
        "--prompt_type",
        type=str,
        choices=["main", "efficiency", "equality"],
        required=True,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[], required=True)
    args = parser.parse_args()

    model = args.model
    prompt_type = args.prompt_type
    seeds = args.seeds

    # Fixed params
    max_worker_productivity_gap = 18
    worker_productivity_gaps = [18]
    worker_wage = 1
    num_periods = 30
    num_workers = 4

    for seed, worker_productivity_gap in tqdm(
        product(seeds, worker_productivity_gaps),
        desc="experiment",
    ):
        args = EfficiencyFairnessArgs(
            model=model,
            prompt_type=prompt_type,
            seed=seed,
            num_periods=num_periods,
            num_workers=num_workers,
            worker_productivity_gap=worker_productivity_gap,
            max_worker_productivity_gap=max_worker_productivity_gap,
            worker_wage=worker_wage,
        )

        print("Running experiment with args:")
        for key, value in args._asdict().items():
            print(f"{key}: {value}")
        print("\n")

        run(args)
