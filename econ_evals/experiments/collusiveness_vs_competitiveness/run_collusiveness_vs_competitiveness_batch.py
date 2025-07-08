from econ_evals.experiments.collusiveness_vs_competitiveness.run_collusion_litmus_test import (
    run_collusion,
    CollusionArgs,
)

from econ_evals.utils.helper_functions import get_time_string
from tqdm import tqdm

from econ_evals.utils.llm_tools import ALL_MODELS
import argparse

ALPHAS = [1, 3.2, 10]

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, choices=ALL_MODELS, required=True)
    parser.add_argument(
        "--prompt_type",
        type=str,
        choices=[
            "monopoly_v1",
            "collusion_v1",
            "collusion_v1_reasoning",
            "monopoly_v1_reasoning",
        ],
        required=True,
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[], required=True)
    args = parser.parse_args()

    model = args.model
    prompt_type = args.prompt_type
    seeds = args.seeds

    num_periods = 300
    if prompt_type == "monopoly_v1" or prompt_type == "monopoly_v1_reasoning":
        num_agents = 1
    elif prompt_type == "collusion_v1" or prompt_type == "collusion_v1_reasoning":
        num_agents = 2
    else:
        raise NotImplementedError(f"Prompt type {prompt_type} not implemented")

    args_to_run = []
    for seed in seeds:
        alpha = ALPHAS[seed % 3]
        args_to_run.append(
            CollusionArgs(
                seed=seed,
                num_periods=num_periods,
                num_agents=num_agents,
                model=model,
                prompt_type=prompt_type,
                verbose=False,
                alpha=alpha,
                a=(2,) * num_agents,
                c=(1,) * num_agents,
                mu=1 / 4,
                multiplier=100,
                a0=0,
            )
        )

    sub_dirname = f"{get_time_string()}__{prompt_type}__{model}__{num_agents}agent__seeds{'_'.join(map(str,seeds))}"

    for args in tqdm(args_to_run, desc="Run"):
        print("Running pricing experiment with args:")
        for key, value in vars(args).items():
            print(f"{key}: {value}")

        run_collusion(args, sub_dirname=sub_dirname)
