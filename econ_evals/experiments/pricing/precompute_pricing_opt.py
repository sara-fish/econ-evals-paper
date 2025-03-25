import pandas as pd
from econ_evals.utils.helper_functions import get_base_dir_path
from ast import literal_eval
from econ_evals.experiments.pricing.pricing_market_logic_multiproduct import (
    get_monopoly_prices_varying_alphas,
)
import os


from tqdm import tqdm

import pickle


if __name__ == "__main__":
    print(
        "Precomputing OPT pricing strategy and saving to dirname_to_monopoly_prices.pkl..."
    )

    big_dirnames = [
        dirname
        for dirname in os.listdir(get_base_dir_path() / "experiments/pricing/logs")
        if dirname.startswith("2025-")
    ]

    base_log_dir = get_base_dir_path() / "experiments" / "pricing" / "logs"

    START_ROUND_IDX = 50
    END_ROUND_IDX = 100

    table = []

    dirname_to_monopoly_prices = {}

    for big_dirname in tqdm(big_dirnames):
        for dirname in os.listdir(base_log_dir / big_dirname):
            log_path = base_log_dir / big_dirname / dirname / "logs.csv"
            global_params = pd.read_csv(
                base_log_dir / big_dirname / dirname / "global_params.csv"
            ).to_dict(orient="records")[0]

            seed = global_params["seed"]

            log_df = pd.read_csv(log_path)

            if len(log_df) == 0:
                print(f"Skipping {log_path}")
                continue

            market_df = log_df.dropna(subset=["prices"])[
                ["prices", "quantities", "profits", "attempt_num"]
            ].reset_index(drop=True)

            market_df["prices"] = market_df["prices"].apply(literal_eval)
            market_df["quantities"] = market_df["quantities"].apply(literal_eval)
            market_df["profits"] = market_df["profits"].apply(literal_eval)

            model = log_df["model"].dropna().values[0]

            num_attempts = global_params["num_attempts"]

            a0 = global_params["a0"]
            a_tuple = literal_eval(global_params["a_tuple"])
            alpha_list = literal_eval(global_params["alpha_list"])
            multiplier_list = literal_eval(global_params["multiplier_list"])
            costs = literal_eval(global_params["costs"])
            c = list(costs.values())
            mu = global_params["mu"]
            sigma = global_params["sigma"]
            group_idxs = literal_eval(global_params["group_idxs"])

            monopoly_prices = []
            monopoly_profits = []

            monopoly_prices = get_monopoly_prices_varying_alphas(
                a0=a0,
                a=a_tuple,
                mu=mu,
                alpha_list=alpha_list,
                c=c,
                multiplier_list=multiplier_list,
                sigma=sigma,
                group_idxs=group_idxs,
            )

            dirname_to_monopoly_prices[dirname] = monopoly_prices

    with open(
        get_base_dir_path() / "experiments/pricing/" / "dirname_to_monopoly_prices.pkl",
        "wb",
    ) as f:
        pickle.dump(dirname_to_monopoly_prices, f)
