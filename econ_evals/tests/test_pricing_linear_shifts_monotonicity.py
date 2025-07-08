import time
import unittest

from econ_evals.experiments.pricing.pricing_market_logic_multiproduct import (
    get_monopoly_prices_varying_alphas,
)

from econ_evals.experiments.pricing.generate_instance import generate_instance

from econ_evals.experiments.pricing.run_pricing_batch import (
    DIFFCULTY_TO_NUM_PRODUCTS,
)

import warnings

import matplotlib.pyplot as plt

from econ_evals.utils.helper_functions import get_base_dir_path


import numpy as np

NUM_REPETITIONS = 5


def is_monotone(l: list[float]) -> bool:  # noqa: E741
    """
    Given a list of numbers, return True if it's monotone (either all increasing or all decreasing), False otherwise
    """
    return all(l[i] <= l[i + 1] for i in range(len(l) - 1)) or all(
        l[i] >= l[i + 1] for i in range(len(l) - 1)
    )


def count_monotone_failures(l: list[float], eps=1e-4) -> bool:  # noqa: E741
    increase_count = 0
    decrease_count = 0
    for prev, next in zip(l[:-1], l[1:]):
        if prev + eps <= next:
            increase_count += 1
        elif prev - eps >= next:
            decrease_count += 1
    return min(increase_count, decrease_count)


class TestPricingLinearShiftsMonotonicity(unittest.TestCase):
    def test_monotonicity(self):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module="scipy.optimize._differentiable_functions",
            )
            sigma = 0.5
            mu = 1
            start_multiplier = 100
            group_idx_p = 0.2
            group_idx_cutoff_proportion = 0.25
            num_attempts = 100
            model = ""

            env_type = "linear_shifts"
            difficulty = "Hard"

            num_products = DIFFCULTY_TO_NUM_PRODUCTS[difficulty]
            noise_param = 0

            for seed in range(NUM_REPETITIONS):
                my_random = np.random.RandomState(seed)

                product_ids = [f"Product_{i}" for i in range(1, num_products + 1)]

                starting_alphas = my_random.uniform(1, 10, size=num_products).tolist()
                cost_list = my_random.uniform(1, 10, size=num_products).tolist()
                costs = {
                    product_id: cost for product_id, cost in zip(product_ids, cost_list)
                }

                a_list = my_random.uniform(1, 10, size=num_products).tolist()
                a_tuple = tuple(a_list)

                period_length = my_random.randint(10, 20)

                args = generate_instance(
                    num_attempts=num_attempts,
                    prompt_type="v1",
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

                monopoly_prices = get_monopoly_prices_varying_alphas(
                    a0=args.a0,
                    a=a_tuple,
                    mu=mu,
                    alpha_list=args.alpha_list,
                    c=[costs[id] for id in product_ids],
                    multiplier_list=args.multiplier_list,
                    sigma=sigma,
                    group_idxs=args.group_idxs,
                )

                for good_idx, _ in enumerate(product_ids):
                    good_prices = [
                        period_i_prices[good_idx] for period_i_prices in monopoly_prices
                    ]
                    monotonicity_failures = count_monotone_failures(good_prices, eps=0)
                    if monotonicity_failures > 0:
                        print("Test failed. Info:")
                        print(f"seed={seed}")
                        print(f"costs={costs}")
                        print(f"starting_alphas={starting_alphas}")
                        print(f"good_idx={good_idx}")
                        print(f"good_prices={good_prices}")
                        print(f"monotonicity_failures={monotonicity_failures}")
                        print()
                        plt.plot(good_prices)
                        plt.title(f"seed={seed}\ngood_idx={good_idx}")
                        plt.tight_layout()
                        plt.savefig(
                            get_base_dir_path()
                            / "tests"
                            / f"temp/prices_{int(time.time())}.png"
                        )
                        plt.clf()


if __name__ == "__main__":
    unittest.main()
