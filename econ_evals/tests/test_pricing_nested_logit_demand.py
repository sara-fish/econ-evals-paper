import unittest
import warnings

from econ_evals.experiments.pricing.generate_instance import generate_instance
from econ_evals.experiments.pricing.run_pricing_batch import (
    DIFFCULTY_TO_NUM_PRODUCTS,
)
from econ_evals.tests.logit_demand_market_logic import (
    get_quantities as get_quantities_logit,
    get_profits as get_profits_logit,
    get_nash_prices as get_nash_prices_logit,
    get_monopoly_prices as get_monopoly_prices_logit,
)

from econ_evals.experiments.pricing.pricing_market_logic_multiproduct import (
    get_monopoly_prices_varying_alphas,
    get_quantities,
    get_profits,
    get_monopoly_prices,
    get_nash_prices,
)

import numpy as np

NUM_REPETITIONS = 10


class TestMonopolyBetterThanNash(unittest.TestCase):
    def test_monopoly_better_than_nash(self):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module="scipy.optimize._differentiable_functions",
            )
            rng = np.random.default_rng(0)

            for _ in range(NUM_REPETITIONS):
                n = rng.integers(2, 10)
                a0 = rng.uniform(-0.1, 0.1)
                a = rng.uniform(2, 3, n)
                alpha = rng.uniform(0.5, 10, n)
                mu = 1 / 4
                multiplier = 100
                sigma = rng.uniform(0, 0.9)  # if sigma is too large, we get overflow

                c = rng.uniform(1, 3, n)

                monopoly_prices = get_monopoly_prices(
                    a0=a0,
                    a=a,
                    mu=mu,
                    alpha=alpha,
                    c=c,
                    multiplier=multiplier,
                    sigma=sigma,
                    group_idxs=(1,) * n,
                )

                nash_prices = get_nash_prices(
                    a0=a0,
                    a=a,
                    mu=mu,
                    alpha=alpha,
                    c=c,
                    multiplier=multiplier,
                    sigma=sigma,
                    group_idxs=(1,) * n,
                )

                monopoly_profits = get_profits(
                    p=monopoly_prices,
                    c=c,
                    a0=a0,
                    a=a,
                    mu=mu,
                    multiplier=multiplier,
                    alpha=alpha,
                    sigma=sigma,
                    group_idxs=(1,) * n,
                )

                nash_profits = get_profits(
                    p=nash_prices,
                    c=c,
                    a0=a0,
                    a=a,
                    mu=mu,
                    multiplier=multiplier,
                    alpha=alpha,
                    sigma=sigma,
                    group_idxs=(1,) * n,
                )

                total_monopoly_profits = sum(monopoly_profits)
                total_nash_profits = sum(nash_profits)

                self.assertGreaterEqual(total_monopoly_profits, total_nash_profits)


class TestSpillover(unittest.TestCase):
    def test_spillover(self):
        # If price of one good increases, then quantities sold of the other goods increase

        rng = np.random.default_rng(0)

        for _ in range(NUM_REPETITIONS):
            n = rng.integers(2, 10)

            p1 = rng.uniform(1, 3, n)
            i = rng.integers(0, n)
            p2 = p1.copy()
            p2[i] = p1[i] + rng.uniform(0.1, 0.5)

            num_groups = rng.integers(1, max(2, n // 3))
            if num_groups == 1:
                group_idxs = (1,) * n
            else:
                group_idxs = rng.choice(num_groups, n)
                while set(group_idxs) != set(range(1, num_groups + 1)):
                    group_idxs = rng.choice(range(1, num_groups + 1), n)

            a0 = rng.uniform(-0.1, 0.1)
            a = rng.uniform(2, 3, n)
            alpha = rng.uniform(0.5, 10, n)
            mu = 1 / 4
            multiplier = 100

            quantities1 = get_quantities(
                p=p1,
                a0=a0,
                a=a,
                mu=mu,
                multiplier=multiplier,
                alpha=alpha,
                sigma=0,
                group_idxs=group_idxs,
            )

            quantities2 = get_quantities(
                p=p2,
                a0=a0,
                a=a,
                mu=mu,
                multiplier=multiplier,
                alpha=alpha,
                sigma=0,
                group_idxs=group_idxs,
            )

            for j in range(n):
                if j == i:
                    self.assertLessEqual(quantities2[j], quantities1[j])
                else:
                    self.assertGreaterEqual(quantities2[j], quantities1[j])


class TestMatchLogitDemand(unittest.TestCase):
    def test_match(self):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module="scipy.optimize._differentiable_functions",
            )

            rng = np.random.default_rng(0)

            a0 = 0
            a = (2, 2)
            c = (1, 1)
            mu = 1 / 4
            multiplier = 100

            for _ in range(NUM_REPETITIONS):
                alpha = rng.uniform(0.5, 10)

                p = rng.uniform(alpha * 1, alpha * 3, 2)

                quantities = get_quantities(
                    p=p,
                    a0=a0,
                    a=a,
                    mu=mu,
                    multiplier=multiplier,
                    alpha=(alpha, alpha),
                    sigma=0,
                    group_idxs=(1, 1),
                )

                quantities_logit = get_quantities_logit(
                    p=p,
                    multiplier=multiplier,
                    alpha=alpha,
                    a0=a0,
                    a=a,
                    mu=mu,
                )

                self.assertAlmostEqual(quantities[0], quantities_logit[0])
                self.assertAlmostEqual(quantities[1], quantities_logit[1])

                profits = get_profits(
                    p=p,
                    c=c,
                    a0=a0,
                    a=a,
                    mu=mu,
                    multiplier=multiplier,
                    alpha=(alpha, alpha),
                    sigma=0,
                    group_idxs=(1, 1),
                )

                profits_logit = get_profits_logit(
                    p=p,
                    c=c,
                    multiplier=multiplier,
                    alpha=alpha,
                    a0=a0,
                    a=a,
                    mu=mu,
                )

                self.assertAlmostEqual(profits[0], profits_logit[0])
                self.assertAlmostEqual(profits[1], profits_logit[1])

                monopoly_prices = get_monopoly_prices(
                    a0=a0,
                    a=a,
                    mu=mu,
                    alpha=(alpha, alpha),
                    c=c,
                    multiplier=multiplier,
                    sigma=0,
                    group_idxs=(1, 1),
                )

                monopoly_prices_logit = get_monopoly_prices_logit(
                    a0=a0, a=a, mu=mu, alpha=alpha, c=c, multiplier=multiplier
                )

                self.assertAlmostEqual(
                    monopoly_prices[0], monopoly_prices_logit[0], places=4
                )
                self.assertAlmostEqual(
                    monopoly_prices[1], monopoly_prices_logit[1], places=4
                )

                nash_prices = get_nash_prices(
                    a0=a0,
                    a=a,
                    mu=mu,
                    alpha=(alpha, alpha),
                    multiplier=multiplier,
                    sigma=0,
                    group_idxs=(1, 1),
                    c=c,
                )

                nash_prices_logit = get_nash_prices_logit(
                    a0=a0, a=a, mu=mu, alpha=alpha, c=c
                )

                self.assertAlmostEqual(nash_prices[0], nash_prices_logit[0], places=4)
                self.assertAlmostEqual(nash_prices[1], nash_prices_logit[1], places=4)


class TestMonopolyBest(unittest.TestCase):
    def test_monopoly_best_full_perturb(self):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module="scipy.optimize._differentiable_functions",
            )
            for seed in range(NUM_REPETITIONS):
                sigma = 0.5
                mu = 1
                start_multiplier = 100
                group_idx_p = 0.2
                group_idx_cutoff_proportion = 0.25
                num_attempts = 100
                my_random = np.random.RandomState(seed)
                model = "gpt-4o-2024-11-20"

                env_type = "linear_shifts"
                difficulty = "Hard"

                num_products = DIFFCULTY_TO_NUM_PRODUCTS[difficulty]
                noise_param = 0  # DIFFCULTY_TO_NOISE_PARAM[difficulty]

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

                # Now, we want to check that if you set some other prices, they never earn more profit than the monopoly prices

                for _ in range(NUM_REPETITIONS):
                    period = my_random.choice(range(len(monopoly_prices)))

                    this_period_monopoly_prices = monopoly_prices[period]
                    this_period_monopoly_profit = sum(
                        get_profits(
                            p=this_period_monopoly_prices,
                            c=[costs[id] for id in product_ids],
                            a0=args.a0,
                            a=a_tuple,
                            mu=mu,
                            multiplier=args.multiplier_list[period],
                            alpha=args.alpha_list[period],
                            sigma=sigma,
                            group_idxs=args.group_idxs,
                        )
                    )

                    for _ in range(NUM_REPETITIONS):  # try other prices
                        p_shift = my_random.uniform(
                            -0.1, 0.1, len(this_period_monopoly_prices)
                        )
                        p = [
                            p1 + p2
                            for p1, p2 in zip(this_period_monopoly_prices, p_shift)
                        ]

                        profit = sum(
                            get_profits(
                                p=p,
                                c=[costs[id] for id in product_ids],
                                a0=args.a0,
                                a=a_tuple,
                                mu=mu,
                                multiplier=args.multiplier_list[period],
                                alpha=args.alpha_list[period],
                                sigma=sigma,
                                group_idxs=args.group_idxs,
                            )
                        )

                        self.assertLessEqual(profit, this_period_monopoly_profit + 1e-4)

    def test_monopoly_best_single_perturb(self):
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                category=UserWarning,
                module="scipy.optimize._differentiable_functions",
            )
            for seed in range(NUM_REPETITIONS):
                sigma = 0.5
                mu = 1
                start_multiplier = 100
                group_idx_p = 0.2
                group_idx_cutoff_proportion = 0.25
                num_attempts = 100
                my_random = np.random.RandomState(seed)
                model = "gpt-4o-2024-11-20"

                env_type = "linear_shifts"
                difficulty = "Hard"

                num_products = DIFFCULTY_TO_NUM_PRODUCTS[difficulty]
                noise_param = 0

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

                # Now, we want to check that if you set some other prices, they never earn more profit than the monopoly prices

                for _ in range(NUM_REPETITIONS):
                    period = my_random.choice(range(len(monopoly_prices)))

                    this_period_monopoly_prices = monopoly_prices[period]
                    this_period_monopoly_profit = sum(
                        get_profits(
                            p=this_period_monopoly_prices,
                            c=[costs[id] for id in product_ids],
                            a0=args.a0,
                            a=a_tuple,
                            mu=mu,
                            multiplier=args.multiplier_list[period],
                            alpha=args.alpha_list[period],
                            sigma=sigma,
                            group_idxs=args.group_idxs,
                        )
                    )

                    for _ in range(NUM_REPETITIONS):  # try other prices
                        random_idx = my_random.choice(
                            range(len(this_period_monopoly_prices))
                        )
                        shift = my_random.uniform(-0.1, 0.1)
                        p = this_period_monopoly_prices.copy()
                        p[random_idx] += shift

                        profit = sum(
                            get_profits(
                                p=p,
                                c=[costs[id] for id in product_ids],
                                a0=args.a0,
                                a=a_tuple,
                                mu=mu,
                                multiplier=args.multiplier_list[period],
                                alpha=args.alpha_list[period],
                                sigma=sigma,
                                group_idxs=args.group_idxs,
                            )
                        )

                        self.assertLessEqual(profit, this_period_monopoly_profit + 1e-4)
