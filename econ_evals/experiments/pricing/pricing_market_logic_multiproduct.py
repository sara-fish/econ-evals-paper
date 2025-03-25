import math
from math import exp, isclose
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar

import warnings

warnings.filterwarnings(
    "ignore", category=UserWarning, module="scipy.optimize._differentiable_functions"
)


MAX_NUM_STEPS = (
    1000  # max num iterations to do in fixpoint calculation for finding Nash
)

NUM_OPT_REF_POINTS = 20


def get_quantities(
    *,
    p: tuple[float],
    a0: float,
    a: tuple[float],
    mu: float,
    alpha: tuple[float],
    multiplier: float,
    sigma: float,
    group_idxs: tuple[int],
) -> list:
    # Following these notes: https://www.nathanhmiller.org/nlnotes.pdf
    assert len(a) == len(alpha) == len(p) == len(group_idxs)
    assert min(group_idxs) == 1
    assert set(group_idxs) == set(range(1, max(group_idxs) + 1))

    num_groups = max(group_idxs) + 1

    delta0 = a0 / mu
    deltas = [(ai - pi / alphai) / mu for ai, pi, alphai in zip(a, p, alpha)]

    # This is D_g in the notes, where g = group_idx
    group_weights = [exp(delta0) / (1 - sigma)]
    for group_idx in sorted(set(group_idxs)):
        group_i_weight = sum(
            [
                exp(deltai / (1 - sigma))
                for i, deltai in enumerate(deltas)
                if group_idxs[i] == group_idx
            ]
        )
        group_weights.append(group_i_weight)

    assert len(group_weights) == num_groups

    # this is \overline{s_g} in the notes
    group_g_selection_probs = [
        Dg ** (1 - sigma) / sum([Dg ** (1 - sigma) for Dg in group_weights])
        for Dg in group_weights
    ]

    assert isclose(sum(group_g_selection_probs), 1)

    # this is \overline{s_{j|g}} in the notes
    conditional_product_selection_probabilities = [] + [
        exp(delta / (1 - sigma)) / group_weights[group_idxs[i]]
        for i, delta in enumerate(deltas)
    ]

    assert all(
        [
            isclose(
                sum(
                    [
                        prob_j_given_g
                        for j_group, prob_j_given_g in zip(
                            group_idxs, conditional_product_selection_probabilities
                        )
                        if j_group == group_idx
                    ]
                ),
                1,
            )
            for group_idx in range(1, num_groups)
        ]
    )

    quantities = [
        multiplier * prob_j_given_g * group_g_selection_probs[group_idxs[j]]
        for j, prob_j_given_g in enumerate(conditional_product_selection_probabilities)
    ]

    return quantities


def get_profits(
    *,
    p: tuple[float],
    c: tuple[float],
    a0: float,
    a: tuple[float],
    mu: float,
    alpha: tuple[float],
    multiplier: float,
    sigma: float,
    group_idxs: tuple[int],
) -> list:
    q = get_quantities(
        p=p,
        a0=a0,
        a=a,
        mu=mu,
        alpha=alpha,
        multiplier=multiplier,
        sigma=sigma,
        group_idxs=group_idxs,
    )
    assert len(q) == len(p) == len(c) == len(alpha) == len(p)
    profits = [qi * (pi / alphai - ci) for (qi, pi, ci, alphai) in zip(q, p, c, alpha)]
    return profits


def get_monopoly_prices(
    *,
    a0: float,
    a: tuple[float],
    mu: float,
    alpha: tuple[float],
    c: tuple[float],
    multiplier: float,
    sigma: float,
    group_idxs: tuple[int],
) -> list[float]:
    """
    Find a vector of prices that monopolist would set for each of the different goods.
    """

    def inv_profit(
        p, a0=a0, a=a, mu=mu, alpha=alpha, c=c, sigma=sigma, group_idxs=group_idxs
    ):
        return -1 * sum(
            get_profits(
                p=p,
                a0=a0,
                a=a,
                mu=mu,
                alpha=alpha,
                c=c,
                multiplier=multiplier,
                sigma=sigma,
                group_idxs=group_idxs,
            )
        )

    result = minimize(
        inv_profit,
        [ci * alphai for ci, alphai in zip(c, alpha)],
        bounds=[(ci * alphai, None) for ci, alphai in zip(c, alpha)],
        method="trust-constr",
        options={"gtol": 1e-8, "maxiter": 5000, "finite_diff_rel_step": 1e-7},
    )
    optimal_prices = list([float(i) for i in result.x])
    return optimal_prices


def get_monopoly_prices_varying_alphas(
    *,
    a0: float,
    a: tuple[float],
    mu: float,
    alpha_list: list[tuple[float]],
    c: tuple[float],
    multiplier_list: list[float],
    sigma: float,
    group_idxs: tuple[int],
) -> list[list[float]]:
    """
    For each alpha in alpha_list, find the monopoly prices.

    Rather than call get_monopoly_prices for each alpha (this can run into numerical stability issues), we just compute the OPT a couple times and then scale up prices by their corresponding alpha
    """

    assert len(alpha_list) == len(multiplier_list)

    assert all(
        [all([alpha > 0 for alpha in alphas]) for alphas in alpha_list]
    ), "All alphas must be positive"

    my_random = np.random.RandomState(0)

    reference_idxs = my_random.choice(
        range(len(alpha_list)), size=NUM_OPT_REF_POINTS, replace=False
    )

    monopoly_prices_at_reference_idxs = [
        get_monopoly_prices(
            a0=a0,
            a=a,
            mu=mu,
            alpha=alpha_list[idx],
            c=c,
            multiplier=multiplier_list[idx],
            sigma=sigma,
            group_idxs=group_idxs,
        )
        for idx in reference_idxs
    ]
    assert len(monopoly_prices_at_reference_idxs) == NUM_OPT_REF_POINTS

    # For each of these reference monopoly prices, compute all the other monopoly prices by scaling up by alpha
    monopoly_prices_over_time_at_reference_idxs = []

    for idx, reference_idx in enumerate(reference_idxs):
        reference_alpha = alpha_list[reference_idx]
        reference_monopoly_prices = monopoly_prices_at_reference_idxs[idx]
        # list of length (len(alpha_list)), of best guess of monopoly price, scaling up from reference monopoly price
        monopoly_prices_over_time = []  # list len num_periods = len(alpha_list)
        for alpha in alpha_list:  # iterating over periods
            assert len(alpha) == len(reference_alpha)  # this should also be num goods
            monopoly_prices_over_time.append(
                [
                    reference_price_good_i * (alpha_good_i / reference_alpha_good_i)
                    for reference_price_good_i, alpha_good_i, reference_alpha_good_i in zip(
                        reference_monopoly_prices, alpha, reference_alpha
                    )
                ]
            )
            assert len(monopoly_prices_over_time[-1]) == len(a) == len(c)  # num goods
        assert len(monopoly_prices_over_time) == len(alpha_list)
        monopoly_prices_over_time_at_reference_idxs.append(monopoly_prices_over_time)

    assert len(monopoly_prices_at_reference_idxs) == NUM_OPT_REF_POINTS

    # in a perfect world, each of the sublists in monopoly_prices_at_reference_idxs would be the same, but due to numerical instability, they are not
    # so we take the median, and also assert that they're not too far apart (if they are, then something went horribly wrong)

    # Take the median
    monopoly_prices = [
        [
            np.median(
                [
                    monopoly_prices_over_time_at_reference_idxs[reference_idx][period][
                        good
                    ]
                    for reference_idx in range(NUM_OPT_REF_POINTS)
                ]
            )
            for good in range(len(a))
        ]
        for period in range(len(alpha_list))
    ]

    assert len(monopoly_prices) == len(alpha_list)
    assert all(
        [len(monopoly_prices[period]) == len(a) for period in range(len(alpha_list))]
    )

    # in case you are wondering why we don't scale by the multiplier or something -- we could have just set it to 1 and not
    # passed it in, it doesn't matter, this is just to make the interface less confusing (and also idk may as well do the optimization
    # with the rough OOM of that multiplier since we'll be in that regime anyway)

    return monopoly_prices


def get_best_response(
    *,
    p: tuple[float],
    i: int,
    a0: float,
    a: tuple[float],
    mu: float,
    alpha: tuple[float],
    multiplier: float,
    sigma: float,
    group_idxs: tuple[int],
    c: tuple[float],
    lower_bound_pi: float,
    upper_bound_pi: float,
) -> list:
    """
    Let p be a vector of prices the opponents play (where p[i]=None) and i be the vector of the agent we are considering.
    This function computes the best response of agent i, given that the other agents are playing p.
    """

    assert p[i] is None  # all opponent prices are given except pi (which we seek)

    def pi_inv_profit_given_p(pi, p=p, a0=a0, a=a, mu=mu, alpha=alpha, c=c):
        p = p[:i] + (pi,) + p[i + 1 :]  # p[i] = pi
        quantities = get_quantities(
            p=p,
            a0=a0,
            a=a,
            mu=mu,
            alpha=alpha,
            multiplier=multiplier,
            sigma=sigma,
            group_idxs=group_idxs,
        )
        pi_profit = quantities[i] * (pi / alpha[i] - c[i])
        return -1 * pi_profit

    result = minimize_scalar(
        pi_inv_profit_given_p, bounds=(lower_bound_pi, upper_bound_pi)
    )

    optimal_price = result.x

    return list(p[:i] + (optimal_price,) + p[i + 1 :])


def get_nash_prices(
    *,
    a0: float,
    a: tuple[float],
    mu: float,
    alpha: tuple[float],
    multiplier: float,
    sigma: float,
    group_idxs: tuple[int],
    c: tuple[float],
    seed=0,
    eps=1e-8,
) -> list:
    """
    Given market params, returns vector p, which is the Nash equilibrium prices each agent should set in a 1-shot game.
    (really hope it's unique)

    This is computed by finding a fixpoint, which is done by repeatedly applying the best response function.
    """

    rng = random.Random(seed)  # local randomness

    lower_bound_nash = min([ci * alphai for ci, alphai in zip(c, alpha)])
    upper_bound_nash = max(
        get_monopoly_prices(
            a0=a0,
            a=a,
            mu=mu,
            alpha=alpha,
            c=c,
            multiplier=multiplier,
            sigma=sigma,
            group_idxs=group_idxs,
        )
    )

    num_agents = len(c)

    p = tuple(
        [rng.uniform(lower_bound_nash, upper_bound_nash) for _ in range(num_agents)]
    )

    is_not_converged = True
    num_steps = 0

    while is_not_converged and num_steps < MAX_NUM_STEPS:
        best_response_p = tuple(
            [
                get_best_response(
                    p=p[:i] + (None,) + p[i + 1 :],
                    i=i,
                    a0=a0,
                    a=a,
                    mu=mu,
                    alpha=alpha,
                    c=c,
                    multiplier=multiplier,
                    sigma=sigma,
                    group_idxs=group_idxs,
                    lower_bound_pi=lower_bound_nash,
                    upper_bound_pi=upper_bound_nash,
                )[i]
                for i in range(num_agents)
            ]
        )

        is_not_converged = math.dist(p, best_response_p) > eps
        p = best_response_p
        num_steps += 1

    return list([float(i) for i in p])
