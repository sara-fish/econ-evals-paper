from math import exp
import math
import random
from scipy.optimize import minimize_scalar, minimize


# This is the logic used for logit demand in https://arxiv.org/pdf/2404.00806
# This is not used anywhere in our experiments, but it is used in the unit tests for nested logit demand,
# to make sure it reduces to this

DEFAULT_a0 = 0
DEFAULT_a = (2, 2)
DEFAULT_c = (1, 1)
DEFAULT_mu = 1 / 4
DEFAULT_alpha = 1
DEFAULT_MULTIPLIER = 10000

MAX_NUM_STEPS = (
    1000  # max num iterations to do in fixpoint calculation for finding Nash
)


def get_quantities(
    *,
    p,
    a0=DEFAULT_a0,
    a=DEFAULT_a,
    mu=DEFAULT_mu,
    alpha=DEFAULT_alpha,
    multiplier=DEFAULT_MULTIPLIER,
) -> list:
    numerators = [exp((ai - pi / alpha) / mu) for (ai, pi) in zip(a, p)]
    denom = exp(a0 / mu) + sum(numerators)
    numerators = [n * multiplier for n in numerators]
    return [n / denom for n in numerators]


def get_profits(
    *,
    p,
    a0=DEFAULT_a0,
    a=DEFAULT_a,
    mu=DEFAULT_mu,
    alpha=DEFAULT_alpha,
    c=DEFAULT_c,
    multiplier=DEFAULT_MULTIPLIER,
) -> list:
    q = get_quantities(p=p, a0=a0, a=a, mu=mu, alpha=alpha, multiplier=multiplier)
    profits = [qi * (pi / alpha - ci) for (qi, pi, ci) in zip(q, p, c)]
    return profits


def get_monopoly_prices(
    *,
    a0=DEFAULT_a0,
    a=DEFAULT_a,
    mu=DEFAULT_mu,
    alpha=DEFAULT_alpha,
    c=DEFAULT_c,
    multiplier=DEFAULT_MULTIPLIER,
) -> list:
    """
    Find a vector of prices that monopolist would set for each of the different goods.
    """

    def inv_profit(p, a0=a0, a=a, mu=mu, alpha=alpha, c=c):
        return -1 * sum(
            get_profits(p=p, a0=a0, a=a, mu=mu, alpha=alpha, c=c, multiplier=multiplier)
        )

    result = minimize(inv_profit, c)
    optimal_prices = list([float(i) for i in result.x])
    return optimal_prices


def get_best_response(
    *,
    p,
    i,
    nash_lower_bound,
    nash_upper_bound,
    a0=DEFAULT_a0,
    a=DEFAULT_a,
    mu=DEFAULT_mu,
    alpha=DEFAULT_alpha,
    c=DEFAULT_c,
) -> list:
    """
    Let p be a vector of prices the opponents play (where p[i]=None) and i be the vector of the agent we are considering.
    This function computes the best response of agent i, given that the other agents are playing p.
    """

    assert p[i] is None  # all opponent prices are given except pi (which we seek)

    def pi_inv_profit_given_p(pi, p=p, a0=a0, a=a, mu=mu, alpha=alpha, c=c):
        p = p[:i] + (pi,) + p[i + 1 :]  # p[i] = pi
        numerators = [exp((ai - pi / alpha) / mu) for ai, pi in zip(a, p)]
        denominator = sum(numerators) + exp(a0 / mu)
        quantities = [numerator / denominator for numerator in numerators]
        pi_profit = quantities[i] * (pi - alpha * c[i])
        return -1 * pi_profit

    result = minimize_scalar(
        pi_inv_profit_given_p, bounds=(nash_lower_bound, nash_upper_bound)
    )

    optimal_price = result.x

    return list(p[:i] + (optimal_price,) + p[i + 1 :])


def get_nash_prices(
    *,
    a0=DEFAULT_a0,
    a=DEFAULT_a,
    mu=DEFAULT_mu,
    alpha=DEFAULT_alpha,
    c=DEFAULT_c,
    seed=0,
    eps=1e-8,
) -> list:
    """
    Given market params, returns vector p, which is the Nash equilibrium prices each agent should set in a 1-shot game.
    (really hope it's unique)

    This is computed by finding a fixpoint, which is done by repeatedly applying the best response function.
    """

    rng = random.Random(seed)  # local randomness

    nash_lower_bound = min(c) * alpha
    nash_upper_bound = max(get_monopoly_prices(a0=a0, a=a, mu=mu, alpha=alpha, c=c))

    num_agents = len(c)

    p = tuple(
        [rng.uniform(nash_lower_bound, nash_upper_bound) for _ in range(num_agents)]
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
                    nash_lower_bound=0,
                    nash_upper_bound=nash_upper_bound,
                )[i]
                for i in range(num_agents)
            ]
        )

        is_not_converged = math.dist(p, best_response_p) > eps
        p = best_response_p
        num_steps += 1

    return list([float(i) for i in p])
