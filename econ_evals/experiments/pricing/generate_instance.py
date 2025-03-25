import numpy as np
import math
from pydantic import BaseModel
from enum import Enum


ENV_TYPE_TO_PROMPT_TYPE = {
    "constant": "v1",
    "linear_shifts": "v1",
    "periodic_shifts": "v1",
}


class EnvType(Enum):
    CONSTANT = "constant"
    LINEAR_SHIFTS = "linear_shifts"
    PERIODIC_SHIFTS = "periodic_shifts"


class PricingArgs(BaseModel):
    num_attempts: int
    prompt_type: str
    seed: int
    model: str
    verbose: bool
    noise_param: float
    period_length: int
    a0: float
    a_tuple: tuple[float, ...]
    alpha_list: list[list[float]]
    multiplier_list: list[float]
    costs: dict[str, float]
    group_idxs: list[int]
    mu: float
    sigma: float
    product_ids: list[str]
    env_type: EnvType
    slopes: list[float]
    amplitudes: list[float]


def generate_group_idxs(
    *,
    num_products: int,
    p: float,
    cutoff_proportion: float,
    my_random: np.random.RandomState,
) -> list[int]:
    """
    This function will generate the group_idxs using a truncated geometric distribution.
    It will sample from a geometric distribution with probability p, but will only accept values
    less than the ceiling of cutoff_proportion * num_products. We will continue to sample until we
    have all the unique values from 1 to the ceiling of cutoff_proportion * num_products.
    """
    group_idxs = []
    while True:
        for _ in range(num_products):
            group_idx = my_random.geometric(p)
            while group_idx > math.ceil(cutoff_proportion * num_products):
                group_idx = my_random.geometric(p)
            group_idxs.append(group_idx)
        if len(set(group_idxs)) == math.ceil(cutoff_proportion * num_products):
            break
        else:
            group_idxs = []
    assert len(group_idxs) == num_products
    assert set(group_idxs) == set(
        range(1, math.ceil(cutoff_proportion * num_products) + 1)
    )
    return group_idxs


def generate_alpha_list(
    *,
    num_attempts: int,
    num_products: int,
    env_type: EnvType,
    period_length: int,
    starting_alphas: list,
    my_random: np.random.RandomState,
) -> tuple[list[list[float]], dict[str, list]]:
    """
    Given an env_type, precompute the alpha values for all num_attempts periods, and all num_products products.

    Return:
    - alpha_list, list of length num_attempts, with the alpha values for each product at each attempt number.
    - dict of extra data (possible keys: slopes if linear_shofts, amplitudes if periodic_shifts, for logging)
    """
    assert len(starting_alphas) == num_products
    amplitudes = []
    slopes = []
    if env_type == EnvType.CONSTANT.value:
        alpha_list = [starting_alphas.copy() for _ in range(num_attempts)]
    elif env_type == EnvType.LINEAR_SHIFTS.value:
        alpha_list = [[] for _ in range(num_attempts)]
        for starting_alpha in starting_alphas:
            slope = my_random.uniform(
                -starting_alpha / (2 * num_attempts),
                starting_alpha / (2 * num_attempts),
            )
            slopes.append(slope)
            for i in range(num_attempts):
                alpha_list[i].append(starting_alpha + slope * i)
    elif env_type == EnvType.PERIODIC_SHIFTS.value:
        alpha_list = [[] for _ in range(num_attempts)]
        for starting_alpha in starting_alphas:
            amplitude = my_random.uniform(starting_alpha / 4, starting_alpha / 2)
            amplitudes.append(amplitude)
            for i in range(num_attempts):
                alpha_list[i].append(
                    starting_alpha
                    + amplitude * math.sin(i * 2 * math.pi / period_length)
                )
    else:
        raise NotImplementedError(f"not recognized env_type={env_type}")
    assert len(alpha_list) == num_attempts
    assert all(len(alpha_list_i) == num_products for alpha_list_i in alpha_list)
    assert not (
        env_type == EnvType.LINEAR_SHIFTS.value and len(slopes) != num_products
    ) or (env_type != EnvType.LINEAR_SHIFTS.value and len(slopes) == num_products)
    assert not (
        env_type == EnvType.PERIODIC_SHIFTS.value and len(amplitudes) != num_products
    ) or (env_type != EnvType.PERIODIC_SHIFTS.value and len(amplitudes) == num_products)
    return alpha_list, {"slopes": slopes, "amplitudes": amplitudes}


def generate_multiplier_list(
    *,
    num_attempts: int,
    noise_param: float,
    start_multiplier: float,
    my_random: np.random.RandomState,
) -> list[float]:
    """
    Generate list of multipliers. This represents the overall volume of demand. Changing this doesn't effect optimal pricing behavior.
    What it does is it gives some noise to the feedback. Each time, we multiply by exp(mean 0 normal noise).
    """
    multiplier_list = [
        start_multiplier + my_random.normal(0, noise_param) for _ in range(num_attempts)
    ]
    assert len(multiplier_list) == num_attempts
    return multiplier_list


def generate_instance(
    num_attempts: int,
    seed: int,
    model: str,
    env_type: EnvType,
    num_products: int,
    noise_param: float,
    sigma: float,
    mu: float,
    start_multiplier: float,
    group_idx_p: float,
    group_idx_cutoff_proportion: float,
    product_ids: list[str],
    costs: dict[str, float],
    a_tuple: tuple,
    starting_alphas: list[float],
    period_length: int,
    my_random: np.random.RandomState,
) -> PricingArgs:
    prompt_type = ENV_TYPE_TO_PROMPT_TYPE[env_type]

    group_idxs = generate_group_idxs(
        num_products=num_products,
        my_random=my_random,
        p=group_idx_p,
        cutoff_proportion=group_idx_cutoff_proportion,
    )

    alpha_list, alpha_supp_data = generate_alpha_list(
        num_attempts=num_attempts,
        num_products=num_products,
        my_random=my_random,
        env_type=env_type,
        starting_alphas=starting_alphas,
        period_length=period_length,
    )

    a0 = 0

    multiplier_list = generate_multiplier_list(
        num_attempts=num_attempts,
        noise_param=noise_param,
        start_multiplier=start_multiplier,
        my_random=my_random,
    )

    args = PricingArgs(
        num_attempts=num_attempts,
        prompt_type=prompt_type,
        seed=seed,
        model=model,
        verbose=False,
        noise_param=noise_param,
        period_length=period_length,
        a0=a0,
        a_tuple=a_tuple,
        alpha_list=alpha_list,
        multiplier_list=multiplier_list,
        costs=costs,
        group_idxs=group_idxs,
        mu=mu,
        sigma=sigma,
        product_ids=product_ids,
        env_type=env_type,
        slopes=alpha_supp_data["slopes"],
        amplitudes=alpha_supp_data["amplitudes"],
    )

    return args
