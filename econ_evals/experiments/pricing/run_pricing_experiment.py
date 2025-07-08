import os
from pathlib import Path
import pandas as pd
import numpy as np

from econ_evals.utils.helper_functions import (
    get_base_dir_path,
    get_time_string,
    parse_dict,
)
from econ_evals.utils.llm_tools import call_llm
from econ_evals.experiments.pricing.prompts import get_prompts
from econ_evals.experiments.pricing.generate_instance import (
    PricingArgs,
)

from econ_evals.experiments.pricing.pricing_market_logic_multiproduct import (
    get_quantities,
    get_profits,
    get_monopoly_prices,
)
from tenacity import retry, stop_after_attempt, retry_if_exception_type

from typing import Any, Optional
from tqdm import tqdm

MAX_LLM_QUERIES_PER_PERIOD = 40

MAX_SUBMIT_PRICING_ATTEMPTS = 10


LOG_COLS = [
    "attempt_num",
    "prompt_type",
    "model",
    "system",
    "tools",
    "tool_choice",
    "messages",
    "temperature",
    "max_tokens",
    "response",
    "completion",
    "status",
    "prices",
    "quantities",
    "profits",
    "total profits",
    "costs",
]


class ToolUseException(Exception):
    pass


def make_readable_previous_attempts_data(
    prices_list: list[dict[str, float]],
    quantities_list: list[dict[str, float]],
    profits_list: list[dict[str, float]],
    costs: dict[str, float],
    product_ids: list[str],
) -> str:
    assert len(prices_list) == len(quantities_list) == len(profits_list)

    output = ""
    for i in range(len(prices_list)):
        output += f"Attempt {i}:\n"
        for j in range(len(product_ids)):
            output += f"{product_ids[j]}:\n"
            output += f"Price: {prices_list[i][product_ids[j]]:.2f}\n"
            output += f"Quantity: {quantities_list[i][product_ids[j]]:.2f}\n"
            output += f"Profit: {profits_list[i][product_ids[j]]:.2f}\n"
            output += f"Cost: {costs[product_ids[j]]:.2f}\n"
        output += "\n"
    return output


class PricingAgent:
    def __init__(
        self,
        *,
        id: str,
        model: str,
        prompt_type: str,
        temperature: float,
        log_dirname: Path,
        product_ids: list[str],
        upper_bound_list: list[float],
    ):
        self.id = id
        self.model = model
        self.prompt_type = prompt_type
        self.temperature = temperature
        self.log_dirname = log_dirname
        self.product_ids = product_ids
        self.upper_bound_list = upper_bound_list

        self.notes = [""]
        self.data_accessed_this_period = False

    def use_tool(
        self,
        name: str,
        input: dict,
        prices_list: list[dict[str, float]],
        quantities_list: list[dict[str, float]],
        profits_list: list[dict[str, float]],
        costs: dict[str, float],
    ) -> tuple[str, Any]:
        """
        Use tool

        Return:
        - internal status message (e.g. SUCCESS, PERIOD_OVER, ERROR)
            - SUCCESS: tool use worked, continue
            - PERIOD_OVER: tool use worked & ended period (alloc was produced)
            - ERROR: tool use failed, will need to retry or critique or something
        - output (that LLM will see)
        """

        if name == "get_previous_pricing_data":
            if self.data_accessed_this_period:
                return (
                    "SUCCESS",
                    str(
                        {
                            "previous_pricing_data": "Data already accessed. See previous response. Instead, read/write notes or submit prices."
                        }
                    ),
                )
            else:
                self.data_accessed_this_period = True
                readable_data = make_readable_previous_attempts_data(
                    prices_list=prices_list,
                    quantities_list=quantities_list,
                    profits_list=profits_list,
                    costs=costs,
                    product_ids=self.product_ids,
                )
                return "SUCCESS", str({"previous_pricing_data": readable_data})
        elif name == "get_product_ids":
            return "SUCCESS", str({"product_ids": self.product_ids})
        elif name == "get_attempt_number":
            return "SUCCESS", str({"attempt_number": len(prices_list)})
        elif name == "write_notes":
            assert len(self.notes) == len(prices_list) + 1
            if "notes" in input:
                self.notes[-1] += input["notes"]
                return "SUCCESS", "Successfully wrote notes."
            else:
                return "ERROR", "Malformed input, expected notes argument"
        elif name == "read_notes":
            if "attempt_number" in input:
                attempt_number = int(input["attempt_number"])
                if attempt_number < len(self.notes):
                    return "SUCCESS", str({"notes": self.notes[attempt_number]})
                else:
                    return (
                        "ERROR",
                        f"No notes found for attempt number {attempt_number}",
                    )
            else:
                return "ERROR", "Malformed input, expected attempt_number argument"
        elif name == "set_prices":
            if "prices_dict_str" in input:
                prices = input.get("prices_dict_str")
                if isinstance(prices, float) or isinstance(prices, int):
                    return (
                        "RETRY_ERROR",
                        "Malformed input, prices_dict_str should be a string representation of a dictionary and not a value. For example, your input should be \"{'prices_dict_str' : {'Product_1': 10, 'Product_2' : 20, ...}\"",
                    )
                try:
                    prices = parse_dict(prices)
                except ValueError as _:
                    return (
                        "ERROR",
                        "Malformed input, could not parse prices_dict_str as dictionary",
                    )
                unpriced_products = set(self.product_ids).difference(set(prices.keys()))
                if unpriced_products:
                    return (
                        "RETRY_ERROR",
                        f"Malformed input, missing prices for products: {unpriced_products}",
                    )
                extra_products = set(prices.keys()).difference(set(self.product_ids))
                if extra_products:
                    return (
                        "RETRY_ERROR",
                        f"Malformed input, the following product IDs are unknown: {extra_products}",
                    )
                return "PERIOD_OVER", prices
            else:
                return "ERROR", "Malformed input, expected assignment argument"
        else:
            return "ERROR", f"Invalid tool '{name}'"

    @retry(
        retry=retry_if_exception_type(ToolUseException),
        stop=stop_after_attempt(
            MAX_SUBMIT_PRICING_ATTEMPTS
        ),  # num times to retry attempt before giving up
    )
    def make_pricing_attempt(
        self,
        *,
        prices_list: list[dict[str, float]],
        quantities_list: list[dict[str, float]],
        profits_list: list[dict[str, float]],
        costs: dict[str, float],
        max_queries: int,
        verbose: bool = True,
    ) -> tuple[Optional[dict[str, int]], str]:
        """
        Return:
        - prices (or None if none provided)
        - str (status message if alloc was None)
        """

        self.data_accessed_this_period = False
        system, initial_prompt, tools, tools_action_only, reply_prompt = get_prompts(
            self.prompt_type
        )
        attempt_num = len(prices_list)
        initial_prompt = initial_prompt.format(
            upper_bound_price=self.upper_bound_list[attempt_num]
        )
        messages = [{"role": "user", "content": initial_prompt}]

        for i in range(max_queries):
            with open(self.log_dirname / "info.txt", "a") as f:
                f.write("\n\n=========================\n")
                f.write(f"Attempt {attempt_num}, Query {i}")
                f.write("\n=========================\n")
            if verbose:
                print(f"Attempt {attempt_num}, Query {i}")

            log, response, completion = call_llm(
                model=self.model,
                system=system,
                tools=tools if i != max_queries - 1 else tools_action_only,
                messages=messages,
                tool_choice={"type": "any"},
                temperature=self.temperature,
                caching=True,
            )

            with open(self.log_dirname / "info.txt", "a") as f:
                f.write(f"Response: {response}\n")

            # Save to logs
            pd.DataFrame(
                [{**log, "attempt_num": attempt_num, "prompt_type": self.prompt_type}],
                columns=LOG_COLS,
            ).to_csv(self.log_dirname / "logs.csv", mode="a", header=False, index=False)

            # Append completion to messages
            messages.append({"role": "assistant", "content": completion["content"]})

            # For each tool in completion, compute result and then append to messages
            tool_result_content = []
            for content in completion["content"]:
                if content["type"] == "tool_use":
                    id, input, name = content["id"], content["input"], content["name"]

                    if not isinstance(input, dict):
                        # In rare cases, LLM might write unparseable arguments for tool
                        raise ToolUseException(f"Invalid input {input} for tool {name}")

                    # Do some logging
                    with open(self.log_dirname / "info.txt", "a") as f:
                        f.write(f"Using tool {name} with input {input}\n")
                    if verbose:
                        print(f"Using tool {name} with input {input}")

                    # Use tool
                    status, output = self.use_tool(
                        name=name,
                        input=input,
                        prices_list=prices_list,
                        quantities_list=quantities_list,
                        profits_list=profits_list,
                        costs=costs,
                    )

                    # Do some logging
                    with open(self.log_dirname / "info.txt", "a") as f:
                        f.write(f"Tool {name} status: {status}\n\n")
                        f.write(f"Tool {name} output: {output}\n\n")
                    if verbose:
                        print(f"Tool status: {status}")
                        print(f"Tool output: {output}")

                    # Process the tool
                    if status == "ERROR":
                        # for now, this will just retry the query
                        raise ToolUseException(
                            f"Tool {name} failed with status {status}: {output}"
                        )
                    elif status == "PERIOD_OVER":
                        prices = output
                        break
                    elif status == "RETRY_ERROR":
                        # When a tool use throws an error that we want to give to the LLM
                        # and have it self-correct.
                        tool_result_content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": id,
                                "content": output,
                            }
                        )
                    elif status == "SUCCESS":
                        tool_result_content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": id,
                                "content": output,
                            }
                        )
                    else:
                        raise NotImplementedError(
                            f"Tool status {status} not implemented"
                        )
            else:
                messages.append(
                    {
                        "role": "user",
                        "content": tool_result_content
                        + [{"type": "text", "text": reply_prompt}],
                    }
                )
            if status == "PERIOD_OVER":
                ## Do clean up for next attempt
                assert len(self.notes) == len(prices_list) + 1
                self.notes.append("")
                self.data_accessed_this_period = False
                if verbose:
                    print("Period over.")
                break

        else:
            # LLM did not run the done_setting_prices command in sufficient time, so advancing period manually here.
            if verbose:
                print("LLM did not produce prices in time.")
            raise ToolUseException("LLM did not produce prices in time")

        return prices, ""


def run_pricing_experiment(
    a0: float,
    a_tuple: tuple[float],
    alpha_list: list[list[float]],
    multiplier_list: list[float],
    costs: dict[str, float],
    mu: float,
    sigma: float,
    group_idxs: list[int],
    product_ids: list[str],
    log_dirname: str,
    num_attempts: int,
    prompt_type: str,
    model: str,
    my_random: np.random.RandomState,
    verbose: bool,
):
    # Set upper bound price to be 1.5-2.5 times the max monopoly price
    upper_bound_multiplier = my_random.uniform(1.5, 2.5)
    max_price_per_period = [
        max(
            get_monopoly_prices(
                a0=a0,
                a=a_tuple,
                mu=mu,
                alpha=alpha_list[attempt_num],
                c=tuple(costs.values()),
                multiplier=multiplier_list[attempt_num],
                sigma=sigma,
                group_idxs=group_idxs,
            )
        )
        for attempt_num in range(num_attempts)
    ]

    max_values = []
    # Update upper bound price every 10 attempts
    for i in range(0, num_attempts, 10):
        chunk = max_price_per_period[i : i + 10]
        max_values.append(max(chunk))
    upper_bound_values = [
        round(max_value * upper_bound_multiplier, 2) for max_value in max_values
    ]
    upper_bound_list = [upper_bound_values[i // 10] for i in range(num_attempts)]

    agent = PricingAgent(
        id="0",
        model=model,
        prompt_type=prompt_type,
        temperature=1,
        log_dirname=log_dirname,
        product_ids=product_ids,
        upper_bound_list=upper_bound_list,
    )

    prices_list = []  # List of, for each attempt #, prices given in a dict
    quantities_list = []  # List of, for each attempt #, quantities sold in a dict
    profits_list = []  # List of, for each attempt #, profits earned in a dict

    for attempt_num in tqdm(range(num_attempts)):
        with open(log_dirname / "info.txt", "a") as f:
            f.write("\n\n==================\n")
            f.write(f"Attempt {attempt_num}")
            f.write("\n===================\n")

        prices, status = agent.make_pricing_attempt(
            prices_list=prices_list,
            quantities_list=quantities_list,
            profits_list=profits_list,
            costs=costs,
            max_queries=MAX_LLM_QUERIES_PER_PERIOD,
            verbose=verbose,
        )

        if prices is None:
            with open(log_dirname / "info.txt", "a") as f:
                f.write(f"LLM didn't produce prices. Status: {status}\n")
            pd.DataFrame(
                [
                    {
                        "attempt_num": attempt_num,
                        "prices": None,
                        "status": status,
                    }
                ],
                columns=LOG_COLS,
            ).to_csv(log_dirname / "logs.csv", mode="a", header=False, index=False)
            return status

        else:
            assert prices is not None
            # Compute prices, quantities, and profits without rounding (round at the end)
            price_list = [prices[id] for id in product_ids]
            quantity_list = get_quantities(
                p=price_list,
                a0=a0,
                a=a_tuple,
                mu=mu,
                alpha=alpha_list[attempt_num],
                multiplier=multiplier_list[attempt_num],
                sigma=sigma,
                group_idxs=group_idxs,
            )
            c_list = [costs[id] for id in product_ids]
            profit_list = get_profits(
                p=price_list,
                c=c_list,
                a0=a0,
                a=a_tuple,
                mu=mu,
                alpha=alpha_list[attempt_num],
                multiplier=multiplier_list[attempt_num],
                sigma=sigma,
                group_idxs=group_idxs,
            )
            # Round quantities and profits
            prices_rounded = {id: round(p, 2) for id, p in prices.items()}
            quantity_list_rounded = [round(q, 2) for q in quantity_list]
            profit_list_rounded = [round(p, 2) for p in profit_list]
            prices_list.append(prices_rounded)
            quantities_list.append(
                {id: q for id, q in zip(product_ids, quantity_list_rounded)}
            )
            profits_list.append(
                {id: pro for id, pro in zip(product_ids, profit_list_rounded)}
            )

            with open(log_dirname / "info.txt", "a") as f:
                f.write(f"Pricing plan: {prices_list[-1]}\n")
                f.write(f"Quantities: {quantities_list[-1]}\n")
                f.write(f"Profits: {profits_list[-1]}\n")
                f.write(f"Total Profits: {sum(profits_list[-1].values())}\n")

            if verbose:
                print(f"Pricing plan: {prices_list[-1]}")
                print(f"Quantities: {quantities_list[-1]}")
                print(f"Profits: {profits_list[-1]}")
                print(f"Total Profits: {sum(profits_list[-1].values())}")

            pd.DataFrame(
                [
                    {
                        "attempt_num": attempt_num,
                        "prices": prices_list[-1],
                        "quantities": quantities_list[-1],
                        "profits": profits_list[-1],
                        "total profits": sum(profits_list[-1].values()),
                        "costs": costs,
                        "status": status,
                    }
                ],
                columns=LOG_COLS,
            ).to_csv(log_dirname / "logs.csv", mode="a", header=False, index=False)


def run(args: PricingArgs, difficulty: str, log_subdirname: str = ""):
    params_str = (
        f"{args.model}__{difficulty}__{args.env_type}__{args.num_attempts}attempts"
    )
    log_dirname = (
        get_base_dir_path()
        / "experiments/pricing/logs/"
        / log_subdirname
        / f"{get_time_string()}__{params_str}"
    )
    os.makedirs(log_dirname)
    if not os.path.exists(log_dirname / "logs.csv"):
        pd.DataFrame(columns=LOG_COLS).to_csv(log_dirname / "logs.csv", index=False)

    my_random = np.random.RandomState(args.seed)

    # Save global params to file
    with open(log_dirname / "global_params.csv", "w") as f:
        data = vars(args)
        data["num_attempts"] = args.num_attempts
        data["prompt_type"] = args.prompt_type
        data["seed"] = args.seed
        data["model"] = args.model
        data["verbose"] = args.verbose
        data["noise_param"] = args.noise_param
        data["period_length"] = args.period_length
        data["a0"] = args.a0
        data["a_tuple"] = args.a_tuple
        data["alpha_list"] = args.alpha_list
        data["multiplier_list"] = args.multiplier_list
        data["costs"] = args.costs
        data["group_idxs"] = args.group_idxs
        data["mu"] = args.mu
        data["sigma"] = args.sigma
        data["product_ids"] = args.product_ids
        data["difficulty"] = difficulty
        pd.DataFrame([data]).to_csv(f, index=False)

    run_pricing_experiment(
        a0=args.a0,
        a_tuple=args.a_tuple,
        alpha_list=args.alpha_list,
        multiplier_list=args.multiplier_list,
        costs=args.costs,
        mu=args.mu,
        sigma=args.sigma,
        group_idxs=args.group_idxs,
        product_ids=args.product_ids,
        log_dirname=log_dirname,
        num_attempts=args.num_attempts,
        prompt_type=args.prompt_type,
        model=args.model,
        my_random=my_random,
        verbose=args.verbose,
    )
