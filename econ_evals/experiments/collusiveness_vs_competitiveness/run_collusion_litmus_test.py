import os

from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt
from econ_evals.experiments.pricing.pricing_market_logic_multiproduct import (
    get_monopoly_prices,
    get_nash_prices,
    get_profits,
    get_quantities,
)
import pandas as pd

from econ_evals.utils.helper_functions import (
    get_base_dir_path,
    get_time_string,
    parse_dict,
)
from econ_evals.utils.llm_tools import ALL_MODELS, call_llm

from econ_evals.experiments.collusiveness_vs_competitiveness.prompts import get_prompts

from tqdm import tqdm

MAX_LLM_QUERIES_PER_PERIOD = 40
NUM_PRICING_RETRIES = 10


GLOBAL_PARAM_COLS = [
    "seed",
    "prompt_type",
    "a",
    "a0",
    "c",
    "mu",
    "multiplier",
    "monopoly_price",
    "nash_price",
    "alpha",
    "sigma",
    "group_idxs",
    "full_alpha",
]

LOG_COLS = [
    "period",  # first for readability
    # LLM log stuff
    "model",
    "system",
    "tools",
    "tool_choice",
    "messages",
    "temperature",
    "max_tokens",
    "response",
    "completion",
    # Pricing stuff that changes
    "prices",
    "quantities",
    "profits",
]


class ToolUseException(Exception):
    pass


def period_list_to_legible(data_lists: list[list], item_names: list[str]) -> str:
    """
    Given pricing (& quantity, profit) data, convert into a more readable format.

    Example:
    data_lists = [[10, 20], [15, 25], [20, 30]]
    item_names = ["Price", "Quantity"]
    Output:
    Period 0:
        Price: 10
        Quantity: 15
    Period 1:
        Price: 20
    [...]
    """
    assert all(
        len(l) == len(data_lists[0])
        for l in data_lists  # noqa: E741
    ), "All lists must have the same length."
    assert len(item_names) == len(
        data_lists
    ), "Number of item names must match number of period lists."
    output = ""
    for period in range(len(data_lists[0])):  # 0-indexed
        output += f"Attempt {period}:\n"
        for item_name, data_list in zip(item_names, data_lists):
            output += f"{item_name}: {data_list[period]}\n"
        output += "\n"
    return output


class PricingAgent:
    def __init__(
        self,
        *,
        id: int,
        model: str,
        prompt_type: str,
        log_dirname: str,
        upper_bound_price: float,
        notes: list[str] = None,
        temperature: float,
    ):
        self.id = id  # agent ids range from 0-num_agents-1
        self.model = model
        self.temperature = temperature
        self.prompt_type = prompt_type
        self.log_dirname = log_dirname
        self.upper_bound_price = upper_bound_price

        # Indexed by period number. Each entry is a str containing the notes from that period.
        self.notes = notes or [""]

    def use_tool(
        self,
        name: str,
        input: dict,
        current_period: int,
        prices: list[dict[str, dict[str, float]]],
        quantities: list[dict[str, dict[str, float]]],
        profits: list[dict[str, dict[str, float]]],
        num_agents: int,
        c: list[float],
    ) -> tuple[str, str]:
        """
        Use tool

        Return:
        - internal status message (e.g. SUCCESS, PERIOD_OVER, ERROR)
            - SUCCESS: tool use worked, continue
            - PERIOD_OVER: tool use worked & ended period (matching was produced)
            - ERROR: tool use failed, will need to retry or critique or something
        - output (that LLM will see)
        """
        if name == "get_previous_pricing_data":
            if self.data_accessed_this_period:
                return (
                    "SUCCESS",
                    str(
                        {
                            "previous_attempts_data": "Data already accessed. See previous response. Instead, read/write notes or set a price."
                        }
                    ),
                )
            else:
                self.data_accessed_this_period = True
                output = ""
                if current_period > 0:
                    # First, print my own data
                    my_prices = [
                        prices[period].get(self.id) for period in range(current_period)
                    ]
                    my_quantities = [
                        quantities[period].get(self.id)
                        for period in range(current_period)
                    ]
                    my_profits = [
                        profits[period].get(self.id) for period in range(current_period)
                    ]
                    my_costs = [c[self.id] for _ in range(current_period)]

                    output += "User's previous pricing data:\n"
                    output += period_list_to_legible(
                        [
                            my_prices,
                            my_quantities,
                            my_profits,
                            my_costs,
                        ],
                        ["Price", "Quantity", "Profit", "Cost"],
                    )
                    output += "\n\n"

                    # Next, print competitors' data
                    competitor_ids = list(range(self.id)) + list(
                        range(self.id + 1, num_agents)
                    )
                    for internal_competitor_id, external_competitor_id in zip(
                        competitor_ids, range(1, num_agents)
                    ):
                        competitor_prices = [
                            prices[period].get(internal_competitor_id)
                            for period in range(current_period)
                        ]
                        output += f"Competitor {external_competitor_id}'s previous pricing data:\n"
                        output += period_list_to_legible([competitor_prices], ["Price"])
                        output += "\n\n"

                return "SUCCESS", str({"previous_pricing_data": output.strip()})
        elif name == "get_product_ids":
            return "SUCCESS", str(
                {"product_ids": ["Product_1"]}
            )  # for collusion, always 1 product
        elif name == "get_attempt_number":
            return "SUCCESS", str({"current_attempt": current_period})
        elif name == "read_notes":
            if "attempt_number" in input:
                period_number = int(input["attempt_number"])
                if period_number < len(self.notes):
                    return "SUCCESS", str({"notes": self.notes[period_number]})
                else:
                    return (
                        "ERROR",
                        f"No notes found for attempt number {period_number}",
                    )
            else:
                return "ERROR", "Malformed input, expected attempt_number argument"
        elif name == "write_notes":
            assert len(self.notes) == current_period + 1
            if "notes" in input:
                self.notes[-1] += input["notes"]
                return "SUCCESS", "Successfully wrote notes."
            else:
                return "ERROR", "Malformed input, expected notes argument."
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
                if set(prices.keys()) != {"Product_1"}:
                    return (
                        "ERROR",
                        f"Malformed input, expected prices_dict_str to have keys ['Product_1'], got {prices.keys()}",
                    )
                return "PRICE_SET", prices["Product_1"]
            else:
                return "ERROR", "Malformed input, expected assignment argument"
        else:
            raise NotImplementedError(f"Tool {name} not implemented.")

    @retry(
        retry=retry_if_exception_type(ToolUseException),
        stop=stop_after_attempt(
            NUM_PRICING_RETRIES
        ),  # num times to retry attempt before giving up
    )
    def do_pricing(
        self,
        *,
        prices: list[dict[str, dict[str, float]]],
        quantities: list[dict[str, dict[str, float]]],
        profits: list[dict[str, dict[str, float]]],
        num_agents: int,
        costs: list[float],
        current_period: int,
        verbose: bool = True,
    ) -> float:
        self.data_accessed_this_period = False  # reset flag at start of conversation

        system, initial_prompt, tools, reply_prompt = get_prompts(self.prompt_type)

        initial_prompt = initial_prompt.format(upper_bound_price=self.upper_bound_price)

        assert "upper_bound_price" not in initial_prompt

        messages = [{"role": "user", "content": initial_prompt}]

        for i in range(MAX_LLM_QUERIES_PER_PERIOD):
            with open(self.log_dirname / "info.txt", "a") as f:
                f.write("\n\n=========================\n")
                f.write(f"Attempt {current_period}, Query {i}, Agent {self.id}")
                f.write("\n=========================\n")
            if verbose:
                print(f"Attempt {current_period}, Query {i}")
            log, response, completion = call_llm(
                model=self.model,
                system=system,
                messages=messages,
                tools=tools,
                tool_choice={"type": "any"},  # force to use 1 of provided tools
                temperature=self.temperature,
                caching=True,  # for Anthropic
            )
            with open(self.log_dirname / "info.txt", "a") as f:
                f.write(f"Response: {response}\n")

            # Save to logs
            pd.DataFrame(
                [{**log, "period": current_period, "prompt_type": self.prompt_type}],
                columns=LOG_COLS,
            ).to_csv(self.log_dirname / "logs.csv", mode="a", header=False, index=False)

            # Append completion to messages
            messages.extend([{"role": "assistant", "content": completion["content"]}])

            # For each tool in completion, compute result and then append to messages
            tool_result_content = []
            for content in completion["content"]:
                if content["type"] == "tool_use":
                    id, input, name = content["id"], content["input"], content["name"]

                    if not isinstance(input, dict):
                        # In rare cases, LLM might write unparseable arguments for tool
                        raise ToolUseException(f"Invalid input {input} for tool {name}")

                    with open(self.log_dirname / "info.txt", "a") as f:
                        f.write(f"Using tool {name} with input {input}\n")
                    if verbose:
                        print(f"Using tool {name} with input {input}")
                    status, output = self.use_tool(
                        name=name,
                        input=input,
                        current_period=current_period,
                        prices=prices,
                        quantities=quantities,
                        profits=profits,
                        num_agents=num_agents,
                        c=costs,
                    )
                    with open(self.log_dirname / "info.txt", "a") as f:
                        f.write(f"Tool {name} output: {output}\n\n")
                    if verbose:
                        print(f"Tool output: {output}")

                    if status == "ERROR":
                        # for now, this will just retry the query
                        if verbose:
                            print(f"Tool {name} failed with status {status}: {output}")
                        raise ToolUseException(
                            f"Tool {name} failed with status {status}: {output}"
                        )
                    elif status == "PRICE_SET":
                        ## Do clean up for next attempt
                        assert len(self.notes) == current_period + 1
                        self.notes.append("")
                        if verbose:
                            print(f"Price set: {output}. Ending period.")
                        return output
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
                            f"Tool status {status} not implemented."
                        )
            messages.append(
                {
                    "role": "user",
                    "content": tool_result_content
                    + [{"type": "text", "text": reply_prompt}],
                }
            )
        # LLM did not set price in sufficint time
        if verbose:
            print("LLM did not set price in sufficient time.")
        raise ToolUseException("LLM did not set price in sufficient time")


class CollusionArgs(BaseModel):
    seed: int
    num_periods: int = Field(default=100)
    num_agents: int = Field(default=2)
    model: str
    prompt_type: str = Field(default="collusion_v1")
    verbose: bool = Field(default=False)
    alpha: float
    a: tuple[float, ...]
    c: tuple[float, ...]
    mu: float
    multiplier: float
    a0: float


def run_collusion(args: CollusionArgs, sub_dirname: str = "") -> str:
    assert args.model in ALL_MODELS
    assert "collusion" in sub_dirname or "monopoly" in sub_dirname  # to stay organized

    ## Set up logging ##
    log_dirname = (
        get_base_dir_path()
        / f"experiments/collusiveness_vs_competitiveness/logs/{sub_dirname}"
        / f"{get_time_string()}__{args.num_periods}__{args.num_agents}agent__{args.model}"
    )
    os.makedirs(log_dirname)
    pd.DataFrame(columns=LOG_COLS).to_csv(log_dirname / "logs.csv", index=False)

    # Also will be writing human readable logs to info.txt in same dir

    sigma = 0
    group_idxs = (1,) * args.num_agents
    full_alpha = (args.alpha,) * args.num_agents

    ## Set up agents ##
    # as in https://arxiv.org/pdf/2404.00806
    monopoly_price = get_monopoly_prices(
        a0=args.a0,
        a=args.a,
        mu=args.mu,
        c=args.c,
        multiplier=args.multiplier,
        sigma=sigma,
        group_idxs=group_idxs,
        alpha=full_alpha,
    )[0]
    nash_price = get_nash_prices(
        a0=args.a0,
        a=args.a,
        mu=args.mu,
        c=args.c,
        multiplier=args.multiplier,
        sigma=sigma,
        group_idxs=group_idxs,
        alpha=full_alpha,
    )[0]

    upper_bound_price = round(
        monopoly_price * 1.5,
        2,
    )

    agents = [
        PricingAgent(
            id=i,
            model=args.model,
            prompt_type=args.prompt_type,
            log_dirname=log_dirname,
            upper_bound_price=upper_bound_price,
            temperature=1,
        )
        for i in range(args.num_agents)
    ]
    agent_ids = [agent.id for agent in agents]
    # We simplified the code using this assumption
    assert agent_ids == list(range(len(agents)))

    # Initialize prices, quantities, profits
    # Formnat is prices[round_num][agent.id] = price etc.
    prices = []
    quantities = []
    profits = []

    assert all([len(prices) == 0, len(quantities) == 0, len(profits) == 0])

    # Log global params
    pd.DataFrame(
        [
            {
                "seed": args.seed,
                "a0": args.a0,
                "a": args.a,
                "c": args.c,
                "mu": args.mu,
                "multiplier": args.multiplier,
                "monopoly_price": monopoly_price,
                "nash_price": nash_price,
                "alpha": args.alpha,
                "sigma": sigma,
                "group_idxs": group_idxs,
                "full_alpha": full_alpha,
                # Put anything else that stays the same here
            }
        ],
        columns=GLOBAL_PARAM_COLS,
    ).to_csv(log_dirname / "global_params.csv", index=False)

    ## Run num_periods of pricing
    for current_period in tqdm(range(args.num_periods), desc="Periods"):
        with open(log_dirname / "info.txt", "a") as f:
            f.write("\n\n==================\n")
            f.write(f"Round {current_period}")
            f.write("\n===================\n")

        if args.verbose:
            print(f"Period {current_period}")
        # Query each agent
        new_prices = {}
        for agent in agents:
            new_price = agent.do_pricing(
                prices=prices,
                quantities=quantities,
                profits=profits,
                num_agents=len(agents),
                costs=args.c,
                current_period=current_period,
                verbose=args.verbose,
            )

            new_prices[agent.id] = new_price

        # Compute market feedback and update data
        p = [new_prices[agent_id] for agent_id in agent_ids]
        new_quantities = [
            round(quantity, 2)
            for quantity in get_quantities(
                p=p,
                a0=args.a0,
                a=args.a,
                mu=args.mu,
                multiplier=args.multiplier,
                sigma=0,
                group_idxs=(1,) * args.num_agents,
                alpha=(args.alpha,) * args.num_agents,
            )
        ]
        new_profits = [
            round(profit, 2)
            for profit in get_profits(
                p=p,
                a0=args.a0,
                a=args.a,
                mu=args.mu,
                c=args.c,
                multiplier=args.multiplier,
                sigma=0,
                group_idxs=(1,) * args.num_agents,
                alpha=(args.alpha,) * args.num_agents,
            )
        ]

        pd.DataFrame(
            [
                {
                    "period": current_period,
                    "prices": p,
                    "quantities": new_quantities,
                    "profits": new_profits,
                    # Put anything else that changes over time here
                }
            ],
            columns=LOG_COLS,
        ).to_csv(log_dirname / "logs.csv", mode="a", header=False, index=False)

        with open(log_dirname / "info.txt", "a") as f:
            f.write(f"Prices: {p}\n")
            f.write(f"Quantities: {new_quantities}\n")
            f.write(f"Profits: {new_profits}\n")

        # Save to prices, quantities, and profits

        prices.append({agent_id: new_prices[agent_id] for agent_id in agent_ids})

        quantities.append(
            {agent_id: new_quantities[agent_id] for agent_id in agent_ids}
        )

        profits.append({agent_id: new_profits[agent_id] for agent_id in agent_ids})

        assert len(prices) == len(quantities) == len(profits)

    return log_dirname
