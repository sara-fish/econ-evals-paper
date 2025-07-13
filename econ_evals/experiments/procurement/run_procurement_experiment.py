import os
from pathlib import Path
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from econ_evals.experiments.procurement.opt_solver import Menu
from econ_evals.utils.helper_functions import (
    get_base_dir_path,
    get_time_string,
    parse_dict,
)
from econ_evals.utils.llm_tools import call_llm
from prompts import get_prompts

from econ_evals.experiments.procurement.generate_instance import generate_instance
from econ_evals.experiments.procurement.opt_solver import compute_opt, evaluate_alloc
from tenacity import retry, stop_after_attempt, retry_if_exception_type

from typing import Any, Literal, Optional
from tqdm import tqdm

MAX_LLM_QUERIES_PER_PERIOD = 40

MAX_SUBMIT_PLAN_ATTEMPTS = 10


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
    "alloc",
    "status",
    "cost",
    "utility",
    "is_feasible",
    "infeasible_reason",
    "request_timestamp",
    "response_timestamp",
]


class ToolUseException(Exception):
    pass


def make_readable_previous_attempts_data(
    alloc_attempts: list[dict[str, str]],
    alloc_costs: list[float],
    alloc_utilities: list[float],
    alloc_is_feasibles: list[bool],
    alloc_infeasible_reasons: list[str],
) -> str:
    assert (
        len(alloc_attempts)
        == len(alloc_costs)
        == len(alloc_utilities)
        == len(alloc_infeasible_reasons)
        == len(alloc_is_feasibles)
    )

    output = ""
    for attempt_num in range(len(alloc_attempts)):
        output += f"Attempt {attempt_num}:\n"
        output += f"Purchase plan proposed: {alloc_attempts[attempt_num]}\n"
        if alloc_is_feasibles[attempt_num]:
            output += f"Purchase plan results: supports {alloc_utilities[attempt_num]:.2f} workers and incurs cost of {alloc_costs[attempt_num]:.2f}\n"
        else:
            output += (
                f"Purchase plan was invalid: {alloc_infeasible_reasons[attempt_num]}\n"
            )
        output += "\n"
    return output


class ProcurementAgent:
    def __init__(
        self,
        *,
        id: str,
        model: str,
        prompt_type: str,
        temperature: float,
        log_dirname: Path,
    ):
        self.id = id
        self.model = model
        self.prompt_type = prompt_type
        self.temperature = temperature
        self.log_dirname = log_dirname

        self.notes = [""]
        self.data_accessed_this_period = False
        self.menu_accessed_this_period = False

    def use_tool(
        self,
        name: str,
        input: dict,
        menu: Menu,
        budget: float,
        alloc_attempts: list[dict[str, str]],
        alloc_costs: list[float],
        alloc_utilities: list[float],
        alloc_is_feasibles: list[bool],
        alloc_infeasible_reasons: list[str],
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

        if name == "get_previous_purchase_data":
            if self.data_accessed_this_period:
                return (
                    "SUCCESS",
                    str(
                        {
                            "previous_attempts_data": "Data already accessed. See previous response. Instead, read/write notes or submit a purchase plan."
                        }
                    ),
                )
            else:
                readable_previous_attempts_data = make_readable_previous_attempts_data(
                    alloc_attempts=alloc_attempts,
                    alloc_costs=alloc_costs,
                    alloc_utilities=alloc_utilities,
                    alloc_is_feasibles=alloc_is_feasibles,
                    alloc_infeasible_reasons=alloc_infeasible_reasons,
                )
                self.data_accessed_this_period = True
                return "SUCCESS", str(
                    {"previous_purchase_data": readable_previous_attempts_data}
                )

        elif name == "get_equipment_information":
            if self.menu_accessed_this_period:
                return "SUCCESS", str(
                    "Menu already accessed. See previous response. Instead, read/write notes or submit a purchase plan."
                )
            else:
                self.menu_accessed_this_period = True
                return "SUCCESS", menu.to_str()
        elif name == "get_budget":
            return "SUCCESS", str({"budget": budget})
        elif name == "get_attempt_number":
            return "SUCCESS", str({"attempt_number": len(alloc_attempts)})
        elif name == "write_notes":
            assert len(self.notes) == len(alloc_attempts) + 1
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
        elif name == "submit_purchase_plan":
            if "purchase_plan" in input:
                alloc = input.get("purchase_plan")
                try:
                    alloc = parse_dict(alloc)
                except ValueError as _:
                    return (
                        "ERROR",
                        "Malformed input, could not parse purchase_plan as JSON",
                    )
                # Fill unspecified values with 0
                for entry_id in menu.id_to_entries.keys():
                    if entry_id not in alloc:
                        alloc[entry_id] = 0
                if isinstance(alloc, dict) and set(alloc.keys()) == set(
                    menu.id_to_entries.keys()
                ):
                    return "PERIOD_OVER", alloc
                else:
                    return (
                        "ERROR",
                        "Malformed input, expected assignment to be 1-1 mapping of worker to task",
                    )
            else:
                return "ERROR", "Malformed input, expected assignment argument"
        else:
            return "ERROR", f"Invalid tool '{name}'"

    @retry(
        retry=retry_if_exception_type(ToolUseException),
        stop=stop_after_attempt(
            MAX_SUBMIT_PLAN_ATTEMPTS
        ),  # num times to retry attempt before giving up
    )
    def make_purchase_attempt(
        self,
        *,
        alloc_attempts: list[dict[str, int]],
        alloc_costs: list[float],
        alloc_utilities: list[float],
        alloc_is_feasibles: list[bool],
        alloc_infeasible_reasons: list[str],
        max_queries: int,
        menu: Menu,
        budget: float,
        verbose: bool = True,
    ) -> tuple[Optional[dict[str, int]], str]:
        """
        Return:
        - alloc (or None if none provided)
        - str (status message if alloc was None)
        """

        self.data_accessed_this_period = False
        self.menu_accessed_this_period = False

        initial_prompt, system, tools, tools_action_only, reply_prompt = get_prompts(
            self.prompt_type
        )

        attempt_num = len(alloc_attempts)

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
                caching=True,  # temporary since it's not working yet
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
                        menu=menu,
                        budget=budget,
                        alloc_attempts=alloc_attempts,
                        alloc_costs=alloc_costs,
                        alloc_utilities=alloc_utilities,
                        alloc_is_feasibles=alloc_is_feasibles,
                        alloc_infeasible_reasons=alloc_infeasible_reasons,
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
                        alloc = output
                        break
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
                assert len(self.notes) == len(alloc_attempts) + 1
                self.notes.append("")
                self.data_accessed_this_period = False
                self.menu_accessed_this_period = False
                if verbose:
                    print("Period over.")
                break

        else:
            # LLM did not run the done_setting_prices command in sufficient time, so advancing period manually here.
            if verbose:
                print("LLM did not produce alloc in time.")
            raise ToolUseException("LLM did not produce alloc in time")

        return alloc, ""


def run_scheduling_experiment(
    menu: Menu,
    item_groups: list[list[str]],
    item_to_effectiveness: dict[str, int],
    group_weights: list[float],
    agg_type: Literal["min", "prod"],
    budget: int,
    opt_utility: float,
    log_dirname: str,
    num_attempts: int,
    prompt_type: str,
    model: str,
    verbose: bool,
):
    agent = ProcurementAgent(
        id="0",
        model=model,
        prompt_type=prompt_type,
        temperature=1,
        log_dirname=log_dirname,
    )

    alloc_attempts = []  # List of, for each attempt #, purchase plan given (allocation)
    alloc_costs = []  # List of, for each attempt #, cost of purchase plan
    alloc_utilities = []  # List of, for each attempt #, utility of purchase plan
    alloc_is_feasibles = []  # List of for each attempt #, whether purchase plan is feasible (bool)
    alloc_infeasible_reasons = []  # List of for each attempt #, reason for infeasibility (str) (if it was feasible, then "")

    for attempt_num in tqdm(range(num_attempts)):
        with open(log_dirname / "info.txt", "a") as f:
            f.write("\n\n==================\n")
            f.write(f"Attempt {attempt_num}")
            f.write("\n===================\n")

        alloc, status = agent.make_purchase_attempt(
            alloc_attempts=alloc_attempts,
            alloc_costs=alloc_costs,
            alloc_utilities=alloc_utilities,
            alloc_is_feasibles=alloc_is_feasibles,
            alloc_infeasible_reasons=alloc_infeasible_reasons,
            max_queries=MAX_LLM_QUERIES_PER_PERIOD,
            menu=menu,
            budget=budget,
            verbose=verbose,
        )

        if alloc is None:
            with open(log_dirname / "info.txt", "a") as f:
                f.write(f"LLM didn't produce matching: {status}")

            pd.DataFrame(
                [
                    {
                        "attempt_num": attempt_num,
                        "alloc": alloc,
                        "status": status,
                    }
                ],
                columns=LOG_COLS,
            ).to_csv(log_dirname / "logs.csv", mode="a", header=False, index=False)

            return status

        else:
            assert alloc is not None

            alloc_attempts.append(alloc)

            is_feasible, invalid_reason, total_cost, total_utility = evaluate_alloc(
                menu=menu,
                alloc=alloc,
                item_groups=item_groups,
                item_to_effectiveness=item_to_effectiveness,
                budget=budget,
                group_weights=group_weights,
                agg_type=agg_type,
            )

            alloc_costs.append(total_cost)
            alloc_utilities.append(total_utility)
            alloc_is_feasibles.append(is_feasible)
            alloc_infeasible_reasons.append(invalid_reason)

            with open(log_dirname / "info.txt", "a") as f:
                f.write(f"Purchase plan: {alloc}\n")
                f.write(f"Cost: {total_cost}\n")
                f.write(f"Utility: {total_utility}\n")
                f.write(f"Is feasible: {is_feasible}\n")
                if not is_feasible:
                    f.write(f"Reason for infeasibility: {invalid_reason}\n")
            if verbose:
                print(f"Purchase plan: {alloc}")
                print(f"Cost: {total_cost}")
                print(f"Utility: {total_utility}")
                print(f"Is feasible: {is_feasible}")
                if not is_feasible:
                    print(f"Reason for infeasibility: {invalid_reason}")

            pd.DataFrame(
                [
                    {
                        "attempt_num": attempt_num,
                        "alloc": alloc,
                        "cost": total_cost,
                        "utility": total_utility,
                        "is_feasible": is_feasible,
                        "infeasible_reason": invalid_reason,
                    }
                ],
                columns=LOG_COLS,
            ).to_csv(log_dirname / "logs.csv", mode="a", header=False, index=False)

            if (
                total_utility >= opt_utility - 1e-4 and is_feasible
            ):  # We hit OPT, subject to slight floating point errors
                with open(log_dirname / "info.txt", "a") as f:
                    f.write("Found optimal solution\n")
                # return "Found optimal solution" # don't manually break off -- wait for LLM to say it's done


class ProcurementArgs(BaseModel):
    num_attempts: int = Field(default=20)
    prompt_type: str = Field(default="v1")
    num_inputs: int
    num_alternatives_per_input: int
    num_entries: int
    NUM_ITEMS_PER_ENTRY_P: float = Field(default=0.8)
    QUANTITY_PER_ITEM_P: float = Field(default=0.5)
    OFFER_QTY_IN_SAMPLE_BUNDLE_P: float = Field(default=0.5)
    MIN_EFFECTIVENESS: int = Field(default=1)
    MAX_EFFECTIVENESS: int = Field(default=1)
    agg_type: Literal["min", "prod"]
    seed: int = Field(default=0)
    model: str = Field(default="gpt-4o-2024-08-06")
    verbose: bool = Field(default=False)


def run(args: ProcurementArgs, log_subdirname: str = ""):
    # Set log directory
    params_str = f"{args.prompt_type}__{args.model}__{args.num_entries}__{args.num_inputs}__{args.num_alternatives_per_input}__{args.agg_type}"
    log_dirname = (
        get_base_dir_path()
        / "experiments/procurement/logs/"
        / log_subdirname
        / f"{get_time_string()}__{params_str}"
    )
    os.makedirs(log_dirname)
    if not os.path.exists(log_dirname / "logs.csv"):
        pd.DataFrame(columns=LOG_COLS).to_csv(log_dirname / "logs.csv", index=False)
    # Also will be writing human readable logs to info.txt in same dir

    my_random = np.random.RandomState(args.seed)

    # For now, just use uniform group weights
    if args.agg_type == "min":
        group_weights = [1 for _ in range(args.num_inputs)]
    elif args.agg_type == "prod":
        group_weights = [1 / args.num_inputs for _ in range(args.num_inputs)]
    else:
        raise NotImplementedError(f"Aggregation type {args.agg_type} not implemented")

    # Generate instance
    menu, budget, item_groups, start_alloc, item_to_effectiveness = generate_instance(
        num_inputs=args.num_inputs,
        num_alternatives_per_input=args.num_alternatives_per_input,
        num_entries=args.num_entries,
        my_random=my_random,
        NUM_ITEMS_PER_ENTRY_P=args.NUM_ITEMS_PER_ENTRY_P,
        QUANTITY_PER_ITEM_P=args.QUANTITY_PER_ITEM_P,
        OFFER_QTY_IN_SAMPLE_BUNDLE_P=args.OFFER_QTY_IN_SAMPLE_BUNDLE_P,
        MIN_EFFECTIVENESS=args.MIN_EFFECTIVENESS,
        MAX_EFFECTIVENESS=args.MAX_EFFECTIVENESS,
    )

    # Compute opt of generated instance
    opt_alloc, opt_alloc_log = compute_opt(
        menu,
        budget,
        item_groups,
        item_to_effectiveness,
        group_weights=group_weights,
        agg_type=args.agg_type,
        start_alloc=start_alloc,
    )
    is_feasible, invalid_reason, opt_cost, opt_utility = evaluate_alloc(
        menu=menu,
        alloc=opt_alloc,
        item_groups=item_groups,
        item_to_effectiveness=item_to_effectiveness,
        budget=budget,
        group_weights=group_weights,
        agg_type=args.agg_type,
    )

    assert is_feasible
    assert not invalid_reason

    # Write global params to file
    with open(log_dirname / "global_params.csv", "w") as f:
        data = vars(args)
        data["num_inputs"] = args.num_inputs
        data["num_alternatives_per_input"] = args.num_alternatives_per_input
        data["num_entries"] = args.num_entries
        data["agg_type"] = args.agg_type
        data["seed"] = args.seed
        data["model"] = args.model
        data["verbose"] = args.verbose
        data["seed"] = args.seed
        data["NUM_ITEMS_PER_ENTRY_P"] = args.NUM_ITEMS_PER_ENTRY_P
        data["QUANTITY_PER_ITEM_P"] = args.QUANTITY_PER_ITEM_P
        data["OFFER_QTY_IN_SAMPLE_BUNDLE_P"] = args.OFFER_QTY_IN_SAMPLE_BUNDLE_P
        data["MIN_EFFECTIVENESS"] = args.MIN_EFFECTIVENESS
        data["MAX_EFFECTIVENESS"] = args.MAX_EFFECTIVENESS
        data["group_weights"] = group_weights
        data["menu"] = menu.to_dict()
        data["budget"] = budget
        data["item_groups"] = item_groups
        data["item_to_effectiveness"] = item_to_effectiveness
        data["start_alloc"] = start_alloc
        data["opt_alloc"] = opt_alloc
        data["opt_cost"] = opt_cost
        data["opt_utility"] = opt_utility
        data["opt_alloc_log"] = opt_alloc_log
        pd.DataFrame([data]).to_csv(f, index=False)

    # Now run experiment

    run_scheduling_experiment(
        menu=menu,
        item_groups=item_groups,
        item_to_effectiveness=item_to_effectiveness,
        budget=budget,
        group_weights=group_weights,
        agg_type=args.agg_type,
        opt_utility=opt_utility,
        log_dirname=log_dirname,
        num_attempts=args.num_attempts,
        prompt_type=args.prompt_type,
        model=args.model,
        verbose=args.verbose,
    )
