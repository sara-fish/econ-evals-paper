from pathlib import Path
import numpy as np
import pandas as pd
import os

from tqdm import tqdm

from econ_evals.experiments.efficiency_vs_equality.instance_generation import (
    compute_max_efficiency_alloc,
    compute_per_worker_revenue_from_alloc,
    generate_task_sizes,
    generate_worker_productivities,
    compute_worker_pay_of_alloc,
    compute_revenue_from_alloc,
)
from econ_evals.utils.helper_functions import (
    get_base_dir_path,
    get_time_string,
    parse_dict,
)
from econ_evals.utils.llm_tools import call_llm

from econ_evals.experiments.efficiency_vs_equality.prompts import get_prompts


from typing import Any, NamedTuple

from tenacity import retry, stop_after_attempt, retry_if_exception_type

MAX_LLM_QUERIES_PER_PERIOD = 40

##  NOTE: throughout this, an alloc maps TASKS to WORKERS.
# I know this is annoying, but it makes some other stuff easier down the line (task IDs are unique, whereas worker IDs are multiply assigned over the course of many periods)


class ToolUseException(Exception):
    pass


LOG_COLS = [
    "seed",
    "period_num",
    # LLMLog
    "prompt_type",
    "model",
    "system",
    "tools",
    "tool_choices",
    "messages",
    "temperature",
    "max_tokens",
    "response",
    "completion",
    # Task-specific
    "alloc",
    "worker_pay",
    "cumulative_worker_pay",
    "company_revenue",
    "worker_to_company_revenue",
    "worker_inequality",
    "status",
]

DATA_COLS = [
    "period_num",
    "alloc",
    "worker_pay",
    "cumulative_worker_pay",
    "company_revenue",
    "worker_to_company_revenue",
    "worker_inequality",
]


def is_valid_matching(
    matching: dict[str, str], worker_ids: list[str], task_ids_this_period: list[str]
) -> tuple[bool, str]:
    """
    Taken from scheduling
    Given a matching (dict mapping task to worker), check if it is a valid 1-1 bijection between worker_ids and task_ids

    Return:
    - True/False (true if matching is valid, false otherwise)
    - if False, string explaining why invalid (that the LLM will read). If True, empty string
    """
    if not isinstance(matching, dict):
        return False, "Assignment must be a dictionary mapping tasks to workers"
    unmatched_workers = set(worker_ids).difference(set(matching.values()))
    unmatched_tasks = set(task_ids_this_period).difference(set(matching.keys()))
    unknown_workers = set(matching.values()).difference(set(worker_ids))
    unknown_tasks = set(matching.keys()).difference(set(task_ids_this_period))
    if unmatched_workers:
        return False, "Assignment doesn't include workers: " + str(unmatched_workers)
    if unmatched_tasks:
        return False, "Assignment doesn't include tasks: " + str(unmatched_tasks)
    if unknown_workers:
        return False, "Assignment includes invalid workers: " + str(unknown_workers)
    if unknown_tasks:
        return False, "Assignment includes invalid tasks: " + str(unknown_tasks)
    return True, ""


class EfficiencyFairnessArgs(NamedTuple):
    model: str
    prompt_type: str
    seed: int
    num_periods: int
    num_workers: int
    worker_productivity_gap: float
    max_worker_productivity_gap: float
    worker_wage: float


def make_readable_period_history(
    worker_ids: list[str],
    task_ids: list[str],
    task_id_to_task_size: dict[str, int],
    worker_per_period_pay: list[dict[str, float]],
    worker_cumulative_pay: list[dict[str, float]],
    worker_to_company_revenue: list[dict[str, float]],
    company_revenue: list[float],
    alloc_attempts: list[dict[str, str]],
) -> str:
    """
    Return: string to be read by LLM with information about previous periods

    - For a given period, includes task IDs and sizes
    - which task each worker did, how much each worker earned in the period,
    - what the cumulative worker pay for each worker and company revenue are.
    """

    if len(alloc_attempts) == 0:
        return "No data yet available."

    assert (
        len(worker_per_period_pay)
        == len(worker_cumulative_pay)
        == len(company_revenue)
        == len(alloc_attempts)
    )
    assert len(task_ids) == len(task_id_to_task_size)
    assert all(len(alloc) == len(worker_ids) for alloc in alloc_attempts)

    output = ""

    for i, alloc in enumerate(alloc_attempts):
        output += f"Period {i}:\n"
        for task_id, worker_id in alloc.items():
            this_period_pay = round(worker_per_period_pay[i][worker_id], 2)
            this_period_company_revenue = round(
                worker_to_company_revenue[i][worker_id], 2
            )
            cumulative_pay = round(worker_cumulative_pay[i][worker_id], 2)
            output += f"\tWorker {worker_id} did Task {task_id} (size {task_id_to_task_size[task_id]}) and was paid ${this_period_pay}. From worker {worker_id} completing task {task_id}, the company earned ${this_period_company_revenue} in revenue. Worker {worker_id}'s total pay so far is ${cumulative_pay}.\n"
        this_period_company_revenue = round(company_revenue[i], 2)
        cumulative_company_revenue = round(sum(company_revenue[: i + 1]), 2)
        output += f"\tThis period, the company earned ${this_period_company_revenue} in revenue. The company's total revenue so far is ${cumulative_company_revenue}.\n"
    return output


class EfficiencyFairnessAgent:
    def __init__(
        self,
        model: str,
        prompt_type: str,
        temperature: float,
        log_dirname: Path,
    ):
        self.model = model
        self.prompt_type = prompt_type
        self.temperature = temperature
        self.log_dirname = log_dirname

        self.notes = [""]
        self.data_accessed_this_period = False

    def use_tool(
        self,
        name: str,
        input: dict,
        period_num: int,
        worker_ids: list[str],
        task_ids: list[str],
        task_ids_this_period: list[str],
        task_id_to_task_size: dict[str, int],
        worker_per_period_pay: list[dict[str, float]],
        worker_cumulative_pay: list[dict[str, float]],
        worker_to_company_revenue: list[dict[str, float]],
        company_revenue: list[float],
        alloc_attempts: list[dict[str, str]],
    ) -> tuple[str, Any]:
        """
        Use tool

        Return:
        - internal status message (e.g. SUCCESS, PERIOD_OVER, ERROR)
            - SUCCESS: tool use worked, continue
            - PERIOD_OVER: tool use worked & ended period (matching was produced)
            - ERROR: tool use failed, will need to retry or critique or something
        - output (that LLM will see)
        """

        if name == "get_worker_ids":
            return "SUCCESS", str({"worker_ids": worker_ids})
        elif name == "get_task_info":
            output = ""
            for task_id in task_ids_this_period:
                output += f"Task {task_id} size = {task_id_to_task_size[task_id]}\n"
            return "SUCCESS", output
        elif name == "get_period_number":
            return "SUCCESS", str({"period_number": period_num})
        elif name == "get_previous_periods_data":
            if self.data_accessed_this_period:
                return (
                    "SUCCESS",
                    str(
                        {
                            "get_previous_periods_data": "Data already accessed this period. See previous response. Instead, write notes or submit an assignment."
                        }
                    ),
                )
            else:
                readable_previous_periods_data = make_readable_period_history(
                    worker_ids=worker_ids,
                    task_ids=task_ids,
                    task_id_to_task_size=task_id_to_task_size,
                    worker_per_period_pay=worker_per_period_pay,
                    worker_cumulative_pay=worker_cumulative_pay,
                    worker_to_company_revenue=worker_to_company_revenue,
                    company_revenue=company_revenue,
                    alloc_attempts=alloc_attempts,
                )
                self.data_accessed_this_period = True
                return "SUCCESS", str(
                    {"previous_periods_data": readable_previous_periods_data}
                )
        elif name == "write_notes":
            assert len(self.notes) == period_num + 1
            if "notes" in input:
                self.notes[-1] += input["notes"]
                return "SUCCESS", "Successfully wrote notes."
            else:
                return "ERROR", "Malformed input, expected notes argument"
        elif name == "read_notes":
            if "period_number" in input:
                period_number = int(input["period_number"])
                if period_number < len(self.notes):
                    return "SUCCESS", str({"notes": self.notes[period_number]})
                else:
                    return (
                        "ERROR",
                        f"No notes found for period number {period_number}",
                    )
            else:
                return "ERROR", "Malformed input, expected period_number argument"
        elif name == "submit_assignment":
            if "assignment" in input:
                matching = input.get("assignment")
                try:
                    matching = parse_dict(matching)  # Google outputs string, not dict
                except ValueError as _:
                    return (
                        "RETRY_ERROR",
                        "Malformed input, could not parse assignment as JSON",
                    )

                is_valid, invalid_reason = is_valid_matching(
                    matching=matching,
                    worker_ids=worker_ids,
                    task_ids_this_period=task_ids_this_period,
                )
                if is_valid:
                    return "PERIOD_OVER", matching
                else:
                    return (
                        "RETRY_ERROR",
                        f"Malformed input: '{invalid_reason}'",
                    )
            else:
                return "ERROR", "Malformed input, expected assignment argument"
        else:
            return "ERROR", f"Invalid tool '{name}'"

    @retry(
        retry=retry_if_exception_type(ToolUseException),
        stop=stop_after_attempt(10),  # num times to retry attempt before giving up
    )
    def make_matching_attempt(
        self,
        worker_ids: list[str],
        task_ids: list[str],
        task_ids_this_period: list[str],
        task_id_to_task_size: dict[str, int],
        worker_per_period_pay: list[dict[str, float]],
        worker_cumulative_pay: list[dict[str, float]],
        worker_to_company_revenue: list[dict[str, float]],
        company_revenue: list[float],
        alloc_attempts: list[dict[str, str]],
        period_num: int,
    ):
        """
        Returns: matching between workers and tasks

        - Calls llm to get matching between workers and tasks e.g. {W1:T1, W2:T3, W3:T2},
        - Checkes if matching is valid, processes if so.
        """
        initial_prompt, system_prompt, tools, reply_prompt = get_prompts(
            self.prompt_type
        )

        messages = [{"role": "user", "content": initial_prompt}]

        for i in range(MAX_LLM_QUERIES_PER_PERIOD):
            with open(self.log_dirname / "info.txt", "a") as f:
                f.write("\n\n=========================\n")
                f.write(f"Period {period_num}, Query {i}")
                f.write("\n=========================\n")

            log, response, completion = call_llm(
                model=self.model,
                system=system_prompt,
                tools=tools,
                messages=messages,
                tool_choice={"type": "any"},
                temperature=self.temperature,
                caching=True,
            )

            with open(self.log_dirname / "info.txt", "a") as f:
                f.write(f"Response: {response}\n")

            # Save to logs
            pd.DataFrame(
                [{**log, "period_num": period_num, "prompt_type": self.prompt_type}],
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

                    # Use tool
                    status, output = self.use_tool(
                        name=name,
                        input=input,
                        period_num=period_num,
                        worker_ids=worker_ids,
                        task_ids=task_ids,
                        task_ids_this_period=task_ids_this_period,
                        task_id_to_task_size=task_id_to_task_size,
                        worker_per_period_pay=worker_per_period_pay,
                        worker_cumulative_pay=worker_cumulative_pay,
                        worker_to_company_revenue=worker_to_company_revenue,
                        company_revenue=company_revenue,
                        alloc_attempts=alloc_attempts,
                    )

                    # Do some logging
                    with open(self.log_dirname / "info.txt", "a") as f:
                        f.write(f"Tool {name} status: {status}\n\n")
                        f.write(f"Tool {name} output: {output}\n\n")

                    # Process the tool
                    if status == "ERROR":
                        # for now, this will just retry the query
                        raise ToolUseException(
                            f"Tool {name} failed with status {status}: {output}"
                        )
                    elif status == "PERIOD_OVER":
                        alloc = output
                        break
                    elif status == "SUCCESS" or status == "RETRY_ERROR":
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
                # Do clean up for next period

                assert len(self.notes) == period_num + 1
                self.notes.append("")
                self.data_accessed_this_period = False
                break

        else:
            # LLM did not run the done_setting_prices command in sufficient time, so advancing period manually here.
            with open(self.log_dirname / "info.txt", "a") as f:
                f.write("LLM did not produce matching in time.")
            raise ToolUseException

        return alloc


def run(args: EfficiencyFairnessArgs):
    params_str = f"{args.model}__{args.num_periods}__{args.num_workers}__{args.worker_productivity_gap}__{args.max_worker_productivity_gap}__{args.worker_wage}__{args.seed}__{args.prompt_type}"

    log_dirname = (
        get_base_dir_path()
        / "experiments/efficiency_vs_equality/logs"
        / f"{get_time_string()}__{params_str}"
    )

    os.makedirs(log_dirname)
    pd.DataFrame(columns=LOG_COLS).to_csv(log_dirname / "logs.csv", index=False)
    pd.DataFrame(columns=DATA_COLS).to_csv(log_dirname / "data.csv", index=False)
    # Also will be writing human readable logs to info.txt in same dir

    my_random = np.random.RandomState(args.seed)

    num_tasks = args.num_workers * args.num_periods

    worker_ids = [f"W{i}" for i in range(1, args.num_workers + 1)]
    task_ids = [f"T{i}" for i in range(1, num_tasks + 1)]

    task_id_to_task_size, task_id_to_worker_id_max_equality = generate_task_sizes(
        num_periods=args.num_periods,
        worker_ids=worker_ids,
        task_ids=task_ids,
        my_random=my_random,
    )

    worker_id_to_productivities = generate_worker_productivities(
        worker_productivity_gap=args.worker_productivity_gap,
        max_worker_productivity_gap=args.max_worker_productivity_gap,
        worker_ids=worker_ids,
        my_random=my_random,
    )

    worker_id_to_wage = {
        worker_id: args.worker_wage for worker_id in worker_ids
    }  # used to be max_tradeoff / wage_scale

    task_id_to_worker_id_max_efficiency = compute_max_efficiency_alloc(
        task_id_to_task_size=task_id_to_task_size,
        worker_id_to_productivities=worker_id_to_productivities,
        task_ids=task_ids,
        num_periods=args.num_periods,
    )

    company_revenue_in_max_efficiency = compute_revenue_from_alloc(
        alloc=task_id_to_worker_id_max_efficiency,
        worker_id_to_productivities=worker_id_to_productivities,
        task_id_to_task_size=task_id_to_task_size,
    )

    company_revenue_in_max_equality = compute_revenue_from_alloc(
        alloc=task_id_to_worker_id_max_equality,
        worker_id_to_productivities=worker_id_to_productivities,
        task_id_to_task_size=task_id_to_task_size,
    )

    assert company_revenue_in_max_efficiency >= company_revenue_in_max_equality

    worker_pay_in_max_efficiency = compute_worker_pay_of_alloc(
        alloc=task_id_to_worker_id_max_efficiency,
        task_id_to_task_size=task_id_to_task_size,
        worker_id_to_wage=worker_id_to_wage,
    )

    worker_pay_in_max_equality = compute_worker_pay_of_alloc(
        alloc=task_id_to_worker_id_max_equality,
        task_id_to_task_size=task_id_to_task_size,
        worker_id_to_wage=worker_id_to_wage,
    )

    worker_inequality_in_max_efficiency = max(
        worker_pay_in_max_efficiency.values()
    ) - min(worker_pay_in_max_efficiency.values())

    worker_inequality_in_max_equality = max(worker_pay_in_max_equality.values()) - min(
        worker_pay_in_max_equality.values()
    )

    assert worker_inequality_in_max_equality < 1e-6  # up to floating point error is 0

    with open(log_dirname / "global_params.csv", "w") as f:
        data = args._asdict()
        data["worker_ids"] = worker_ids
        data["task_ids"] = task_ids
        data["task_id_to_task_size"] = task_id_to_task_size
        data["worker_id_to_wage"] = worker_id_to_wage
        data["worker_id_to_productivities"] = worker_id_to_productivities
        data["task_id_to_worker_id_max_equality"] = task_id_to_worker_id_max_equality
        data["task_id_to_worker_id_max_efficiency"] = (
            task_id_to_worker_id_max_efficiency
        )
        data["company_revenue_in_max_efficiency"] = company_revenue_in_max_efficiency
        data["company_revenue_in_max_equality"] = company_revenue_in_max_equality
        data["worker_pay_in_max_efficiency"] = worker_pay_in_max_efficiency
        data["worker_pay_in_max_equality"] = worker_pay_in_max_equality
        data["worker_inequality_in_max_efficiency"] = (
            worker_inequality_in_max_efficiency
        )
        data["worker_inequality_in_max_equality"] = worker_inequality_in_max_equality
        assert "seed" in data
        pd.DataFrame([data]).to_csv(f, index=False)

    run_efficiency_vs_equality_experiment(
        worker_ids=worker_ids,
        task_ids=task_ids,
        num_periods=args.num_periods,
        task_id_to_task_size=task_id_to_task_size,
        worker_id_to_wage=worker_id_to_wage,
        worker_id_to_productivities=worker_id_to_productivities,
        log_dirname=log_dirname,
        model=args.model,
        prompt_type=args.prompt_type,
        my_random=my_random,
    )


def run_efficiency_vs_equality_experiment(
    worker_ids: list[str],
    task_ids: list[str],
    num_periods: int,
    task_id_to_task_size: dict[str, int],
    worker_id_to_wage: dict[str, float],
    worker_id_to_productivities: dict[str, float],
    log_dirname: Path,
    model: str,
    prompt_type: str,
    my_random: np.random.RandomState,
):
    agent = EfficiencyFairnessAgent(
        model=model,
        prompt_type=prompt_type,
        temperature=1,
        log_dirname=log_dirname,
    )

    alloc_attempts = []  # List of, for each period, the alloc LLM gave
    worker_per_period_pay = []  # List of, for each period, the pay of each worker from that period
    worker_cumulative_pay = []  # List of, for each period, the cumulative pay of each worker up to that period
    worker_to_company_revenue = []  # List of, for each period, how much each worker made the company
    company_revenue = []  # List of, for each period, the company's pay from that period

    for period_num in tqdm(range(num_periods)):
        with open(log_dirname / "info.txt", "a") as f:
            f.write("\n\n==================\n")
            f.write(f"Period {period_num}")
            f.write("\n===================\n")

        task_ids_this_period = task_ids[
            period_num * len(worker_ids) : (period_num + 1) * len(worker_ids)
        ]

        alloc = agent.make_matching_attempt(
            worker_ids=worker_ids,
            task_ids=task_ids,
            task_ids_this_period=task_ids_this_period,
            task_id_to_task_size=task_id_to_task_size,
            worker_per_period_pay=worker_per_period_pay,
            worker_cumulative_pay=worker_cumulative_pay,
            worker_to_company_revenue=worker_to_company_revenue,
            company_revenue=company_revenue,
            alloc_attempts=alloc_attempts,
            period_num=period_num,
        )

        if alloc is None:
            with open(log_dirname / "info.txt", "a") as f:
                f.write("ERROR: LLM did not produce matching in time\n")

            pd.DataFrame(
                [
                    {
                        "period_num": period_num,
                        "alloc": alloc,
                        "status": "Out of time",
                    }
                ],
                columns=LOG_COLS,
            ).to_csv(log_dirname / "logs.csv", mode="a", header=False, index=False)

            raise RuntimeError("Out of time")

        else:
            # continue experiment

            # Compute and save feedback
            alloc_attempts.append(alloc)
            # worker_per_period_pay
            worker_id_to_this_period_pay = compute_worker_pay_of_alloc(
                alloc=alloc,
                task_id_to_task_size=task_id_to_task_size,
                worker_id_to_wage=worker_id_to_wage,
            )
            worker_per_period_pay.append(worker_id_to_this_period_pay)
            # worker_cumulative_pay
            if period_num == 0:
                worker_cumulative_pay.append(
                    {
                        worker_id: worker_id_to_this_period_pay[worker_id]
                        for worker_id in worker_ids
                    }
                )
            else:
                worker_cumulative_pay.append(
                    {
                        worker_id: worker_cumulative_pay[-1][worker_id]
                        + worker_id_to_this_period_pay[worker_id]
                        for worker_id in worker_ids
                    }
                )
            # worker_to_company_revenue
            worker_to_company_revenue.append(
                compute_per_worker_revenue_from_alloc(
                    alloc=alloc,
                    worker_id_to_productivities=worker_id_to_productivities,
                    task_id_to_task_size=task_id_to_task_size,
                )
            )
            # company_revenue
            company_revenue.append(
                compute_revenue_from_alloc(
                    alloc=alloc,
                    worker_id_to_productivities=worker_id_to_productivities,
                    task_id_to_task_size=task_id_to_task_size,
                )
            )
            # worker_inequality
            worker_inequality = max(worker_cumulative_pay[-1].values()) - min(
                worker_cumulative_pay[-1].values()
            )

            with open(log_dirname / "info.txt", "a") as f:
                f.write("Alloc: " + str(alloc) + "\n")
                f.write("Worker pay: " + str(worker_id_to_this_period_pay) + "\n")
                f.write("Company revenue: " + str(company_revenue[-1]) + "\n")
                f.write(
                    "Cumulative worker pay: " + str(worker_cumulative_pay[-1]) + "\n"
                )
                f.write(
                    "Worker to company revenue:"
                    + str(worker_to_company_revenue[-1])
                    + "\n"
                )
                f.write("Worker inequality: " + str(worker_inequality) + "\n")

            pd.DataFrame(
                [
                    {
                        "period_num": period_num,
                        "alloc": alloc,
                        "worker_pay": worker_id_to_this_period_pay,
                        "company_revenue": company_revenue[-1],
                        "worker_to_company_revenue": worker_to_company_revenue[-1],
                        "cumulative_worker_pay": worker_cumulative_pay[-1],
                        "worker_inequality": worker_inequality,
                    }
                ],
                columns=LOG_COLS,
            ).to_csv(log_dirname / "logs.csv", mode="a", header=False, index=False)

            pd.DataFrame(
                [
                    {
                        "period_num": period_num,
                        "alloc": alloc,
                        "worker_pay": worker_id_to_this_period_pay,
                        "company_revenue": company_revenue[-1],
                        "worker_to_company_revenue": worker_to_company_revenue[-1],
                        "cumulative_worker_pay": worker_cumulative_pay[-1],
                        "worker_inequality": worker_inequality,
                    }
                ],
                columns=DATA_COLS,
            ).to_csv(log_dirname / "data.csv", mode="a", header=False, index=False)
