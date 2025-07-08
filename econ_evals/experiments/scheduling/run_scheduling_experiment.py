import os
from pathlib import Path
import numpy as np
import pandas as pd
from pydantic import BaseModel, Field
from tqdm import tqdm

from econ_evals.experiments.scheduling.generate_preferences import (
    generate_preferences,
)
from econ_evals.utils.helper_functions import (
    get_base_dir_path,
    get_time_string,
    parse_dict,
)
from econ_evals.utils.llm_tools import call_llm
from prompts import get_prompts
from stable_matching_environment import (
    get_blocking_pairs,
    is_valid_matching,
    select_blocking_pair,
)
from tenacity import retry, stop_after_attempt, retry_if_exception_type

from typing import Any, Literal

MAX_LLM_QUERIES_PER_PERIOD = 40

MAX_MATCHING_ATTEMPTS = 10


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
    "worker_prefs",
    "task_prefs",
    "matching",
    "blocking_pairs",
    "blocking_pair",
]


class ToolUseException(Exception):
    pass


def make_readable_previous_attempts_data(
    matching_attempts: list[dict[str, str]],
    blocking_pair_results: list[list[tuple[str, str]]],
) -> str:
    assert len(matching_attempts) == len(blocking_pair_results)
    output = ""
    for attempt_num in range(len(matching_attempts)):
        output += f"Attempt {attempt_num}:\n"
        output += f"Assignment proposed: {matching_attempts[attempt_num]}\n"
        for i, blocking_pair in enumerate(blocking_pair_results[attempt_num]):
            worker, task = blocking_pair
            task_of_worker = matching_attempts[attempt_num][worker]
            worker_of_task = {v: k for k, v in matching_attempts[attempt_num].items()}[
                task
            ]
            output += f"\t({i+1}) Problem with assignment: worker {worker} was matched to task {task_of_worker} and worker {worker_of_task} was assigned to {task}. However, worker {worker} would have preferred task {task}, and in fact worker {worker} is more suited to task {task} than worker {worker_of_task}.\n"
        output += "\n"
    return output


class SchedulingAgent:
    def __init__(
        self,
        *,
        id: str,
        model: str,
        prompt_type: str,
        final_prompt_type: str,
        temperature: float,
        log_dirname: Path,
    ):
        self.id = id
        self.model = model
        self.prompt_type = prompt_type
        self.final_prompt_type = final_prompt_type
        self.temperature = temperature
        self.log_dirname = log_dirname

        self.notes = [""]
        self.data_accessed_this_period = False

    def use_tool(
        self,
        name: str,
        input: dict,
        matching_attempts: list[dict[str, str]],
        blocking_pair_results: list[list[tuple[str, str]]],
        worker_ids: list[str],
        task_ids: list[str],
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
        if name == "get_attempt_number":
            return "SUCCESS", str({"attempt_number": len(matching_attempts)})
        elif name == "get_worker_ids":
            return "SUCCESS", str({"worker_ids": worker_ids})
        elif name == "get_task_ids":
            return "SUCCESS", str({"task_ids": task_ids})
        elif name == "get_previous_attempts_data":
            if self.data_accessed_this_period:
                return (
                    "SUCCESS",
                    str(
                        {
                            "previous_attempts_data": "Data already accessed this period. See previous response. Instead, read/write notes or submit an assignment."
                        }
                    ),
                )
            else:
                readable_previous_attempts_data = make_readable_previous_attempts_data(
                    matching_attempts, blocking_pair_results
                )
                self.data_accessed_this_period = True
                return "SUCCESS", str(
                    {"previous_attempts_data": readable_previous_attempts_data}
                )
        elif name == "write_notes":
            assert len(self.notes) == len(matching_attempts) + 1
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
        elif name == "submit_assignment":
            if "assignment" in input:
                matching = input.get("assignment")
                try:
                    matching = parse_dict(matching)  # Google outputs string, not dict
                except ValueError as _:
                    return (
                        "ERROR",
                        "Malformed input, could not parse assignment as JSON",
                    )
                is_valid, invalid_reason = is_valid_matching(
                    matching, worker_ids, task_ids
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
        stop=stop_after_attempt(
            MAX_MATCHING_ATTEMPTS
        ),  # num times to retry attempt before giving up
    )
    def make_matching_attempt(
        self,
        *,
        matching_attempts: list[dict[str, str]],
        blocking_pair_results: list[list[tuple[str, str]]],
        worker_ids: list[str],
        task_ids: list[str],
        max_queries: int,
        is_final_attempt: bool,
        verbose: bool = True,
    ):
        attempt_num = len(matching_attempts)
        if is_final_attempt:
            prompt_type = self.final_prompt_type
        else:
            prompt_type = self.prompt_type

        initial_prompt, system, tools, tools_action_only, reply_prompt = get_prompts(
            prompt_type
        )

        self.data_accessed_this_period = False

        messages = [{"role": "user", "content": initial_prompt}]

        matching = None

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
                tools=(
                    tools if i != max_queries - 1 else tools_action_only
                ),  # For final attempt, require taking an action
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
                        matching_attempts=matching_attempts,
                        blocking_pair_results=blocking_pair_results,
                        worker_ids=worker_ids,
                        task_ids=task_ids,
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
                        matching = output
                        break
                    elif status == "SUCCESS":
                        tool_result_content.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": id,
                                "content": output,
                            }
                        )
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
                assert len(self.notes) == len(matching_attempts) + 1
                self.notes.append("")
                self.data_accessed_this_period = False
                if verbose:
                    print("Period over.")
                break

        else:
            if verbose:
                print("LLM did not produce matching in time.")
            raise ToolUseException("LLM did not produce matching in time")

        return matching


def run_scheduling_experiment(
    worker_prefs: dict[str, list[str]],
    task_prefs: dict[str, list[str]],
    log_dirname: str,
    num_attempts: int,
    num_blocking_pairs: int,
    prompt_type: str,
    final_prompt_type: str,
    model: str,
    verbose: bool,
    blocking_pair_selection_method: str,
    my_random: np.random.RandomState,
) -> str:
    """
    Return value:
    - "Out of time", LLM did not produce matching in time
    - "Successfully found stable matching"
    - "Exited normally", LLM didn't find stable matching, but everything worked until the end
    """
    assert len(worker_prefs) == len(task_prefs)
    for _, worker_pref_list in task_prefs.items():
        assert set(worker_prefs.keys()) == set(worker_pref_list)
    for _, task_pref_list in worker_prefs.items():
        assert set(task_prefs.keys()) == set(task_pref_list)

    agent = SchedulingAgent(
        id="0",
        model=model,
        prompt_type=prompt_type,
        final_prompt_type=final_prompt_type,
        temperature=1,
        log_dirname=log_dirname,
    )

    matching_attempts = []  # List of, for each attempt #, matching LLM gave
    blocking_pairs_results = []  # List of, for each attempt #, blocking pairs matching had (INTERNAL -- LLM doesn't see this)
    blocking_pair_results = []  # List of, for each attempt #, blocking pair(s) given back to LLM (up to num_blocking_pairs)

    for attempt_num in tqdm(range(num_attempts)):
        with open(log_dirname / "info.txt", "a") as f:
            f.write("\n\n==================\n")
            f.write(f"Attempt {attempt_num}")
            f.write("\n===================\n")

        is_final_attempt = attempt_num >= num_attempts - 1

        matching = agent.make_matching_attempt(
            matching_attempts=matching_attempts,
            blocking_pair_results=blocking_pair_results,
            max_queries=MAX_LLM_QUERIES_PER_PERIOD,
            worker_ids=list(worker_prefs.keys()),
            task_ids=list(task_prefs.keys()),
            is_final_attempt=is_final_attempt,
            verbose=verbose,
        )

        if matching is None:  # LLM did not produce matching in time
            with open(log_dirname / "info.txt", "a") as f:
                f.write("ERROR: LLM did not produce matching in time\n")

            pd.DataFrame(
                [
                    {
                        "attempt_num": attempt_num,
                        "worker_prefs": worker_prefs,
                        "task_prefs": task_prefs,
                        "matching": matching,
                    }
                ],
                columns=LOG_COLS,
            ).to_csv(log_dirname / "logs.csv", mode="a", header=False, index=False)

            raise RuntimeError("Out of time")

        else:
            # Otherwise, continue experiment

            blocking_pairs = get_blocking_pairs(
                matching=matching,
                worker_prefs=worker_prefs,
                task_prefs=task_prefs,
            )

            if blocking_pairs:
                blocking_pair = select_blocking_pair(
                    blocking_pairs=blocking_pairs,
                    num_blocking_pairs=num_blocking_pairs,
                    worker_prefs=worker_prefs,
                    task_prefs=task_prefs,
                    blocking_pair_selection_method=blocking_pair_selection_method,
                    matching=matching,
                    matching_attempts=matching_attempts,
                    blocking_pair_results=blocking_pair_results,
                    my_random=my_random,
                )
            else:
                blocking_pair = None

            matching_attempts.append(matching)
            blocking_pairs_results.append(blocking_pairs)
            blocking_pair_results.append(blocking_pair)

            with open(log_dirname / "info.txt", "a") as f:
                f.write(
                    f"Blocking pairs: {blocking_pairs} ({len(blocking_pairs)} total)\n"
                )
                f.write(f"Blocking pair(s) shared: {blocking_pair}\n")
            if verbose:
                print(f"Blocking pairs: {blocking_pairs} ({len(blocking_pairs)} total)")
                print(f"Blocking pair(s) shared: {blocking_pair}")

            pd.DataFrame(
                [
                    {
                        "attempt_num": attempt_num,
                        "worker_prefs": worker_prefs,
                        "task_prefs": task_prefs,
                        "matching": matching,
                        "blocking_pairs": blocking_pairs,
                        "blocking_pair": blocking_pair,
                    }
                ],
                columns=LOG_COLS,
            ).to_csv(log_dirname / "logs.csv", mode="a", header=False, index=False)

            if blocking_pair is None:
                return "Successfully found stable matching"
    return "Exited normally"


class SchedulingArgs(BaseModel):
    num_attempts: int = Field(
        default=20,
    )
    prompt_type: Literal["v1", "v1_reasoning"] = Field(default="v1")
    final_prompt_type: Literal["final_attempt_v1", "final_attempt_v1_reasoning"] = (
        Field(default="final_attempt_v1")
    )
    num_workers: int
    num_blocking_pairs: int = Field(default=1)
    score_gap_worker: float | None = Field(
        default=None,
    )
    score_gap_task: float | None = Field(
        default=None,
    )
    seed: int = Field(default=0)
    model: str
    verbose: bool = Field(default=False)
    blocking_pair_selection_method: Literal["random", "random_cache"] = Field(
        default="random_cache"
    )


def run(args: SchedulingArgs, log_subdirname: str = ""):
    # Set log directory
    params_str = f"{args.num_workers}__{args.prompt_type}__{args.model}__{args.blocking_pair_selection_method}"
    log_dirname = (
        get_base_dir_path()
        / "experiments/scheduling/logs/"
        / log_subdirname
        / f"{get_time_string()}__{params_str}"
    )
    os.makedirs(log_dirname)
    if not os.path.exists(log_dirname / "logs.csv"):
        pd.DataFrame(columns=LOG_COLS).to_csv(log_dirname / "logs.csv", index=False)
    # Also will be writing human readable logs to info.txt in same dir

    my_random = np.random.RandomState(args.seed)

    worker_ids = [f"W{i}" for i in range(1, args.num_workers + 1)]
    task_ids = [f"T{i}" for i in range(1, args.num_workers + 1)]

    worker_prefs, task_prefs = generate_preferences(
        worker_ids=worker_ids,
        task_ids=task_ids,
        score_gap_worker=args.score_gap_worker,
        score_gap_task=args.score_gap_task,
        my_random=my_random,
    )

    # Write global params to file
    with open(log_dirname / "global_params.csv", "w") as f:
        data = vars(args)
        data["worker_ids"] = worker_ids
        data["task_ids"] = task_ids
        data["worker_prefs"] = worker_prefs
        data["task_prefs"] = task_prefs
        data["seed"] = args.seed
        pd.DataFrame([data]).to_csv(f, index=False)

    # Now run experiment

    status = run_scheduling_experiment(
        worker_prefs=worker_prefs,
        task_prefs=task_prefs,
        log_dirname=log_dirname,
        num_attempts=args.num_attempts,
        num_blocking_pairs=args.num_blocking_pairs,
        prompt_type=args.prompt_type,
        final_prompt_type=args.final_prompt_type,
        model=args.model,
        verbose=args.verbose,
        blocking_pair_selection_method=args.blocking_pair_selection_method,
        my_random=my_random,
    )

    global_params = pd.read_csv(log_dirname / "global_params.csv")
    global_params["status"] = status
    global_params.to_csv(log_dirname / "global_params.csv", index=False)
