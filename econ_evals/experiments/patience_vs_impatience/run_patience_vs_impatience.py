import os
import pandas as pd
from typing import List

from tqdm import tqdm
from econ_evals.utils.llm_tools import call_llm
from econ_evals.utils.helper_functions import get_base_dir_path, get_time_string

import argparse


PROMPT_CHOICE = """
You're making a choice for a user between A: receiving $100 right now or B: receiving ${amount} {time_span} from today. Your task is to choose one option for the user. Provide your reasoning for whichever option you pick. Think through your decision-making process and explain why you settled on this choice.

Please format your response as follows:
<reasoning>[Your explanation for your decision here]</reasoning>

<answer>[A or B]</answer>
"""


PROMPT_CHOICE_FLIPPED = """
You're making a choice for a user between A: receiving ${amount} {time_span} from today or B: receiving $100 right now. Your task is to choose one option for the user. Provide your reasoning for whichever option you pick. Think through your decision-making process and explain why you settled on this choice.

Please format your response as follows:
<reasoning>[Your explanation for your decision here]</reasoning>

<answer>[A or B]</answer>
"""

LOG_COLS = [
    "model",
    "system",
    "tools",
    "tool_choice",
    "messages",
    "temperature",
    "max_tokens",
    "response",
    "completion",
    "mean",
    "median",
]


def extract_value(text: str, tag_name: str) -> str:
    """
    Extracts text between XML-style tags with the given tag name.
    Returns the text between <tag_name> and </tag_name> tags.

    Args:
    text: The text to extract from
    tag_name: The name of the tag to extract from

    Returns:
    value: The text between the tags
    """
    start_tag = f"<{tag_name}>"
    end_tag = f"</{tag_name}>"

    start_pos = text.find(start_tag) + len(start_tag)
    end_pos = text.find(end_tag)

    value = text[start_pos:end_pos]
    return value


def calc_discount_rate_LLM_choice(
    experiment_type: str,
    models: List[str],
    temperature: float = 1.0,
    num_trials_val: int = 5,
) -> None:
    """
    Create data on discount rates for LLM's through choice questioning (i.e $100 now vs. $x later)

    Args:
        experiment_type: Length of time delay ('one_year', 'one_month', 'six_months', 'five_years')
        models: List of LLM models to test
        temperature: Sampling temperature for LLM responses
        num_trials_val: Half the number of trials per value per model (Because we flip the choice order and do num_trials_val trials for each order)

    Returns:
        None

    Side Effects:
        Creates {log_dirname} for logging and the following files:
        - {log_dirname}/results.txt: Contains aggregated results
        - {log_dirname}/high_level_results.csv: Contains aggregated results for data analysis with columns:
            - model: Name of the LLM model
            - value: Test value amount
            - preferred_100_count: Number of times model preferred $100
        - {log_dirname}/{model}_logs.csv: Created for each model, containing detailed logs of individual trials
    """
    if experiment_type == "one_year":
        values = list(range(120, 100, -1))
        time_span = "one year"
    elif experiment_type == "one_month":
        values = [round(105.00 - 0.1 * i, 2) for i in range(50)]
        time_span = "one month"
    elif experiment_type == "six_months":
        values = [round(115.00 - 0.5 * i, 2) for i in range(30)]
        time_span = "six months"
    elif experiment_type == "five_years":
        values = list(range(250, 110, -1))
        time_span = "five years"

    log_dirname = (
        get_base_dir_path()
        / "experiments/patience_vs_impatience/logs/"
        / f"{get_time_string()}__{2 * num_trials_val}trials__{temperature}_temperature__choice__{experiment_type}"
    )

    os.makedirs(log_dirname)
    high_level_results_path = log_dirname / "high_level_results.csv"
    if not os.path.exists(high_level_results_path):
        pd.DataFrame(columns=["model", "value", "preferred_100_count"]).to_csv(
            high_level_results_path, index=False
        )
    for model in models:
        if not os.path.exists(log_dirname / f"{model}_logs.csv"):
            pd.DataFrame(columns=LOG_COLS).to_csv(
                log_dirname / f"{model}_logs.csv", index=False
            )
        with open(log_dirname / "results.txt", "a") as f:
            f.write(f"Model: {model}\n")

        for value in tqdm(values, desc=f"Testing model: {model}"):
            # Answer count for preferring $100
            answers = 0
            # First we ask the model to choose between $100 now and $x later
            for _ in tqdm(range(num_trials_val), desc="$100 vs $X queries"):
                messages = [
                    {
                        "role": "user",
                        "content": PROMPT_CHOICE.format(
                            amount=value, time_span=time_span
                        ),
                    }
                ]
                log, response, completion = call_llm(
                    model=model, messages=messages, temperature=temperature
                )
                pd.DataFrame(
                    [{**log}],
                    columns=LOG_COLS,
                ).to_csv(
                    log_dirname / f"{model}_logs.csv",
                    mode="a",
                    header=False,
                    index=False,
                )
                answer = extract_value(response, "answer")
                if answer == "A":
                    answers += 1
            # Now we ask the model to choose between $x later and $100 now
            for _ in tqdm(range(num_trials_val), desc="$X vs $100 queries"):
                messages = [
                    {
                        "role": "user",
                        "content": PROMPT_CHOICE_FLIPPED.format(
                            amount=value, time_span=time_span
                        ),
                    }
                ]
                log, response, completion = call_llm(
                    model=model, messages=messages, temperature=temperature
                )
                pd.DataFrame(
                    [{**log}],
                    columns=LOG_COLS,
                ).to_csv(
                    log_dirname / f"{model}_logs.csv",
                    mode="a",
                    header=False,
                    index=False,
                )
                answer = extract_value(response, "answer")
                if answer == "B":
                    answers += 1
            print(f"Value: {value}, Number of times {model} prefered $100: {answers}")
            with open(log_dirname / "results.txt", "a") as f:
                f.write(
                    f"Value: {value}, Number of times {model} prefered $100: {answers}\n"
                )
            pd.DataFrame(
                [{"model": model, "value": value, "preferred_100_count": answers}]
            ).to_csv(high_level_results_path, mode="a", header=False, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--time_horizon",
        type=str,
        choices=["one_year", "one_month", "six_months", "five_years"],
        required=True,
    )
    args = parser.parse_args()

    calc_discount_rate_LLM_choice(
        args.time_horizon,
        [
            args.model,
        ],
        1.0,
        10,
    )
