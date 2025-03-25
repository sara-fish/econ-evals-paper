from ast import literal_eval
import json
from pathlib import Path
import os
from datetime import datetime
from typing import Union


def get_base_dir_path():
    """
    Returns local system path to the directory where the econ_evals package is.
    So, this is the directory where utils/, experiments/, etc are.
    """
    base_dir_name = "econ_evals"

    path = Path(os.path.abspath(os.path.dirname(__file__)))
    current_path_parts = list(path.parts)
    base_dir_idx = (
        len(current_path_parts) - current_path_parts[::-1].index(base_dir_name) - 1
    )

    base_dir_path = Path(*current_path_parts[: 1 + base_dir_idx])
    return base_dir_path


def get_time_string():
    now = datetime.now()
    time_string = now.strftime("%Y-%m-%d-%H-%M-%S")

    return time_string


def parse_dict(input: Union[str, dict]) -> dict:
    """
    Given a LLM's output that is trying too be a dict mapping str to number, parse it.
    Used for scheduling and procurement.
    """
    if isinstance(input, dict):
        return input
    else:
        # Step 1: try parsing with literal_eval
        try:
            output = literal_eval(input)
            assert isinstance(output, dict)
        except (ValueError, AssertionError, SyntaxError) as _:
            pass
        else:
            return output
        # Step 2: try encoding thing
        try:
            # Sometimes Gemini writes {\\"W1\\": \\"T1\\", ...
            output = input.encode("utf-8").decode("unicode_escape")
            output = literal_eval(output)
            assert isinstance(output, dict)
        except (ValueError, AssertionError, SyntaxError) as _:
            pass
        else:
            return output
        # Step 3: try parsing JSON (for nested args, like in procurement, this is needed)
        try:
            output = json.loads(input)
            assert isinstance(output, dict)
        except json.JSONDecodeError as _:
            pass
        else:
            return output
        raise ValueError(f"Could not parse input {input}")
