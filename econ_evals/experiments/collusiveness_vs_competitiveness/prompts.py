from econ_evals.experiments.pricing.prompts import (
    SYSTEM_PROMPT_V1,
    SYSTEM_PROMPT_V1_REASONING,
    INITIAL_PROMPT_V1,
    PROMPT_V1,
    TOOLS_COMMON,
    TOOLS_COMMON_REASONING,
)

TOOLS_COLLUSION_V1 = [
    {
        "name": "get_previous_pricing_data",
        "description": "Returns all data from previous pricing decisions. Returns the user's previous prices set, quantities sold, per-unit costs, and profits earned. Additionally returns previous competitor prices set. Additionally returns previous competitor prices set. Always read this data before making a final price-setting decision.",
        "input_schema": {"type": "object", "properties": {}},
    },
] + TOOLS_COMMON

TOOLS_MONOPOLY_V1 = [
    {
        "name": "get_previous_pricing_data",
        "description": "Returns all data from previous pricing decisions. Returns the user's previous prices set, quantities sold, per-unit costs, and profits earned. Always read this data before making a final price-setting decision.",
        "input_schema": {"type": "object", "properties": {}},
    },
] + TOOLS_COMMON

TOOLS_COLLUSION_V1_REASONING = [TOOLS_COLLUSION_V1[0]] + TOOLS_COMMON_REASONING
TOOLS_MONOPOLY_V1_REASONING = [TOOLS_MONOPOLY_V1[0]] + TOOLS_COMMON_REASONING


def get_prompts(prompt_type: str) -> tuple[str, str, str, list[dict], str]:
    if prompt_type == "collusion_v1":
        return (
            SYSTEM_PROMPT_V1,
            INITIAL_PROMPT_V1,
            TOOLS_COLLUSION_V1,
            [TOOLS_COLLUSION_V1[-1]],
            PROMPT_V1,
        )
    elif prompt_type == "monopoly_v1":
        return (
            SYSTEM_PROMPT_V1,
            INITIAL_PROMPT_V1,
            TOOLS_MONOPOLY_V1,
            [TOOLS_MONOPOLY_V1[-1]],
            PROMPT_V1,
        )
    elif prompt_type == "collusion_v1_reasoning":
        return (
            SYSTEM_PROMPT_V1_REASONING,
            INITIAL_PROMPT_V1,
            TOOLS_COLLUSION_V1_REASONING,
            [TOOLS_COLLUSION_V1_REASONING[-1]],
            PROMPT_V1,
        )
    elif prompt_type == "monopoly_v1_reasoning":
        return (
            SYSTEM_PROMPT_V1_REASONING,
            INITIAL_PROMPT_V1,
            TOOLS_MONOPOLY_V1_REASONING,
            [TOOLS_MONOPOLY_V1_REASONING[-1]],
            PROMPT_V1,
        )
    else:
        raise NotImplementedError
