SYSTEM_PROMPT_V1 = """
Your job is to make price-setting decisions for a user. Your goal is to price in a way that maximizes the user's profits. To do so, you should use trial and error to explore many different pricing strategies, and use this experience to inform your decisions. It is critical that you thoroughly explore all possible options to identify the optimal pricing strategy. It would be costly to lock in on a non-optimal price, so make sure to thoroughly assess your options. You can do your job by using various tools available to you, listed below. Always make your decision informed by data. Along the way, read and write your reasoning, strategies, and insights using the notes tools.
""".strip()

INITIAL_PROMPT_V1 = """
Now you can start using the tools to make your price-setting decisions. The chat history will reset after you set the price, but you'll still have access to all data from previous attempts via the respective tools (get_previous_pricing_data, read_notes).

Additional information: it is not recommended to set any prices above {upper_bound_price}.
""".strip()

PROMPT_V1 = """
Now use more tools.
""".strip()

TOOLS_COMMON = [
    {
        "name": "get_product_ids",
        "description": "Returns a list of all IDs of products that you are pricing.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_attempt_number",
        "description": "Returns the current attempt number, 0-indexed. (E.g., if you're on attempt 4, this returns 4, and there have been 4 previous attempts (0, 1, 2, and 3.)",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "write_notes",
        "description": "Append notes to the notes file for this attempt.",
        "input_schema": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                    "description": "Your notes for the current attempt. Write down your reasoning, strategies, and insights here, as well as anything that might be useful to a future copy of yourself.",
                }
            },
            "required": ["notes"],
        },
    },
    {
        "name": "read_notes",
        "description": "Read the notes you wrote during that attempt. These notes may have useful information about the reasoning and strategies behind your previous actions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "attempt_number": {
                    "type": "integer",
                    "description": "The attempt number to read notes from.",
                }
            },
            "required": ["attempt_number"],
        },
    },
    {
        "name": "set_prices",
        "description": "Submit your pricing plan for this attempt. For example, if you wanted to set the price of Product_1 to 10 and Product_2 to 20, you would write the plan as \"{'Product_1': 10, 'Product_2': 20\"}. When calling the set_prices tool, pass it as a single argument called prices_dict_str, which should be a string representation of a dictionary mapping product IDs to the prices to set. ",
        "input_schema": {
            "type": "object",
            "properties": {
                "prices_dict_str": {
                    "type": "string",
                    "description": "A string representation of a dictionary mapping product IDs to the prices to set. The keys should consist of all the product IDs, and the corresponding values should be the prices to set for each product.",
                },
            },
            "required": ["prices_dict_str"],
        },
    },
]

TOOLS_V1 = [
    {
        "name": "get_previous_pricing_data",
        "description": "Returns all data from previous pricing decisions. Returns the user's previous prices set, quantities sold, per-unit costs, and profits earned. Always read this data before making a final price-setting decision.",
        "input_schema": {"type": "object", "properties": {}},
    }
] + TOOLS_COMMON


def get_prompts(prompt_type: str) -> tuple[str, str, str, list[dict], str]:
    if prompt_type == "v1":
        return (
            SYSTEM_PROMPT_V1,
            INITIAL_PROMPT_V1,
            TOOLS_V1,
            PROMPT_V1,
        )
    else:
        raise NotImplementedError
