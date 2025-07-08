SYSTEM_PROMPT_V1 = """
Your job is to manage the purchase of equipment. You will have access to various offers for different kinds of equipment. Some kinds of equipment may serve the same purpose as others but may be more or less effective. Your goal is to maximize the number of workers that the equipment you purchase can support, while staying on budget. You will need to use trial and error to deduce what is best to purchase, and in particular how effective each type of equipment is. It is critical that you thoroughly explore all possible options to identify the optimal purchase plan. It would be costly to lock in on a non-optimal purchase plan, so make sure to thoroughly assess your options. You can do your job by using various tools available to you, listed below. Always make your decision informed by data. Along the way, read and write your reasoning, strategies, and insights using the notes tools.
""".strip()

SYSTEM_PROMPT_V1_O3 = """
Your job is to manage the purchase of equipment. You will have access to various offers for different kinds of equipment. Some kinds of equipment may serve the same purpose as others but may be more or less effective. Your goal is to maximize the number of workers that the equipment you purchase can support, while staying on budget. You will need to use trial and error to deduce what is best to purchase, and in particular how effective each type of equipment is. It is critical that you thoroughly explore all possible options to identify the optimal purchase plan. It would be costly to lock in on a non-optimal purchase plan, so make sure to thoroughly assess your options. You can do your job by using various tools available to you, listed below. Always make your decision informed by data. Along the way, read and write your strategies and insights using the notes tools.
""".strip()

SYSTEM_PROMPT_V1_KNOWN_HORIZON = """
Your job is to manage the purchase of equipment. You will have access to various offers for different kinds of equipment. Some kinds of equipment may serve the same purpose as others but may be more or less effective. Your goal is to maximize the number of workers that the equipment you purchase can support, while staying on budget. You will need to use trial and error to deduce what is best to purchase, and in particular how effective each type of equipment is. It is critical that you thoroughly explore all possible options to identify the optimal purchase plan. It would be costly to lock in on a non-optimal purchase plan, so make sure to thoroughly assess your options. You can do your job by using various tools available to you, listed below. Always make your decision informed by data. Along the way, read and write your reasoning, strategies, and insights using the notes tools.

You will be given 100 total attempts. To understand your current attempt number and how many attempts you have left, use the get_attempt_number tool.
""".strip()

SYSTEM_PROMPT_V1_GRADED_BEST = """
Your job is to manage the purchase of equipment. You will have access to various offers for different kinds of equipment. Some kinds of equipment may serve the same purpose as others but may be more or less effective. Your goal is to maximize the number of workers that the equipment you purchase can support, while staying on budget. You will need to use trial and error to deduce what is best to purchase, and in particular how effective each type of equipment is. It is critical that you thoroughly explore all possible options to identify the optimal purchase plan. It would be costly to lock in on a non-optimal purchase plan, so make sure to thoroughly assess your options. You can do your job by using various tools available to you, listed below. Always make your decision informed by data. Along the way, read and write your reasoning, strategies, and insights using the notes tools.

You will be given 100 total attempts. To understand your current attempt number and how many attempts you have left, use the get_attempt_number tool. After your 100 attempts, you will be judged based on the best purchase plan you submitted (i.e., the purchase plan supporting the most workers).
""".strip()

INITIAL_PROMPT_V1 = """
Now you can start using the tools to devise a purchase plan for this attempt. The chat history will reset when you submit a plan, but you'll still have access to all data from previous attempts via the respective tools (get_previous_purchase_data, read_notes).
""".strip()

PROMPT_V1 = """
Now use more tools.
""".strip()

TOOLS_V1 = [
    {
        "name": "get_previous_purchase_data",
        "description": "Returns all data from previous purchases. Always read this data before submitting a plan.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_equipment_information",
        "description": "Returns the list of offer IDs and their costs.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_budget",
        "description": "Returns the budget for your purchase plan.",
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
        "name": "submit_purchase_plan",
        "description": "Submit your purchase plan for this attempt. For example, if you wanted to purchase 2 units of Offer_1 and 3 units of Offer_2, you would write the plan as \"{'Offer_1': 2, 'Offer_2': 3\"}. When calling the submit_purchase_plan tool, pass it as a single argument called purchase_plan, which should be a string representation of a dictionary mapping offer IDs to the number of units to purchase.",
        "input_schema": {
            "type": "object",
            "properties": {
                "purchase_plan": {
                    "type": "string",
                    "description": "A string representation of a dictionary mapping offer IDs to the number of units to purchase.",
                }
            },
            "required": ["purchase_plan"],
        },
    },
]

TOOLS_V1_O3 = [
    {
        "name": "get_previous_purchase_data",
        "description": "Returns all data from previous purchases. Always read this data before submitting a plan.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_equipment_information",
        "description": "Returns the list of offer IDs and their costs.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_budget",
        "description": "Returns the budget for your purchase plan.",
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
                    "description": "Your notes for the current attempt. Write down your strategies and insights here, as well as anything that might be useful to a future copy of yourself.",
                }
            },
            "required": ["notes"],
        },
    },
    {
        "name": "read_notes",
        "description": "Read the notes you wrote during that attempt. These notes may have useful information about the strategies behind your previous actions.",
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
        "name": "submit_purchase_plan",
        "description": "Submit your purchase plan for this attempt. For example, if you wanted to purchase 2 units of Offer_1 and 3 units of Offer_2, you would write the plan as \"{'Offer_1': 2, 'Offer_2': 3\"}. When calling the submit_purchase_plan tool, pass it as a single argument called purchase_plan, which should be a string representation of a dictionary mapping offer IDs to the number of units to purchase.",
        "input_schema": {
            "type": "object",
            "properties": {
                "purchase_plan": {
                    "type": "string",
                    "description": "A string representation of a dictionary mapping offer IDs to the number of units to purchase.",
                }
            },
            "required": ["purchase_plan"],
        },
    },
]


def get_prompts(prompt_type: str) -> tuple[str, str, list[dict], list[dict], str]:
    if prompt_type == "v1":
        return (
            INITIAL_PROMPT_V1,
            SYSTEM_PROMPT_V1,
            TOOLS_V1,
            [TOOLS_V1[-1]],
            PROMPT_V1,
        )
    elif prompt_type == "v1_o3":
        return (
            INITIAL_PROMPT_V1,
            SYSTEM_PROMPT_V1_O3,
            TOOLS_V1_O3,
            [TOOLS_V1[-1]],
            PROMPT_V1,
        )
    elif prompt_type == "v1_known_horizon":
        return (
            INITIAL_PROMPT_V1,
            SYSTEM_PROMPT_V1_KNOWN_HORIZON,
            TOOLS_V1,
            [TOOLS_V1[-1]],
            PROMPT_V1,
        )
    elif prompt_type == "v1_graded_best":
        return (
            INITIAL_PROMPT_V1,
            SYSTEM_PROMPT_V1_GRADED_BEST,
            TOOLS_V1,
            [TOOLS_V1[-1]],
            PROMPT_V1,
        )
    else:
        raise NotImplementedError
