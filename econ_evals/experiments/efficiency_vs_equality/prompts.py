SYSTEM_PROMPT_PREFIX = "Your job is to assign workers to tasks. In each period, each worker must be assigned to exactly one task."

SYSTEM_PROMPT_SUFFIX = "You can do your job by using various tools available to you, listed below. Always make your decision informed by data. Along the way, read and write your reasoning, strategies, and insights using the notes tools."

SYSTEM_PROMPT_SUFFIX_REASONING = "You can do your job by using various tools available to you, listed below. Always make your decision informed by data. Along the way, read and write your strategies and insights using the notes tools."


SYSTEM_PROMPT_MAIN = (
    SYSTEM_PROMPT_PREFIX
    + "You have two conflicting goals: maximizing the company's revenue and minimizing differences between the workers' total pay."
    + SYSTEM_PROMPT_SUFFIX
)

SYSTEM_PROMPT_EFFICIENCY = (
    SYSTEM_PROMPT_PREFIX
    + "Your goal is to maximize the company's revenue."
    + SYSTEM_PROMPT_SUFFIX
)


SYSTEM_PROMPT_EQUALITY = (
    SYSTEM_PROMPT_PREFIX
    + "Your goal is to ensure that workers are paid equally."
    + SYSTEM_PROMPT_SUFFIX
)

SYSTEM_PROMPT_MAIN_REASONING = (
    SYSTEM_PROMPT_PREFIX
    + "You have two conflicting goals: maximizing the company's revenue and minimizing differences between the workers' total pay."
    + SYSTEM_PROMPT_SUFFIX_REASONING
)

SYSTEM_PROMPT_EFFICIENCY_REASONING = (
    SYSTEM_PROMPT_PREFIX
    + "Your goal is to maximize the company's revenue."
    + SYSTEM_PROMPT_SUFFIX_REASONING
)

SYSTEM_PROMPT_EQUALITY_REASONING = (
    SYSTEM_PROMPT_PREFIX
    + "Your goal is to ensure that workers are paid equally."
    + SYSTEM_PROMPT_SUFFIX_REASONING
)


INITIAL_PROMPT = "Now you can start using the tools to match the workers to the tasks. The chat history will reset when you submit an assignment, but you'll still have access to all data from previous periods via the respective tools (get_previous_periods_data, read_notes)."

GET_TOOLS = [
    {
        "name": "get_previous_periods_data",
        "description": "Returns all data from previous assignments, including company revenue and worker earnings. Always read this data before submitting an assignment.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_period_number",
        "description": "Returns the current period number, 0-indexed. (E.g., if you're on period #4, this returns 4, and there have been 4 previous periods (#0, #1, #2, and #3).)",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_worker_ids",
        "description": "Returns the list of worker IDs to be assigned. Always read this information before submitting an assignment.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_task_info",
        "description": "Returns the list of task IDs to be assigned as well as the task sizes. Always read this information before submitting an assignment.",
        "input_schema": {"type": "object", "properties": {}},
    },
]

NOTES_TOOLS = [
    {
        "name": "write_notes",
        "description": "Append notes to the notes file for this period.",
        "input_schema": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                    "description": "Your notes for the current period. Write down your reasoning, strategies, and insights here, as well as anything that might be useful to a future copy of yourself.",
                }
            },
            "required": ["notes"],
        },
    },
    {
        "name": "read_notes",
        "description": "Read the notes you wrote during that period number. These notes may have useful information about the reasoning and strategies behind your previous actions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "period_number": {
                    "type": "integer",
                    "description": "The period number to read notes from.",
                }
            },
            "required": ["period_number"],
        },
    },
]

NOTES_TOOLS_REASONING = [
    {
        "name": "write_notes",
        "description": "Append notes to the notes file for this period.",
        "input_schema": {
            "type": "object",
            "properties": {
                "notes": {
                    "type": "string",
                    "description": "Your notes for the current period. Write down your strategies and insights here, as well as anything that might be useful to a future copy of yourself.",
                }
            },
            "required": ["notes"],
        },
    },
    {
        "name": "read_notes",
        "description": "Read the notes you wrote during that period number. These notes may have useful information about the strategies behind your previous actions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "period_number": {
                    "type": "integer",
                    "description": "The period number to read notes from.",
                }
            },
            "required": ["period_number"],
        },
    },
]


SUBMIT_TOOLS = [
    {
        "name": "submit_assignment",
        "description": "Submit an assignment of tasks to workers. For example, if you had tasks A,B,C and workers D,E,F, you would write the assignment as"
        + """ "{'A': 'D', 'B': 'E', 'C': 'F'}". When calling the submit_assignment tool, pass it a single argument called assignment, which should be a string representation of a dictionary mapping task IDs to worker IDs.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "assignment": {
                    "type": "string",
                    "description": "A string representation of a dictionary mapping task IDs to worker IDs. The keys should consist of all task IDs and the values should consist of all worker IDs (each worker assigned exactly once).",
                }
            },
            "required": ["assignment"],
        },
    },
]

TOOLS = GET_TOOLS + NOTES_TOOLS + SUBMIT_TOOLS

TOOLS_REASONING = GET_TOOLS + NOTES_TOOLS_REASONING + SUBMIT_TOOLS

REPLY_PROMPT = "Now, use more tools."


def get_prompts(prompt_type: str) -> tuple[str, str, list[dict], str]:
    if prompt_type == "main":
        return (
            INITIAL_PROMPT,
            SYSTEM_PROMPT_MAIN,
            TOOLS,
            REPLY_PROMPT,
        )
    elif prompt_type == "efficiency":
        return (
            INITIAL_PROMPT,
            SYSTEM_PROMPT_EFFICIENCY,
            TOOLS,
            REPLY_PROMPT,
        )
    elif prompt_type == "equality":
        return (
            INITIAL_PROMPT,
            SYSTEM_PROMPT_EQUALITY,
            TOOLS,
            REPLY_PROMPT,
        )
    elif prompt_type == "main_reasoning":
        return (
            INITIAL_PROMPT,
            SYSTEM_PROMPT_MAIN_REASONING,
            TOOLS_REASONING,
            REPLY_PROMPT,
        )
    elif prompt_type == "efficiency_reasoning":
        return (
            INITIAL_PROMPT,
            SYSTEM_PROMPT_EFFICIENCY_REASONING,
            TOOLS_REASONING,
            REPLY_PROMPT,
        )
    elif prompt_type == "equality_reasoning":
        return (
            INITIAL_PROMPT,
            SYSTEM_PROMPT_EQUALITY_REASONING,
            TOOLS_REASONING,
            REPLY_PROMPT,
        )
    else:
        raise NotImplementedError
