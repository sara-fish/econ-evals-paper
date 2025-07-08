SYSTEM_PROMPT_V1 = """
Your job is to assign workers to tasks, where each worker gets exactly one task. Your goal is to produce a valid assignment using trial and error: if your proposed assignment is not valid, you will be informed of its problem(s) and asked to submit another assignment. You can do your job by using various tools available to you, listed below. Always make your decision informed by data. Along the way, read and write your reasoning, strategies, and insights using the notes tools.
""".strip()

SYSTEM_PROMPT_V1_REASONING = """
Your job is to assign workers to tasks, where each worker gets exactly one task. Your goal is to produce a valid assignment using trial and error: if your proposed assignment is not valid, you will be informed of its problem(s) and asked to submit another assignment. You can do your job by using various tools available to you, listed below. Always make your decision informed by data. Along the way, read and write your strategies and insights using the notes tools.
""".strip()

INITIAL_PROMPT_V1 = """
Now you can start using the tools to devise an assignment. The chat history will reset when you submit an assignment, but you'll still have access to all data from previous attempts via the respective tools (get_previous_attempts_data, read_notes).
""".strip()

INITIAL_PROMPT_FINAL_ATTEMPT_V1 = """
Now you can start using the tools to devise an assignment. The chat history will reset when you submit an assignment, but you'll still have access to all data from previous attempts via the respective tools (get_previous_attempts_data, read_notes).

**This is your final attempt.** This time, you should submit the highest quality assignment possible, that has the fewest problems.
""".strip()


PROMPT_V1 = """
Now use more tools.
""".strip()

TOOLS_V1 = [
    {
        "name": "get_previous_attempts_data",
        "description": "Returns all data from previous assignments tried and why they didn't work. Always read this data before submitting an assignment.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_attempt_number",
        "description": "Returns the current attempt number, 0-indexed. (E.g., if you're on attempt #4, this returns 4, and you've made 4 previous attempts (#0, #1, #2, and #3).)",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_worker_ids",
        "description": "Returns the list of worker IDs to be assigned.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_task_ids",
        "description": "Returns the list of task IDs to be assigned.",
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
        "description": "Read the notes you wrote during that attempt number. These notes may have useful information about the reasoning and strategies behind that previous attempt.",
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
        "name": "submit_assignment",
        "description": "Submit an attempt at a valid assignment of workers to tasks. For example, if you had workers A,B,C and tasks 1,2,3, you would write the assignment as"
        + """ "{'A': '1', 'B': '2', 'C': '3'}". When calling the submit_assignment tool, pass it a single argument called assignment, which should be a string representation of a dictionary mapping worker IDs to task IDs.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "assignment": {
                    "type": "string",
                    "description": "A string representation of a dictionary mapping worker IDs to task IDs. The keys should consist of all worker IDs and the values should consist of all task IDs (each task assigned exactly once).",
                }
            },
            "required": ["assignment"],
        },
    },
]

TOOLS_V1_REASONING = [
    {
        "name": "get_previous_attempts_data",
        "description": "Returns all data from previous assignments tried and why they didn't work. Always read this data before submitting an assignment.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_attempt_number",
        "description": "Returns the current attempt number, 0-indexed. (E.g., if you're on attempt #4, this returns 4, and you've made 4 previous attempts (#0, #1, #2, and #3).)",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_worker_ids",
        "description": "Returns the list of worker IDs to be assigned.",
        "input_schema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_task_ids",
        "description": "Returns the list of task IDs to be assigned.",
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
        "description": "Read the notes you wrote during that attempt number. These notes may have useful information about the strategies behind that previous attempt.",
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
        "name": "submit_assignment",
        "description": "Submit an attempt at a valid assignment of workers to tasks. For example, if you had workers A,B,C and tasks 1,2,3, you would write the assignment as"
        + """ "{'A': '1', 'B': '2', 'C': '3'}". When calling the submit_assignment tool, pass it a single argument called assignment, which should be a string representation of a dictionary mapping worker IDs to task IDs.""",
        "input_schema": {
            "type": "object",
            "properties": {
                "assignment": {
                    "type": "string",
                    "description": "A string representation of a dictionary mapping worker IDs to task IDs. The keys should consist of all worker IDs and the values should consist of all task IDs (each task assigned exactly once).",
                }
            },
            "required": ["assignment"],
        },
    },
]

TOOLS_ACTION_ONLY = [TOOLS_V1[-1]]


def get_prompts(prompt_type: str) -> tuple[str, str, list[dict], list[dict], str]:
    if prompt_type == "v1":
        return (
            INITIAL_PROMPT_V1,
            SYSTEM_PROMPT_V1,
            TOOLS_V1,
            TOOLS_ACTION_ONLY,
            PROMPT_V1,
        )
    elif prompt_type == "v1_reasoning":
        return (
            INITIAL_PROMPT_V1,
            SYSTEM_PROMPT_V1_REASONING,
            TOOLS_V1_REASONING,
            TOOLS_ACTION_ONLY,
            PROMPT_V1,
        )
    elif prompt_type == "final_attempt_v1":
        return (
            INITIAL_PROMPT_FINAL_ATTEMPT_V1,
            SYSTEM_PROMPT_V1,
            TOOLS_V1,
            TOOLS_ACTION_ONLY,
            PROMPT_V1,
        )
    elif prompt_type == "final_attempt_v1_reasoning":
        return (
            INITIAL_PROMPT_FINAL_ATTEMPT_V1,
            SYSTEM_PROMPT_V1_REASONING,
            TOOLS_V1_REASONING,
            TOOLS_ACTION_ONLY,
            PROMPT_V1,
        )
    else:
        raise NotImplementedError
