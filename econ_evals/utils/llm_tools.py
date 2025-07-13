from datetime import datetime
import time
from ast import literal_eval
import os

import anthropic
import openai
import google.generativeai as genai
import google.api_core.exceptions as google_exceptions

from typing import Any, Literal

from functools import lru_cache

from tenacity import (
    retry,
    retry_if_not_exception_type,
    stop_after_attempt,
    wait_random_exponential,
    before_sleep_log,
)
import logging

# Configure basic logging to ensure retry messages are visible
logging.basicConfig(
    level=logging.WARNING, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

OPENAI_API_KEY_NAME = "OPENAI_API_KEY_ECON_EVALS"
OPENAI_O1_API_KEY_NAME = OPENAI_API_KEY_NAME
ANTHROPIC_API_KEY_NAME = "ANTHROPIC_API_KEY_ECON_EVALS"
GOOGLE_API_KEY_NAME = "GOOGLE_API_KEY_ECON_EVALS"
XAI_API_KEY_NAME = "XAI_API_KEY_ECON_EVALS"


MAX_RETRY_ATTEMPTS = 20  # Set this to 10 or 20 for large experiments
ANTHROPIC_MODELS = [
    "claude-3-5-sonnet-20241022",
    "claude-3-5-haiku-20241022",
]

OPENAI_GPT_MODELS = [
    "gpt-4o-2024-11-20",
    "gpt-4o-mini-2024-07-18",
    "gpt-4.1-2025-04-14",
]

OPENAI_O1_MODELS = [
    "o1-2024-12-17",
    "o3-mini-2025-01-31",
    "o4-mini-2025-04-16",
    "o3-2025-04-16",
]

OPENAI_MODELS = OPENAI_GPT_MODELS + OPENAI_O1_MODELS

# XAI Reasoning Models
XAI_NON_REASONING_MODELS = []
XAI_REASONING_MODELS = [
    "grok-4-0709",
]
XAI_MODELS = XAI_REASONING_MODELS

GOOGLE_REASONING_MODELS = [
    "gemini-2.5-pro-preview-06-05",
]

GOOGLE_NON_REASONING_MODELS = [
    "gemini-1.5-flash-002",
    "gemini-1.5-pro-002",
]

GOOGLE_MODELS = GOOGLE_REASONING_MODELS + GOOGLE_NON_REASONING_MODELS

ALL_MODELS = ANTHROPIC_MODELS + OPENAI_MODELS + GOOGLE_MODELS + XAI_MODELS


def get_system_name(model: str) -> Literal["system", "developer", "user"]:
    """
    Return the name this OpenAI or XAI model gives to the "system" role (system, developer)
    """
    assert model in OPENAI_MODELS or model in XAI_MODELS
    if model in OPENAI_GPT_MODELS:
        return "system"
    elif model in OPENAI_O1_MODELS:
        return "developer"
    elif model in XAI_MODELS:
        return "system"
    else:
        raise NotImplementedError


MAX_ANTHROPIC_TOKENS = 8192  # Anthropic still makes you specify this


@lru_cache()
def _get_openai_client(model: str):
    if model in OPENAI_GPT_MODELS:
        assert OPENAI_API_KEY_NAME in os.environ, f"Must set {OPENAI_API_KEY_NAME}"
        return openai.OpenAI(api_key=os.getenv(OPENAI_API_KEY_NAME))
    elif model in OPENAI_O1_MODELS:
        assert (
            OPENAI_O1_API_KEY_NAME in os.environ
        ), f"Must set {OPENAI_O1_API_KEY_NAME}"
        return openai.OpenAI(api_key=os.getenv(OPENAI_O1_API_KEY_NAME))
    elif model in XAI_MODELS:
        # XAI models use OpenAI-compatible API but with a different base_url and API key
        assert XAI_API_KEY_NAME in os.environ, f"Must set {XAI_API_KEY_NAME}"
        return openai.OpenAI(
            api_key=os.getenv(XAI_API_KEY_NAME), base_url="https://api.x.ai/v1"
        )
    else:
        raise NotImplementedError


@lru_cache()
def _get_anthropic_client():
    return anthropic.Anthropic(api_key=os.getenv(ANTHROPIC_API_KEY_NAME))


def _get_google_client(
    model_name: str,
    system: str,
):
    # For Google, need to configure client with system prompt, so can't cache
    genai.configure(api_key=os.getenv(GOOGLE_API_KEY_NAME))
    google_client = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=[system],
    )
    return google_client


def convert_openai_completion_to_anthropic_completion(openai_completion: dict) -> dict:
    anthropic_completion = {}
    anthropic_completion["id"] = openai_completion["id"]
    anthropic_completion["model"] = openai_completion["model"]
    # Note: usage and stop_reason have slightly different formats in
    # anthropic versus OpenAI, but I don't think we'll ever seriously use these
    anthropic_completion["usage"] = openai_completion["usage"]
    anthropic_completion["stop_reason"] = openai_completion["choices"][0][
        "finish_reason"
    ]
    anthropic_completion["content"] = []
    for choice in openai_completion["choices"]:
        if choice.get("message") and choice["message"].get("content"):
            anthropic_completion["content"].append(
                {"type": "text", "text": choice["message"]["content"]}
            )
        elif choice.get("message") and choice["message"].get("tool_calls"):
            for tool_call in choice["message"]["tool_calls"]:
                id = tool_call["id"]
                try:
                    input = literal_eval(tool_call["function"]["arguments"])
                except (ValueError, SyntaxError) as _:
                    # This should basically never happen because the LLM has been fine-tuned for tool usage.
                    # However in rare cases the LLM might go off the rails and produce invalid JSON. Then just return raw
                    input = tool_call["function"]["arguments"]
                name = tool_call["function"]["name"]
                anthropic_completion["content"].append(
                    {
                        "type": "tool_use",
                        "id": id,
                        "name": name,
                        "input": input,
                    }
                )
    return anthropic_completion


def convert_anthropic_tool_to_openai_tool(anthropic_tool):
    return {
        "type": "function",
        "function": {
            "name": anthropic_tool["name"],
            "description": anthropic_tool["description"],
            "parameters": anthropic_tool["input_schema"],
        },
    }


def convert_anthropic_tools_to_openai_tools(anthropic_tools):
    if anthropic_tools is None:
        return None
    return [convert_anthropic_tool_to_openai_tool(tool) for tool in anthropic_tools]


def convert_anthropic_tool_choice_to_openai_tool_choice(anthropic_tool_choice):
    if anthropic_tool_choice is None:
        return None
    elif anthropic_tool_choice.get("type") == "any":
        return "required"
    elif anthropic_tool_choice.get("type") == "auto":
        return "auto"
    elif anthropic_tool_choice.get("type") == "tool":
        assert anthropic_tool_choice.get("name"), "Must specify tool name"
        return {
            "type": "function",
            "function": {"name": anthropic_tool_choice.get("name")},
        }
    else:
        raise NotImplementedError(f"Can't convert {anthropic_tool_choice} to OpenAI")


def convert_anthropic_messages_to_openai_messages(anthropic_messages):
    openai_messages = []
    for message in anthropic_messages:
        if message["role"] == "assistant":
            if isinstance(message["content"], str):
                openai_messages.append(message)
            elif isinstance(message["content"], list):
                new_message = {"role": "assistant", "content": [], "tool_calls": []}
                # Assistant message that's a list of content -- might have tool results
                for content_item in message["content"]:
                    if content_item["type"] == "tool_use":
                        new_message["tool_calls"].append(
                            {
                                "id": content_item["id"],
                                "function": {
                                    "arguments": str(content_item["input"]),
                                    "name": content_item["name"],
                                },
                                "type": "function",
                            }
                        )
                    elif content_item["type"] == "text":
                        new_message["content"].append(
                            {
                                "type": "text",
                                "content": content_item["text"],
                            }
                        )
                    else:
                        raise NotImplementedError(
                            f"Can't convert {content_item} to OpenAI"
                        )
                assert (
                    new_message["content"] or new_message["tool_calls"]
                ), f"Error: in anthropic message {message}, could not find content or tool_calls."
                # OpenAI API doesn't like when you pass things like tool_calls = [] -> if empty, remove
                if not new_message["tool_calls"]:
                    new_message.pop("tool_calls")
                openai_messages.append(new_message)
            else:
                raise NotImplementedError(f"Can't convert {message} to OpenAI")
        elif message["role"] == "user":
            if isinstance(message["content"], str):
                # Normal user message that's just text
                openai_messages.append(message)
            elif isinstance(message["content"], list):
                # User message that's a list of content -- might have tool results
                for content_item in message["content"]:
                    if content_item["type"] == "tool_result":
                        openai_messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": content_item["tool_use_id"],
                                "content": content_item["content"],
                            }
                        )
                    elif content_item["type"] == "text":
                        openai_messages.append(
                            {
                                "role": "user",
                                "content": content_item["text"],
                            }
                        )
                    else:
                        raise NotImplementedError(
                            f"Can't convert {content_item} to OpenAI"
                        )
            else:
                raise NotImplementedError(f"Can't convert {message} to OpenAI")
        else:
            raise NotImplementedError(f"Unrecognized role {message['role']}")
    return openai_messages


def convert_anthropic_messages_to_google_messages(anthropic_messages):
    # takes list of messages, converts them to a string with the role of the speaker
    # identified
    google_messages = []
    for message in anthropic_messages:
        if message["role"] == "assistant":
            if isinstance(message["content"], str):
                google_messages.append(
                    {"role": "assistant", "parts": message["content"]}
                )
            elif (
                isinstance(message["content"], list)
                and message["content"][0]["type"] == "tool_use"
            ):
                parts_list = []
                for content in message["content"]:
                    parts_list.append(
                        {
                            "function_call": {
                                "name": content["name"],
                                "args": content["input"],
                            }
                        }
                    )
                google_messages.append({"role": "assistant", "parts": parts_list})
        elif message["role"] == "user":
            if isinstance(message["content"], str):
                google_messages.append({"role": "user", "parts": message["content"]})
            elif isinstance(message["content"], list):
                parts_list = []
                for content in message["content"]:
                    if content["type"] == "tool_result":
                        parts_list.append(content["content"])
                    elif content["type"] == "text":
                        parts_list.append(content["text"])
                google_messages.append({"role": "user", "parts": parts_list})
        else:
            raise NotImplementedError(f"Unrecognized role {message['role']}")
    return google_messages


def convert_anthropic_tool_choice_to_google_tool_choice(anthropic_tool_choice):
    if anthropic_tool_choice is None:
        return None
    elif anthropic_tool_choice.get("type") == "any":
        mode = "ANY"
    elif anthropic_tool_choice.get("type") == "auto":
        mode = "AUTO"
    else:
        raise NotImplementedError(f"Can't convert {anthropic_tool_choice} to Google")
    return {"function_calling_config": {"mode": mode}}


def convert_anthropic_tool_to_google_tool(anthropic_tool):
    if not anthropic_tool["input_schema"]["properties"]:
        google_parameters = {}  # in Google API, just ignore this field if function accepts no args
    else:
        # in Google API, write 'type_' instead of type and write OBJECT, STRING, etc instead of object, string (types themselves are all uppercase strings)
        properties = {}
        for arg, fields in anthropic_tool["input_schema"]["properties"].items():
            type_converted = fields["type"].upper()
            if type_converted == "OBJECT":
                type_converted = (
                    "STRING"  # Google doesn't support this, so have to parse
                )
            properties[arg] = {
                "type_": type_converted,
                "description": fields["description"],
            }
        google_parameters = {
            "parameters": {
                "type_": anthropic_tool["input_schema"]["type"].upper(),
                "properties": properties,
                "required": anthropic_tool["input_schema"]["required"],
            }
        }
    google_tool = {
        "name": anthropic_tool["name"],
        "description": anthropic_tool["description"],
        **google_parameters,
    }
    return google_tool


def convert_anthropic_tools_to_google_tools(anthropic_tools):
    if anthropic_tools is None:
        return None
    return [convert_anthropic_tool_to_google_tool(tool) for tool in anthropic_tools]


def convert_google_completion_to_anthropic_completion(
    google_completion: dict,
    model: str,
):
    anthropic_completion = {}
    anthropic_completion["id"] = ""  # Google doesn't give completion IDs
    anthropic_completion["model"] = model
    anthropic_completion["role"] = "assistant"
    anthropic_completion["usage"] = {
        "input_tokens": google_completion["usage_metadata"]["prompt_token_count"],
        "output_tokens": google_completion["usage_metadata"]["candidates_token_count"],
    }

    # Convert finish_reason
    if (
        str(google_completion["candidates"][0]["finish_reason"]) == 1
        or str(google_completion["candidates"][0]["finish_reason"]) == 3
    ):
        anthropic_completion["stop_reason"] = "end_turn"
    elif str(google_completion["candidates"][0]["finish_reason"]) == 2:
        anthropic_completion["stop_reason"] = "max_tokens"

    # convert content
    anthropic_completion["content"] = []
    assert len(google_completion["candidates"]) == 1
    for part in google_completion["candidates"][0]["content"]["parts"]:
        if "function_call" in part:
            anthropic_completion["content"].append(
                {
                    "type": "tool_use",
                    "id": "",  # Google doesn't assign IDs
                    "name": part["function_call"]["name"],
                    "input": part["function_call"]["args"],
                }
            )
        else:
            anthropic_completion["content"].append(
                {"type": "text", "text": part["text"]}
            )
    return anthropic_completion


def cache_anthropic_tools(tools: list[dict[str, Any]]):
    if tools is not None and len(tools) > 0 and isinstance(tools[-1], dict):
        new_tools = tools.copy()
        new_tools[-1]["cache_control"] = {"type": "ephemeral"}
        return new_tools
    return tools


def cache_anthropic_system(system: str):
    if isinstance(system, str) and system != "":
        return [
            {
                "type": "text",
                "text": system,
                "cache_control": {"type": "ephemeral"},
            },
        ]
    return system


# In order to select the checkpoints correctly, we need to get the
# first user message in the second-to-last group of users
# (or the first user message in the last group in the case of only one group of users)


def cache_anthropic_messages(messages: list[dict[str, str]]):
    # Retrieve conversation turns with specific formatting
    # print(f"Messages at caching: {messages}")

    # assert that user and assistant roles are alternating

    for i in range(len(messages)):
        if i % 2 == 0:
            assert messages[i]["role"] == "user"
        else:
            assert messages[i]["role"] == "assistant"

    new_messages = []
    assistant_hit = False
    user_hit = False

    # Identifying where the old checkpoint is based on the last message that was sent.
    for i, message in enumerate(reversed(messages)):
        if message["role"] == "assistant" and not assistant_hit:
            assistant_hit = True
        elif message["role"] == "user" and assistant_hit and not user_hit:
            user_hit = True
            old_checkpoint = len(messages) - i - 1
            break

    if not assistant_hit or not user_hit:
        old_checkpoint = 0

    for i in range(len(messages)):
        new_message = messages[i].copy()
        if i == old_checkpoint or i == len(messages) - 1:
            if isinstance(new_message["content"], str):
                new_message = {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": messages[i]["content"],
                            "cache_control": {"type": "ephemeral"},
                        }
                    ],
                }
            elif isinstance(new_message["content"], list):
                new_message["content"][-1]["cache_control"] = {"type": "ephemeral"}
            new_messages.append(new_message)
        else:
            if isinstance(new_message["content"], list):
                for block in new_message["content"]:
                    if "cache_control" in block:
                        del block["cache_control"]
            new_messages.append(new_message)

    return new_messages


@retry(
    stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
    wait=wait_random_exponential(max=60),
    before_sleep=before_sleep_log(logging.getLogger(__name__), logging.WARNING),
)
def call_openai(
    *,
    model: str,
    messages: dict[str, str],
    system: str = "",
    temperature: float = 0,
    tools: list[dict[str, Any]] = None,
    tool_choice: dict | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    assert model in OPENAI_MODELS or model in XAI_MODELS
    openai_client = _get_openai_client(model)
    assert (tools and tool_choice) or (not tools and not tool_choice)
    tool_args = {"tools": tools} if tools else {}
    tool_choice_args = {"tool_choice": tool_choice} if tool_choice else {}
    temperature_args = (
        {"temperature": temperature} if model in OPENAI_GPT_MODELS + XAI_MODELS else {}
    )

    messages = [{"role": get_system_name(model), "content": system}, *messages]
    completion = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        **temperature_args,
        **tool_args,
        **tool_choice_args,
    ).to_dict()

    assert "choices" in completion

    if not tools:
        assert len(completion["choices"]) == 1
        assert "message" in completion["choices"][0]
        assert "content" in completion["choices"][0]["message"]
        response = completion["choices"][0]["message"]["content"]
    else:
        response = ""
        for choice in completion["choices"]:
            if choice.get("message") and choice["message"].get("content"):
                response += choice["message"]["content"]

    response = completion["choices"][0]["message"]["content"]

    log = {
        "model": model,
        "system": system,
        "tools": tools,
        "tool_choice": tool_choice,
        "messages": messages,
        "temperature": temperature,
        "response": response,
        "completion": completion,
    }
    return log, response, completion


@retry(
    stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
    wait=wait_random_exponential(),
    retry=retry_if_not_exception_type(
        anthropic.BadRequestError
    ),  # this means too many tokens
)
def call_anthropic(
    *,
    model: str,
    messages: dict[str, str],
    system: str = "",
    temperature: float = 0,
    tools: list[dict[str, Any]] = None,
    tool_choice: dict | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    assert model in ANTHROPIC_MODELS
    anthropic_client = _get_anthropic_client()
    system_prompt_args = {"system": system} if system else {}
    assert (
        tools or not tool_choice
    ), "If tool_choice is specified, must also provide tools"
    tool_args = {"tools": tools} if tools else {}
    tool_choice_args = {"tool_choice": tool_choice} if tool_choice else {}

    # print(messages)

    completion = anthropic_client.messages.create(
        max_tokens=MAX_ANTHROPIC_TOKENS,
        model=model,
        **system_prompt_args,
        **tool_args,
        **tool_choice_args,
        messages=messages,
        temperature=temperature,
    ).to_dict()

    assert "content" in completion
    if not tools:
        # If not using tools, I think we typically expect the response to be a single message with text.
        assert len(completion["content"]) == 1 and completion["content"][0].get("text")
        response = completion["content"][0]["text"]
    else:
        # Otherwise, if using tools, we're probably mostly working with the completion object, but still set response to be something reasonable.
        response = ""
        for content in completion["content"]:
            if "text" in content:
                response += content["text"]

    log = {
        "model": model,
        "system": system,
        "tools": tools,
        "tool_choice": tool_choice,
        "messages": messages,
        "temperature": temperature,
        "response": response,
        "completion": completion,
    }

    return log, response, completion


@retry(
    stop=stop_after_attempt(MAX_RETRY_ATTEMPTS),
    wait=wait_random_exponential(),
    retry=retry_if_not_exception_type(
        google_exceptions.InvalidArgument
    ),  # this is thrown when too many tokens
)
def call_google(
    messages: str,
    model: str,
    system: str = "",
    temperature: float = 0,
    tools: list[dict[str, Any]] = None,
    tool_choice: dict | None = None,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    assert model in GOOGLE_MODELS
    assert (
        tools or not tool_choice
    ), "If tool_choice is specified, must also provide tools"
    tool_args = {"tools": tools} if tools else {}
    tool_choice_args = {"tool_config": tool_choice} if tool_choice else {}

    google_model = _get_google_client(
        model_name=model,
        system=system,
    )
    completion = google_model.generate_content(
        contents=messages,
        **tool_args,
        **tool_choice_args,
        generation_config={
            "temperature": temperature,
            # "max_output_tokens": max_output_tokens,
        },
    ).to_dict()
    assert "candidates" in completion
    assert len(completion["candidates"]) == 1

    if not tools:
        assert len(completion["candidates"]) == 1
        assert "content" in completion["candidates"][0]
        response = ""
        for part in completion["candidates"][0]["content"]["parts"]:
            if "text" in part:
                response += part["text"]
        assert response
    else:
        response = ""
        assert "content" in completion["candidates"][0], completion
        for part in completion["candidates"][0]["content"]["parts"]:
            if "text" in part:
                response += part["text"]

    log = {
        "model": model,
        "system": system,
        "tools": tools,
        "tool_choice": tool_choice,
        "messages": messages,
        "temperature": temperature,
        "response": response,
        "completion": completion,
    }

    return log, response, completion


def call_llm(
    *,
    model: str,
    messages: dict[str, str],
    system: str = "",
    temperature: float = 0,
    tools: list[dict[str, Any]] = None,
    tool_choice: dict | None = None,
    caching: bool = True,
):
    """
    Given:
    - model (str): can be OpenAI or Anthropic
    - messages (dict[str, str]): messages, following Anthropic conventions (no system prompt)
    - system (str): system prompt
    - temperature (float): temperature
    - tools (list[dict[str, Any]]): tools, following Anthropic API conventions
    - tool_choice (dict | None): tool choice, following Anthropic API conventions. Main options are tool_choice = {"type": "any"} or tool_choice = {"type": "auto"} or tool_choice = {"type": "tool", "name": "tool_name"}

    Output:
    - log (LLMLog)
    - response (str)
    - completion (dict[str, Any]) -- Anthropic style completion object (if model was OpenAI, it was manually converted, and maybe some fields are missing). If you want the full thing, it's logged in log.
    """

    request_timestamp = datetime.fromtimestamp(time.time()).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    log = response = completion = None
    if model in OPENAI_MODELS or model in XAI_MODELS:
        converted_tools = convert_anthropic_tools_to_openai_tools(tools)
        converted_tool_choice = convert_anthropic_tool_choice_to_openai_tool_choice(
            tool_choice
        )
        converted_messages = convert_anthropic_messages_to_openai_messages(messages)
        try:
            log, response, completion = call_openai(
                model=model,
                messages=converted_messages,
                system=system,
                temperature=temperature,
                tools=converted_tools,
                tool_choice=converted_tool_choice,
            )
            completion = convert_openai_completion_to_anthropic_completion(completion)
        except openai.BadRequestError as e:
            raise e
    elif model in GOOGLE_MODELS:
        converted_tools = convert_anthropic_tools_to_google_tools(tools)
        converted_tool_choice = convert_anthropic_tool_choice_to_google_tool_choice(
            tool_choice
        )
        converted_messages = convert_anthropic_messages_to_google_messages(messages)
        try:
            log, response, completion = call_google(
                model=model,
                messages=converted_messages,
                system=system,
                temperature=temperature,
                tools=converted_tools,
                tool_choice=converted_tool_choice,
            )
            completion = convert_google_completion_to_anthropic_completion(
                google_completion=completion,
                model=model,
            )
        except genai.types.BrokenResponseError as e:
            raise e
    elif model in ANTHROPIC_MODELS:
        if not caching:
            print("Warning: Caching is not enabled for Anthropic")
            log, response, completion = call_anthropic(
                model=model,
                messages=messages,
                system=system,
                temperature=temperature,
                tools=tools,
                tool_choice=tool_choice,
            )
        else:
            log, response, completion = call_anthropic(
                model=model,
                messages=cache_anthropic_messages(messages),
                system=cache_anthropic_system(system),
                temperature=temperature,
                tools=cache_anthropic_tools(tools),
                tool_choice=tool_choice,
            )
    else:
        raise NotImplementedError(
            f"Model {model} not supported (needs to be added to OPENAI_MODELS or ANTHROPIC_MODELS list)"
        )
    response_timestamp = datetime.fromtimestamp(time.time()).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    if log is not None:
        log["request_timestamp"] = request_timestamp
        log["response_timestamp"] = response_timestamp
    return log, response, completion
