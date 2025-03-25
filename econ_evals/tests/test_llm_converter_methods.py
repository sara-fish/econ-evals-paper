import unittest


from econ_evals.utils.llm_tools import (
    convert_anthropic_tool_choice_to_openai_tool_choice,
    convert_anthropic_tool_to_openai_tool,
    convert_openai_completion_to_anthropic_completion,
    convert_anthropic_tool_choice_to_google_tool_choice,
    convert_anthropic_tool_to_google_tool,
    convert_google_completion_to_anthropic_completion,
)


class TestLLMConverterMethods(unittest.TestCase):
    def test_tool_choice_conversion_anthropic_to_openai(self):
        anthropic_tool_choice1 = {"type": "auto"}
        anthropic_tool_choice2 = {"type": "any"}
        anthropic_tool_choice3 = {"type": "tool", "name": "test_tool"}
        openai_tool_choice1 = convert_anthropic_tool_choice_to_openai_tool_choice(
            anthropic_tool_choice1
        )
        openai_tool_choice2 = convert_anthropic_tool_choice_to_openai_tool_choice(
            anthropic_tool_choice2
        )
        openai_tool_choice3 = convert_anthropic_tool_choice_to_openai_tool_choice(
            anthropic_tool_choice3
        )
        self.assertTrue(openai_tool_choice1 == "auto")
        self.assertTrue(openai_tool_choice2 == "required")
        self.assertEqual(openai_tool_choice3.get("type"), "function")
        self.assertEqual(openai_tool_choice3.get("function").get("name"), "test_tool")

    def test_tool_conversion_anthropic_to_openai(self):
        anthropic_tool = {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature, either 'celsius' or 'fahrenheit'",
                    },
                },
                "required": ["location"],
            },
        }
        openai_tool = convert_anthropic_tool_to_openai_tool(anthropic_tool)
        # print(openai_tool)
        self.assertEqual(openai_tool.get("type"), "function")
        self.assertEqual(openai_tool.get("function").get("name"), "get_weather")
        self.assertEqual(
            openai_tool.get("function").get("description"),
            "Get the current weather in a given location",
        )
        self.assertEqual(
            openai_tool.get("function").get("parameters").get("type"), "object"
        )
        self.assertEqual(
            openai_tool.get("function")
            .get("parameters")
            .get("properties")
            .get("location")
            .get("type"),
            "string",
        )
        self.assertEqual(
            openai_tool.get("function")
            .get("parameters")
            .get("properties")
            .get("location")
            .get("description"),
            "The city and state, e.g. San Francisco, CA",
        )
        self.assertEqual(
            openai_tool.get("function")
            .get("parameters")
            .get("properties")
            .get("unit")
            .get("type"),
            "string",
        )
        self.assertEqual(
            openai_tool.get("function")
            .get("parameters")
            .get("properties")
            .get("unit")
            .get("enum"),
            ["celsius", "fahrenheit"],
        )

    def test_completion_conversion_openai_to_anthropic(self):
        openai_completion = {
            "id": "chatcmpl-AMFIYRPMRPGkKYgakgP9i9xlBLogy",
            "choices": [
                {
                    "finish_reason": "stop",
                    "index": 0,
                    "logprobs": None,
                    "message": {
                        "content": "Sure! Here you go: \n\n1, 2, 3, 4, 5, 6, 7, 8, 9, 10.",
                        "refusal": None,
                        "role": "assistant",
                    },
                }
            ],
            "created": 1729865386,
            "model": "gpt-4o-mini-2024-07-18",
            "object": "chat.completion",
            "system_fingerprint": "fp_f59a81427f",
            "usage": {
                "completion_tokens": 36,
                "prompt_tokens": 15,
                "total_tokens": 51,
                "completion_tokens_details": {"reasoning_tokens": 0},
                "prompt_tokens_details": {"cached_tokens": 0},
            },
        }

        anthropic_completion = convert_openai_completion_to_anthropic_completion(
            openai_completion
        )
        # print(anthropic_completion)

        self.assertEqual(
            anthropic_completion.get("id"), "chatcmpl-AMFIYRPMRPGkKYgakgP9i9xlBLogy"
        )
        self.assertEqual(anthropic_completion.get("model"), "gpt-4o-mini-2024-07-18")

        self.assertTrue(len(anthropic_completion.get("content")) == 1)

        self.assertEqual(anthropic_completion.get("content")[0].get("type"), "text")
        self.assertEqual(
            anthropic_completion.get("content")[0].get("text"),
            "Sure! Here you go: \n\n1, 2, 3, 4, 5, 6, 7, 8, 9, 10.",
        )

        ## usage and stop_tokens / stop reason formats aren't standardized, we don't use them, not bothering

    def test_tool_choice_conversion_anthropic_to_google(self):
        anthropic_tool_choice1 = {"type": "auto"}
        anthropic_tool_choice2 = {"type": "any"}
        google_tool_choice_1 = convert_anthropic_tool_choice_to_google_tool_choice(
            anthropic_tool_choice1
        )
        google_tool_choice_2 = convert_anthropic_tool_choice_to_google_tool_choice(
            anthropic_tool_choice2
        )
        self.assertEqual(
            google_tool_choice_1, {"function_calling_config": {"mode": "AUTO"}}
        )
        self.assertEqual(
            google_tool_choice_2, {"function_calling_config": {"mode": "ANY"}}
        )

    def test_tool_conversion_anthropic_to_google(self):
        anthropic_tool = {
            "name": "get_weather",
            "description": "Get the current weather in a given location",
            "input_schema": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The unit of temperature, either 'celsius' or 'fahrenheit'",
                    },
                },
                "required": ["location"],
            },
        }

        google_tool = convert_anthropic_tool_to_google_tool(anthropic_tool)

        self.assertEqual(google_tool.get("name"), "get_weather")
        self.assertEqual(
            google_tool.get("description"),
            "Get the current weather in a given location",
        )
        self.assertEqual(google_tool.get("parameters").get("type_"), "OBJECT")
        self.assertEqual(
            google_tool.get("parameters")
            .get("properties")
            .get("location")
            .get("type_"),
            "STRING",
        )
        self.assertEqual(
            google_tool.get("parameters")
            .get("properties")
            .get("location")
            .get("description"),
            "The city and state, e.g. San Francisco, CA",
        )
        self.assertEqual(
            google_tool.get("parameters").get("properties").get("unit").get("type_"),
            "STRING",
        )

    def test_completion_conversaion_google_to_anthropic(self):
        google_completion = {
            "candidates": [
                {
                    "content": {"parts": [{"text": "IV\n"}], "role": "model"},
                    "finish_reason": 1,
                    "avg_logprobs": -5.961192073300481e-08,
                    "safety_ratings": [],
                    "token_count": 0,
                    "grounding_attributions": [],
                }
            ],
            "usage_metadata": {
                "prompt_token_count": 15,
                "candidates_token_count": 2,
                "total_token_count": 17,
                "cached_content_token_count": 0,
            },
        }

        anthropic_completion = convert_google_completion_to_anthropic_completion(
            google_completion, model="gemini-1.5-pro-002"
        )

        self.assertTrue(len(anthropic_completion.get("content")) == 1)
        self.assertEqual(anthropic_completion.get("content")[0].get("type"), "text")
        self.assertEqual(anthropic_completion.get("content")[0].get("text"), "IV\n")
        self.assertEqual(anthropic_completion.get("usage").get("input_tokens"), 15)
        self.assertEqual(anthropic_completion.get("usage").get("output_tokens"), 2)


if __name__ == "__main__":
    unittest.main()
