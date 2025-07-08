from ast import literal_eval
import unittest

from econ_evals.utils.llm_tools import (
    call_openai,
    call_anthropic,
    call_llm,
    call_google,
)


class TestLLMMethods(unittest.TestCase):
    def test_call_openai(self):
        messages = [{"role": "user", "content": "What is 2+2"}]
        _, response, _ = call_openai(
            model="gpt-4o-mini-2024-07-18",
            messages=messages,
            system="Respond just with the number",
            temperature=0,
            tools=None,
            tool_choice=None,
        )

        self.assertTrue("4" in response)

    def test_call_anthropic(self):
        messages = [{"role": "user", "content": "What is 2+2"}]
        _, response, completion = call_anthropic(
            model="claude-3-5-haiku-20241022",
            messages=messages,
            system="Respond with just the number",
            temperature=0,
            tools=None,
            tool_choice=None,
        )
        self.assertTrue("4" in response)

    def test_call_google(self):
        messages = [{"role": "user", "parts": ["What is 2+2"]}]
        _, response, _ = call_google(
            model="gemini-1.5-flash-002",
            messages=messages,
            system="Respond with just the number",
            temperature=0,
            tools=None,
            tool_choice=None,
        )
        self.assertTrue("4" in response)

    def test_call_anthropic_tools(self):
        tools = [
            {
                "name": "get_random_string",
                "description": "Returns a random string of a pre-specified length.",
                "input_schema": {
                    "type": "object",
                    "properties": {
                        "length": {
                            "type": "string",
                            "description": "The length of the random string to be computed",
                        }
                    },
                    "required": ["length"],
                },
            }
        ]
        _, _, completion = call_anthropic(
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Give me a random 17 bit string"}],
            temperature=0,
            tools=tools,
            tool_choice={"type": "any"},
        )
        # print(completion)

        self.assertTrue(len(completion["content"]) == 1)
        self.assertTrue(completion["content"][0]["type"] == "tool_use")
        self.assertTrue(completion["content"][0]["input"]["length"] == "17")

    def test_call_openai_tools(self):
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_random_string",
                    "description": "Returns a random string of a pre-specified length.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "length": {
                                "type": "string",
                                "description": "The length of the random string to be computed",
                            }
                        },
                        "required": ["length"],
                    },
                },
            }
        ]
        _, _, completion = call_openai(
            model="gpt-4o-mini-2024-07-18",
            messages=[{"role": "user", "content": "Give me a random 17 bit string"}],
            temperature=0,
            tools=tools,
            tool_choice="required",
        )

        self.assertTrue(len(completion["choices"]) == 1)
        self.assertTrue(completion["choices"][0]["finish_reason"] == "tool_calls")
        self.assertTrue(
            literal_eval(
                completion["choices"][0]["message"]["tool_calls"][0]["function"][
                    "arguments"
                ]
            )["length"]
            == "17"
        )

    def test_call_google_tools(self):
        tools = [
            {
                "name": "get_random_string",
                "description": "Returns a random string of a pre-specified length.",
                "parameters": {
                    "type_": "OBJECT",
                    "properties": {
                        "length": {
                            "type_": "INTEGER",
                            "description": "The length of the random string to be computed",
                        }
                    },
                    "required": ["length"],
                },
            }
        ]

        _, _, completion = call_google(
            model="gemini-1.5-pro-002",
            messages=[{"role": "user", "parts": ["Give me a random 17 bit string"]}],
            temperature=0,
            tools=tools,
            tool_choice={"function_calling_config": {"mode": "ANY"}},
        )

        self.assertTrue(len(completion["candidates"]) == 1)
        self.assertTrue(
            "function_call" in completion["candidates"][0]["content"]["parts"][0]
        )
        self.assertEqual(
            completion["candidates"][0]["content"]["parts"][0]["function_call"]["name"],
            "get_random_string",
        )
        self.assertEqual(
            completion["candidates"][0]["content"]["parts"][0]["function_call"]["args"][
                "length"
            ],
            17,
        )

    def test_call_llm(self):
        for model in [
            "claude-3-5-haiku-20241022",
            "gpt-4o-mini-2024-07-18",
            "gemini-1.5-flash-002",
            "gemini-2.5-pro-preview-06-05",
        ]:
            _, response, _ = call_llm(
                model=model,
                messages=[{"role": "user", "content": "What is 2+2"}],
                system="Respond with just the number",
                temperature=0,
                tools=None,
                tool_choice=None,
            )

            self.assertTrue("4" in response)

    def test_call_llm_tools(self):
        for model in [
            "claude-3-5-haiku-20241022",
            "gpt-4o-mini-2024-07-18",
            "gemini-1.5-flash-002",
            "gemini-2.5-pro-preview-06-05",
        ]:
            tools = [
                {
                    "name": "get_random_string",
                    "description": "Returns a random string of a pre-specified length.",
                    "input_schema": {
                        "type": "object",
                        "properties": {
                            "length": {
                                "type": "string",
                                "description": "The length of the random string to be computed",
                            }
                        },
                        "required": ["length"],
                    },
                }
            ]
            _, _, completion = call_llm(
                model=model,
                messages=[
                    {"role": "user", "content": "Give me a random 17 bit string"}
                ],
                temperature=0,
                tools=tools,
                tool_choice={"type": "any"},
            )

            self.assertTrue(len(completion["content"]) == 1)
            self.assertTrue(completion["content"][0]["type"] == "tool_use")
            self.assertTrue(completion["content"][0]["input"]["length"] == "17")


if __name__ == "__main__":
    unittest.main()
