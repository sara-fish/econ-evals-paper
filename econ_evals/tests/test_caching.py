import unittest

from econ_evals.utils.llm_tools import (
    cache_anthropic_messages,
    cache_anthropic_system,
    cache_anthropic_tools,
)


class TestAnthropicCaching(unittest.TestCase):
    def test_cache_anthropic_system(self):
        # Test that the format of the system caching is correct

        system = "Pick one tool"
        modified_system = cache_anthropic_system(system)
        self.assertTrue(isinstance(modified_system, list))
        self.assertEqual(modified_system[0]["type"], "text")
        self.assertEqual(modified_system[0]["text"], system)
        self.assertEqual(modified_system[0]["cache_control"]["type"], "ephemeral")

    def test_cache_anthropic_tools(self):
        tools = [
            {
                "name": "get_num_consumers",
                "description": "Gets the number of shoppers who want to look at the daily deal.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "get_round",
                "description": "Returns the current round.",
                "input_schema": {"type": "object", "properties": {}},
            },
            {
                "name": "get_pricing_history",
                "description": "Returns client data for previous rounds (prices set for that period's product and how much of the product was sold).",
                "input_schema": {
                    "type": "object",
                    "properties": {},
                },
            },
        ]
        modified_tools = cache_anthropic_tools(tools)
        # Test that the tools are unchanged besides the final tool, which has a cache_control key added to it

        self.assertEqual(isinstance(modified_tools, list), True)
        self.assertEqual(modified_tools[-1]["cache_control"], {"type": "ephemeral"})
        self.assertTrue("cache_control" not in modified_tools[0].keys())
        self.assertEqual(modified_tools[:-1], tools[:-1])

    def test_cache_anthropic_messages(self):
        messages = [
            {"role": "user", "content": "Please pick a tool."},
            {"role": "assistant", "content": "Here are some tools you can use."},
            {"role": "user", "content": "I want to use the tool called 'get_weather'."},
            {"role": "assistant", "content": "I will now use the tool 'get_weather'."},
            {"role": "user", "content": "I will now use the tool 'make_cake'."},
            {"role": "assistant", "content": "The cake is ready."},
            {"role": "user", "content": "Now, please sing a song"},
        ]
        modified_messages = cache_anthropic_messages(messages)
        # print(modified_messages[-1])
        self.assertTrue(isinstance(modified_messages, list))
        self.assertEqual(
            modified_messages[-1]["content"][0]["cache_control"]["type"], "ephemeral"
        )
        self.assertEqual(modified_messages[-1]["role"], "user")
        self.assertEqual(modified_messages[-3]["role"], "user")
        self.assertEqual(
            modified_messages[-3]["content"][0]["cache_control"]["type"], "ephemeral"
        )
        self.assertTrue("cache_control" not in modified_messages[0].keys())

        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "\nNow you can start using the tools to make your price-setting decisions. The chat history will reset every round, but you'll still have access to all data from previous rounds via the respective tools (get_pricing_history, read_public_notes, read_private_notes). Before you set a new price, you should check get_pricing_history to check how much you made by selling the most recent product. Customers will not buy at a price that is the same as or greater than one they have already accepted in that round. If you use the tool sell_new_product, you will advance to a new round and sell a new product to new customers. You should take advantage of the opportunity to sell a new product.\nIt is not recommended to set any prices above 1 dollars. You can set as many prices as you want in a given period.",
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "id": "toolu_01KjPaq55VVztN5y7DNEvTTM",
                        "input": {},
                        "name": "get_round",
                        "type": "tool_use",
                    },
                    {
                        "id": "toolu_01F6GErgczXRBxcJHBrScvGN",
                        "input": {},
                        "name": "get_pricing_history",
                        "type": "tool_use",
                    },
                    {
                        "id": "toolu_01FtSKJKzErmt9PrrE9q8uEZ",
                        "input": {},
                        "name": "get_num_consumers",
                        "type": "tool_use",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01KjPaq55VVztN5y7DNEvTTM",
                        "content": "{'attempt_number': 0}",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01F6GErgczXRBxcJHBrScvGN",
                        "content": "Round number 1:\nNumber of consumers: 6\nNew price:\nPrice: 0.99\nQty: 1/100 sold\nRevenue: 0.053\n\n",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01FtSKJKzErmt9PrrE9q8uEZ",
                        "content": "{'num_consumers': 6}",
                    },
                    {
                        "type": "text",
                        "text": "Now it is time for you to use more tools.",
                    },
                ],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "id": "toolu_01KjPaq55VVztN5y7DNEvTTM",
                        "input": {},
                        "name": "get_round",
                        "type": "tool_use",
                    },
                    {
                        "id": "toolu_01F6GErgczXRBxcJHBrScvGN",
                        "input": {},
                        "name": "get_pricing_history",
                        "type": "tool_use",
                    },
                    {
                        "id": "toolu_01FtSKJKzErmt9PrrE9q8uEZ",
                        "input": {},
                        "name": "get_num_consumers",
                        "type": "tool_use",
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01KjPaq55VVztN5y7DNEvTTM",
                        "content": "{'attempt_number': 0}",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01F6GErgczXRBxcJHBrScvGN",
                        "content": "Round number 1:\nNumber of consumers: 6\nNew price:\nPrice: 0.99\nQty: 1/100 sold\nRevenue: 0.053\n\n",
                    },
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_01FtSKJKzErmt9PrrE9q8uEZ",
                        "content": "{'num_consumers': 6}",
                    },
                    {
                        "type": "text",
                        "text": "Now it is time for you to use more tools.",
                    },
                ],
            },
        ]

        modified_messages = cache_anthropic_messages(messages)
        self.assertTrue(isinstance(modified_messages, list))
        self.assertEqual(
            modified_messages[-1]["content"][-1]["cache_control"]["type"], "ephemeral"
        )
        self.assertEqual(
            modified_messages[-3]["content"][-1]["cache_control"]["type"], "ephemeral"
        )
        self.assertEqual(modified_messages[-1]["role"], "user")
        self.assertTrue(
            "cache_control" not in modified_messages[1]["content"][-1].keys()
        )


if __name__ == "__main__":
    unittest.main()
