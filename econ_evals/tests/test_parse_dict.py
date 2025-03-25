import unittest
from econ_evals.utils.helper_functions import parse_dict


class TestParseDict(unittest.TestCase):
    def test_parse_dict1(self):
        # parsing with JSON where literal_eval doesn't work

        s = """{"purchase_plan":"{'Offer_2': 7, 'Offer_3': 2}","is_final":true}"""
        s_parsed = parse_dict(s)

        self.assertEqual(
            s_parsed,
            {"purchase_plan": "{'Offer_2': 7, 'Offer_3': 2}", "is_final": True},
        )

    def test_parse_dict2(self):
        # normal parsing with literal_eval

        s = """{'W1': 'T1', 'W2': 'T2', 'W3': 'T3', 'W4': 'T7', 'W5': 'T8', 'W6': 'T6', 'W7': 'T4', 'W8': 'T5', 'W9': 'T9', 'W10': 'T10'}"""
        s_parsed = parse_dict(s)

        self.assertEqual(
            s_parsed,
            {
                "W1": "T1",
                "W2": "T2",
                "W3": "T3",
                "W4": "T7",
                "W5": "T8",
                "W6": "T6",
                "W7": "T4",
                "W8": "T5",
                "W9": "T9",
                "W10": "T10",
            },
        )

    def test_parse_dict3(self):
        # this is why we do the utf-8 thing

        s = """{'assignment': '{\\"W1\\": \\"T5\\", \\"W2\\": \\"T2\\", \\"W3\\": \\"T3\\", \\"W4\\": \\"T4\\", \\"W5\\": \\"T1\\", \\"W6\\": \\"T6\\", \\"W7\\": \\"T7\\", \\"W8\\": \\"T8\\", \\"W9\\": \\"T9\\", \\"W10\\": \\"T10\\"}'}"""

        s_parsed = parse_dict(s)

        self.assertEqual(
            s_parsed,
            {
                "assignment": '{"W1": "T5", "W2": "T2", "W3": "T3", "W4": "T4", "W5": "T1", "W6": "T6", "W7": "T7", "W8": "T8", "W9": "T9", "W10": "T10"}'
            },
        )

        s2 = s_parsed.get("assignment")

        s2_parsed = parse_dict(s2)

        self.assertEqual(
            s2_parsed,
            {
                "W1": "T5",
                "W2": "T2",
                "W3": "T3",
                "W4": "T4",
                "W5": "T1",
                "W6": "T6",
                "W7": "T7",
                "W8": "T8",
                "W9": "T9",
                "W10": "T10",
            },
        )
