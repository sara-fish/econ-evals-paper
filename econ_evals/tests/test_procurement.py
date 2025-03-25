import math
import unittest

from econ_evals.experiments.procurement.opt_solver import (
    compute_opt,
    Menu,
    Entry,
    evaluate_alloc,
)
from econ_evals.experiments.procurement.generate_instance import generate_instance
import numpy as np


class TestComputeOpt(unittest.TestCase):
    def test1(self):
        entries = [
            Entry(id="A", type="basic", cost=1, contents={"A": 1}),
            Entry(id="B", type="basic", cost=2, contents={"B": 1}),
            Entry(id="C", type="basic", cost=3, contents={"C": 1}),
        ]

        item_to_effectiveness = {"A": 1, "B": 1, "C": 1}

        menu = Menu(entries)

        entry_groups = [["A", "B", "C"]]

        alloc_min, log_min = compute_opt(
            menu,
            10,
            entry_groups,
            item_to_effectiveness,
            group_weights=[1],
            agg_type="min",
        )

        self.assertEqual(alloc_min, {"A": 10, "B": 0, "C": 0})
        self.assertEqual(log_min["num_solutions"], 1)

        alloc_prod, log_prod = compute_opt(
            menu,
            10,
            entry_groups,
            item_to_effectiveness,
            group_weights=[1],
            agg_type="prod",
        )

        self.assertEqual(alloc_prod, {"A": 10, "B": 0, "C": 0})
        self.assertEqual(log_prod["num_solutions"], 1)

    def test2(self):
        entries = [
            Entry(id="A", type="basic", cost=1, contents={"A": 1}),
            Entry(id="B", type="basic", cost=2, contents={"B": 1}),
            Entry(id="C", type="basic", cost=3, contents={"C": 1}),
        ]

        item_to_effectiveness = {"A": 1, "B": 1, "C": 1}

        menu = Menu(entries)

        entry_groups = [["A", "B"], ["C"]]

        alloc_min, log_min = compute_opt(
            menu,
            12,
            entry_groups,
            item_to_effectiveness,
            group_weights=[1, 1],
            agg_type="min",
        )

        self.assertEqual(alloc_min, {"A": 3, "B": 0, "C": 3})
        self.assertEqual(log_min["num_solutions"], 1)

        # utility = sqrt(A+B) * sqrt(C) s.t. A+2B+3C <= 12
        alloc_prod, log_prod = compute_opt(
            menu,
            12,
            entry_groups,
            item_to_effectiveness,
            group_weights=[0.5, 0.5],
            agg_type="prod",
        )

        self.assertEqual(alloc_prod, {"A": 6, "B": 0, "C": 2})
        self.assertEqual(log_prod["num_solutions"], 1)

    def test3(self):
        entries = [
            Entry(id="A", type="basic", cost=1, contents={"A": 1}),
            Entry(id="B", type="basic", cost=2, contents={"B": 1}),
            Entry(id="C", type="basic", cost=3, contents={"C": 1}),
        ]

        item_to_effectiveness = {"A": 1, "B": 1, "C": 1}

        menu = Menu(entries)

        entry_groups = [["A"], ["B"], ["C"]]

        alloc_min, log_min = compute_opt(
            menu,
            12,
            entry_groups,
            item_to_effectiveness,
            group_weights=[1, 1, 1],
            agg_type="min",
        )

        self.assertEqual(alloc_min, {"A": 2, "B": 2, "C": 2})
        self.assertEqual(log_min["num_solutions"], 1)

        alloc_prod, log_prod = compute_opt(
            menu,
            36,
            entry_groups,
            item_to_effectiveness,
            group_weights=[1 / 3, 1 / 3, 1 / 3],
            agg_type="prod",
        )

        self.assertEqual(alloc_prod, {"A": 12, "B": 6, "C": 4})
        self.assertEqual(log_prod["num_solutions"], 1)

    def test4(self):
        entries = [
            Entry(id="A1", type="basic", cost=2, contents={"A1": 1}),
            Entry(
                id="A2",
                type="bulk_discount",
                cost=1.8,
                min_quantity=10,
                contents={"A2": 1},
            ),
        ]

        item_to_effectiveness = {"A1": 1, "A2": 1}

        menu = Menu(entries)

        entry_groups = [["A1", "A2"]]

        ## Budget large enough to afford bulk
        alloc, log = compute_opt(
            menu,
            18,
            entry_groups,
            item_to_effectiveness,
            group_weights=[1],
            agg_type="min",
        )

        self.assertEqual(alloc, {"A1": 0, "A2": 10})
        self.assertEqual(log["num_solutions"], 1)

        ## Budget too small, buy single
        alloc, log = compute_opt(
            menu,
            16,
            entry_groups,
            item_to_effectiveness,
            group_weights=[1],
            agg_type="min",
        )

        self.assertEqual(alloc, {"A1": 8, "A2": 0})
        self.assertEqual(log["num_solutions"], 1)

    def test5(self):
        entries = [
            Entry(id="A1", type="basic", cost=2, contents={"A1": 1}),
            Entry(
                id="A2",
                type="two_part_tariff",
                fixed_cost=5,
                variable_cost=1,
                contents={"A2": 1},
            ),
        ]

        item_to_effectiveness = {"A1": 1, "A2": 1}

        ## Budget large enough to afford two-part tariff
        menu = Menu(entries)

        entry_groups = [["A1", "A2"]]
        alloc, log = compute_opt(
            menu,
            11,
            entry_groups,
            item_to_effectiveness,
            group_weights=[1],
            agg_type="min",
        )

        self.assertEqual(alloc, {"A1": 0, "A2": 6})
        self.assertEqual(log["num_solutions"], 1)

        ## Transition point, both work
        alloc, log = compute_opt(
            menu,
            10,
            entry_groups,
            item_to_effectiveness,
            group_weights=[1],
            agg_type="min",
        )
        self.assertEqual(log["num_solutions"], 2)

        ## Budget too small, buy single
        alloc, log = compute_opt(
            menu,
            8,
            entry_groups,
            item_to_effectiveness,
            group_weights=[1],
            agg_type="min",
        )

        self.assertEqual(alloc, {"A1": 4, "A2": 0})
        self.assertEqual(log["num_solutions"], 1)

    def test6(self):
        entry1_bundle_cheaper = Entry(
            id="Offer 1",
            type="basic",
            cost=3,
            contents={"A1": 2, "B1": 1},
        )

        entry1_bundle_same = Entry(
            id="Offer 1",
            type="basic",
            cost=4,
            contents={"A1": 2, "B1": 1},
        )

        entry1_bundle_pricier = Entry(
            id="Offer 1",
            type="basic",
            cost=5,
            contents={"A1": 2, "B1": 1},
        )

        entry2 = Entry(
            id="Offer 2",
            type="basic",
            cost=2,
            contents={"B2": 1},
        )

        entry3 = Entry(
            id="Offer 3",
            type="basic",
            cost=1,
            contents={"A2": 1},
        )

        item_to_effectiveness = {
            "A1": 1,
            "B1": 1,
            "A2": 1,
            "B2": 1,
        }

        # Bundle is better deal than separately

        menu = Menu([entry1_bundle_cheaper, entry2, entry3])
        entry_groups = [["A1", "A2"], ["B1", "B2"]]

        alloc, log = compute_opt(
            menu,
            30,
            entry_groups,
            item_to_effectiveness,
            group_weights=[1, 1],
            agg_type="min",
        )

        self.assertEqual(log["num_solutions"], 1)
        self.assertEqual(alloc, {"Offer 1": 6, "Offer 2": 6, "Offer 3": 0})

        # Bundle is same deal as separately

        menu = Menu([entry1_bundle_same, entry2, entry3])

        alloc, log = compute_opt(
            menu,
            30,
            entry_groups,
            item_to_effectiveness,
            group_weights=[1, 1],
            agg_type="min",
        )

        self.assertGreater(log["num_solutions"], 1)  # >= 2 because mixing between two

        # Bundle is worse deal than separately

        menu = Menu([entry1_bundle_pricier, entry2, entry3])

        alloc, log = compute_opt(
            menu,
            30,
            entry_groups,
            item_to_effectiveness,
            group_weights=[1, 1],
            agg_type="min",
        )

        self.assertEqual(log["num_solutions"], 1)
        self.assertEqual(alloc, {"Offer 1": 0, "Offer 2": 10, "Offer 3": 10})


class TestComputeOptNontrivialEffectiveness(unittest.TestCase):
    def test1(self):
        entries = [
            Entry(id="A", type="basic", cost=1, contents={"A": 1}),
            Entry(id="B", type="basic", cost=2, contents={"B": 1}),
            Entry(id="C", type="basic", cost=3, contents={"C": 1}),
        ]

        item_to_effectiveness = {"A": 1, "B": 1, "C": 4}

        menu = Menu(entries)

        entry_groups = [["A", "B", "C"]]

        alloc_min, log_min = compute_opt(
            menu,
            9,
            entry_groups,
            item_to_effectiveness,
            group_weights=[1],
            agg_type="min",
        )

        self.assertEqual(alloc_min, {"A": 0, "B": 0, "C": 3})
        self.assertEqual(log_min["num_solutions"], 1)
        self.assertEqual(log_min["total_utility"], 12)

        alloc_prod, log_prod = compute_opt(
            menu,
            9,
            entry_groups,
            item_to_effectiveness,
            group_weights=[1],
            agg_type="prod",
        )

        self.assertEqual(alloc_prod, {"A": 0, "B": 0, "C": 3})
        self.assertEqual(log_prod["num_solutions"], 1)
        self.assertEqual(log_prod["total_utility"], 12)

    def test2(self):
        entries = [
            Entry(id="A", type="basic", cost=1, contents={"A": 1}),
            Entry(id="B", type="basic", cost=2, contents={"B": 1}),
            Entry(id="C", type="basic", cost=3, contents={"C": 1}),
        ]

        item_to_effectiveness = {"A": 1, "B": 3, "C": 1}

        menu = Menu(entries)

        entry_groups = [["A", "B"], ["C"]]

        alloc_min, log_min = compute_opt(
            menu,
            11,
            entry_groups,
            item_to_effectiveness,
            group_weights=[1, 1],
            agg_type="min",
        )

        self.assertEqual(alloc_min, {"A": 0, "B": 1, "C": 3})
        self.assertEqual(log_min["num_solutions"], 1)
        self.assertEqual(log_min["total_utility"], 3)

        # utility = sqrt(A+3B) * sqrt(C) s.t. A+2B+3C <= 11
        alloc_prod, log_prod = compute_opt(
            menu,
            12,
            entry_groups,
            item_to_effectiveness,
            group_weights=[0.5, 0.5],
            agg_type="prod",
        )

        self.assertEqual(alloc_prod, {"A": 0, "B": 3, "C": 2})
        self.assertEqual(log_prod["num_solutions"], 1)
        self.assertAlmostEqual(log_prod["total_utility"], math.sqrt(18), places=4)

    def test4(self):
        entries = [
            Entry(id="A1", type="basic", cost=2, contents={"A1": 1}),
            Entry(
                id="A2",
                type="bulk_discount",
                cost=1.8,
                min_quantity=10,
                contents={"A2": 1},
            ),
        ]

        item_to_effectiveness = {"A1": 2, "A2": 1}

        menu = Menu(entries)

        entry_groups = [["A1", "A2"]]

        ## Budget large enough to afford bulk
        # but with effectiveness, you should still buy separate
        alloc, log = compute_opt(
            menu,
            18,
            entry_groups,
            item_to_effectiveness,
            group_weights=[1],
            agg_type="min",
        )

        self.assertEqual(alloc, {"A1": 9, "A2": 0})
        self.assertEqual(log["num_solutions"], 1)
        self.assertEqual(log["total_utility"], 18)

        ## Budget too small, buy single (still)
        alloc, log = compute_opt(
            menu,
            16,
            entry_groups,
            item_to_effectiveness,
            group_weights=[1],
            agg_type="min",
        )

        self.assertEqual(alloc, {"A1": 8, "A2": 0})
        self.assertEqual(log["num_solutions"], 1)
        self.assertEqual(log["total_utility"], 16)


class TestGenerateInstance(unittest.TestCase):
    def test_no_exceptions_thrown(self):
        my_random = np.random.RandomState(1)

        for _ in range(10):
            num_inputs = my_random.randint(1, 10)
            num_alternatives_per_input = my_random.randint(1, 10)
            num_entries = my_random.randint(
                num_inputs * num_alternatives_per_input,
                10 * num_inputs * num_alternatives_per_input,
            )

            _, _, _, _, _ = generate_instance(
                num_inputs=num_inputs,
                num_alternatives_per_input=num_alternatives_per_input,
                num_entries=num_entries,
                my_random=my_random,
            )

    def test1(self):
        my_random = np.random.RandomState(1)

        menu1, _, _, _, _ = generate_instance(
            num_inputs=4,
            num_alternatives_per_input=3,
            num_entries=12,
            my_random=my_random,
        )
        entry_types1 = [entry.type for entry in menu1]

        my_random = np.random.RandomState(1)

        menu2, _, _, _, _ = generate_instance(
            num_inputs=4,
            num_alternatives_per_input=3,
            num_entries=12,
            my_random=my_random,
        )
        entry_types2 = [entry.type for entry in menu2]

        assert len(menu1) == len(menu2) == 12
        assert entry_types1 == entry_types2

    def test_evaluate_alloc_min(self):
        for seed in range(5):
            for agg_type in ["min", "prod"]:
                my_random = np.random.RandomState(seed)

                menu, budget, item_groups, _, item_to_effectiveness = generate_instance(
                    3, 3, num_entries=9, my_random=my_random
                )

                if agg_type == "min":
                    group_weights = [1 for _ in item_groups]
                elif agg_type == "prod":
                    group_weights = [1 / len(item_groups) for _ in item_groups]
                else:
                    raise NotImplementedError

                alloc, log = compute_opt(
                    menu,
                    budget,
                    item_groups,
                    item_to_effectiveness,
                    group_weights=group_weights,
                    agg_type=agg_type,
                )
                total_cost_opt = log["total_cost"]
                total_utility_opt = log["total_utility"]

                is_feasible, invalid_reason, total_cost, total_utility = evaluate_alloc(
                    menu,
                    alloc=alloc,
                    budget=budget,
                    item_groups=item_groups,
                    item_to_effectiveness=item_to_effectiveness,
                    group_weights=group_weights,
                    agg_type=agg_type,
                )

                self.assertTrue(is_feasible)
                self.assertEqual(invalid_reason, "")

                self.assertAlmostEqual(total_cost, total_cost_opt, places=4)
                self.assertAlmostEqual(total_utility, total_utility_opt, places=4)
