import unittest

from econ_evals.experiments.pricing.generate_instance import (
    generate_group_idxs,
)

import numpy as np


class TestPricingInstanceGeneration(unittest.TestCase):
    def test_generate_group_idxs(self):
        my_random = np.random.RandomState(0)

        for _ in range(10):
            num_products = np.random.randint(1, 10)

            group_idxs = generate_group_idxs(
                num_products=num_products,
                p=0.2,
                cutoff_proportion=0.25,
                my_random=my_random,
            )
            self.assertEqual(len(group_idxs), num_products)
            self.assertEqual(set(group_idxs), set(range(1, 1 + len(set(group_idxs)))))
