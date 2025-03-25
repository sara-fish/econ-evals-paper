import unittest

from econ_evals.tests.logit_demand_market_logic import (
    get_quantities,
    get_profits,
    get_monopoly_prices,
    get_nash_prices,
)

# We include an implementation of regular logit demand in tests/, just to make sure nested logit reduces to that
# This file makes sure that logit demand works as expected


class TestLogitDemand(unittest.TestCase):
    def test_get_quantities(self):
        for alpha in [0.5, 1, 1.5, 2]:
            quantities = get_quantities(
                p=(1.5 * alpha, 1.5 * alpha), multiplier=100, alpha=alpha
            )

            self.assertAlmostEqual(quantities[0], 46.83105308334812, places=4)
            self.assertAlmostEqual(quantities[1], 46.83105308334812, places=4)

    def test_get_profits(self):
        for alpha in [0.5, 1, 1.5, 2]:
            profits = get_profits(
                p=(1.5 * alpha, 1.5 * alpha), multiplier=100, alpha=alpha
            )

            self.assertAlmostEqual(profits[0], 23.41552654167406, places=3)
            self.assertAlmostEqual(profits[1], 23.41552654167406, places=3)

    def test_get_monopoly_prices(self):
        for alpha in [0.5, 1, 1.5, 2]:
            monopoly_prices = get_monopoly_prices(alpha=alpha)

            self.assertAlmostEqual(monopoly_prices[0], 1.92498 * alpha, places=3)
            self.assertAlmostEqual(monopoly_prices[1], 1.92498 * alpha, places=3)

    def test_get_nash_prices(self):
        for alpha in [0.5, 1, 1.5, 2]:
            nash_prices = get_nash_prices(alpha=alpha)

            self.assertAlmostEqual(nash_prices[0], 1.4729 * alpha, places=3)
            self.assertAlmostEqual(nash_prices[1], 1.4729 * alpha, places=3)


if __name__ == "__main__":
    unittest.main()
