import unittest

from econ_evals.experiments.efficiency_vs_equality.instance_generation import (
    compute_max_efficiency_alloc,
    generate_task_sizes,
    compute_revenue_from_alloc,
    compute_worker_pay_of_alloc,
    compute_greedy_max_efficiency_alloc,
    compute_greedy_max_equality_alloc,
)

import numpy as np


class TestAllocations(unittest.TestCase):
    def test_relative_order(self):
        for seed in range(1000):
            my_random = np.random.RandomState(seed)

            max_worker_productivity_gap = my_random.uniform(0, 20)
            worker_productivity_gap = my_random.uniform(0, max_worker_productivity_gap)

            num_periods = my_random.choice([20, 50, 100])
            num_explore_periods = int(
                num_periods * my_random.choice([0.01, 0.05, 0.1, 0.2])
            )
            num_workers = 4
            worker_ids = [f"W{i}" for i in range(num_workers)]

            avg_productivity = max_worker_productivity_gap / 2 + 1
            worker_productivities = [
                float(x)
                for x in np.linspace(
                    avg_productivity - worker_productivity_gap / 2,
                    avg_productivity + worker_productivity_gap / 2,
                    num_workers,
                )
            ]
            worker_id_to_productivities = {
                worker_id: productivity
                for worker_id, productivity in zip(worker_ids, worker_productivities)
            }
            worker_id_to_wage = {worker_id: 4 for worker_id in worker_ids}
            task_ids = [f"T{i}" for i in range(num_periods * num_workers)]

            task_id_to_task_size, alloc_max_equality = generate_task_sizes(
                num_periods, worker_ids, task_ids, my_random
            )

            alloc_max_efficiency = compute_max_efficiency_alloc(
                task_id_to_task_size, worker_id_to_productivities, task_ids, num_periods
            )

            alloc_approx_max_efficiency = compute_greedy_max_efficiency_alloc(
                task_id_to_task_size=task_id_to_task_size,
                task_ids=task_ids,
                num_periods=num_periods,
                num_workers=num_workers,
                worker_id_to_productivities=worker_id_to_productivities,
                num_explore_periods=num_explore_periods,
                my_random=my_random,
            )

            alloc_approx_max_equality = compute_greedy_max_equality_alloc(
                task_id_to_task_size=task_id_to_task_size,
                task_ids=task_ids,
                num_periods=num_periods,
                num_workers=num_workers,
                worker_id_to_wage=worker_id_to_wage,
            )

            # Construct a random allocation: for each period, randomly assign tasks to workers.
            alloc_random = {}
            for period in range(num_periods):
                period_task_ids = task_ids[
                    period * num_workers : (period + 1) * num_workers
                ]
                permuted_workers = worker_ids.copy()
                my_random.shuffle(permuted_workers)
                for task_id, worker_id in zip(period_task_ids, permuted_workers):
                    alloc_random[task_id] = worker_id

            # Compute revenues from the allocations
            revenue_max_equality = compute_revenue_from_alloc(
                alloc_max_equality,
                worker_id_to_productivities,
                task_id_to_task_size,
            )
            revenue_random = compute_revenue_from_alloc(
                alloc_random,
                worker_id_to_productivities,
                task_id_to_task_size,
            )
            revenue_max_efficiency = compute_revenue_from_alloc(
                alloc_max_efficiency,
                worker_id_to_productivities,
                task_id_to_task_size,
            )

            revenue_approx_max_efficiency = compute_revenue_from_alloc(
                alloc_approx_max_efficiency,
                worker_id_to_productivities,
                task_id_to_task_size,
            )

            revenue_approx_max_equality = compute_revenue_from_alloc(
                alloc_approx_max_equality,
                worker_id_to_productivities,
                task_id_to_task_size,
            )

            # Assert revenues are the order we expect

            # Some stuff we don't test:
            # - random could be worse than max_equality
            # - approx_max_equality could be worse than max_equality

            self.assertTrue(
                revenue_max_equality
                <= revenue_approx_max_efficiency
                <= revenue_max_efficiency,
            )

            self.assertTrue(revenue_approx_max_equality <= revenue_max_efficiency)

            self.assertTrue(
                revenue_random <= revenue_max_efficiency,
            )

            # Compute worker_pay.
            worker_pay_max_equality = compute_worker_pay_of_alloc(
                alloc_max_equality, task_id_to_task_size, worker_id_to_wage
            )
            worker_pay_max_efficiency = compute_worker_pay_of_alloc(
                alloc_max_efficiency, task_id_to_task_size, worker_id_to_wage
            )
            worker_pay_random = compute_worker_pay_of_alloc(
                alloc_random, task_id_to_task_size, worker_id_to_wage
            )

            worker_pay_approx_max_equality = compute_worker_pay_of_alloc(
                alloc_approx_max_equality, task_id_to_task_size, worker_id_to_wage
            )

            worker_pay_approx_max_efficiency = compute_worker_pay_of_alloc(
                alloc_approx_max_efficiency, task_id_to_task_size, worker_id_to_wage
            )

            # For inequality comparison, compute the range (max - min) of worker_pay.
            inequality_max_efficiency = max(worker_pay_max_efficiency.values()) - min(
                worker_pay_max_efficiency.values()
            )
            inequality_random = max(worker_pay_random.values()) - min(
                worker_pay_random.values()
            )

            inequality_max_equality = max(worker_pay_max_equality.values()) - min(
                worker_pay_max_equality.values()
            )

            inequality_approx_max_efficiency = max(
                worker_pay_approx_max_efficiency.values()
            ) - min(worker_pay_approx_max_efficiency.values())

            inequality_approx_max_equality = max(
                worker_pay_approx_max_equality.values()
            ) - min(worker_pay_approx_max_equality.values())

            self.assertEqual(0, inequality_max_equality)

            self.assertTrue(
                inequality_max_equality
                <= inequality_approx_max_equality
                <= inequality_approx_max_efficiency
                <= inequality_max_efficiency,
            )

            self.assertTrue(
                inequality_max_equality
                <= inequality_random
                <= inequality_max_efficiency,
            )


if __name__ == "__main__":
    unittest.main()
