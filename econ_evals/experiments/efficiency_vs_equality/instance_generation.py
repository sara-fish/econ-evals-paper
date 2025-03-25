import numpy as np

AVERAGE_TASK_SIZE = 60


def generate_worker_productivities(
    worker_productivity_gap: float,
    max_worker_productivity_gap: float,
    worker_ids: list[str],
    my_random: np.random.RandomState,
) -> dict[str, float]:
    """
    Generate worker productivities (no randomness -- just evenly spaced values with total gap worker_productivity_gap),
    and mean max_worker_productivity_gap / 2 + 1
    (so that across treatments where we vary worker productivity, average worker productivity is the same)
    """
    num_workers = len(worker_ids)
    avg_productivity = max_worker_productivity_gap / 2 + 1

    worker_productivities = [
        float(x)
        for x in my_random.permutation(
            np.linspace(
                avg_productivity - worker_productivity_gap / 2,
                avg_productivity + worker_productivity_gap / 2,
                num_workers,
            )
        )
    ]
    assert (
        min(worker_productivities) > 0
    ), f"worker_productivity_gap {worker_productivity_gap} too large, causing negative productivities (smallest {min(worker_productivities)})"
    assert len(worker_productivities) == len(worker_ids)
    worker_id_to_productivities = {
        worker_id: productivity
        for worker_id, productivity in zip(worker_ids, worker_productivities)
    }
    return worker_id_to_productivities


def generate_task_sizes(
    num_periods: int,
    worker_ids: list[str],
    task_ids: list[str],
    my_random: np.random.RandomState,
) -> tuple[dict[str, int], dict[str, str]]:
    """
    Randomly sample task sizes, and also return auxiliary information needed to keep track of the most equal allocation
    """
    num_workers = len(worker_ids)
    worker_id_to_task_sizes_in_equal_alloc = {}  # map each worker to a list of task sizes so that if the worker does each of those tasks in each period, at the end all workers will be equal
    total_task_sizes_per_agent = (
        AVERAGE_TASK_SIZE * num_periods
    )  # when all agents earn equally, all earn this much
    for worker_id in worker_ids:
        # For each worker, generate uniform iid task sizes that sum to the total
        splits = (
            [0]
            + list(
                my_random.choice(
                    range(
                        1, total_task_sizes_per_agent
                    ),  # note: data was collected using range(total_task_sizes_per_agent)
                    size=num_periods - 1,
                    replace=False,
                )
            )
            + [total_task_sizes_per_agent]
        )
        splits = list(map(int, sorted(splits)))
        task_sizes = [splits[i + 1] - splits[i] for i in range(len(splits) - 1)]
        assert sum(task_sizes) == total_task_sizes_per_agent
        assert len(task_sizes) == num_periods
        worker_id_to_task_sizes_in_equal_alloc[worker_id] = task_sizes
    assert all(
        [
            len(worker_id_to_task_sizes_in_equal_alloc[worker_id]) == num_periods
            for worker_id in worker_ids
        ]
    )
    task_sizes_per_period = [
        [
            worker_id_to_task_sizes_in_equal_alloc[worker_id][period_num]
            for worker_id in worker_ids
        ]
        for period_num in range(num_periods)
    ]
    task_ids_per_period = [
        list(
            map(
                str,
                my_random.permutation(
                    task_ids[i * num_workers : (i + 1) * num_workers]
                ),
            )
        )
        for i in range(num_periods)
    ]

    task_id_to_task_size = {}
    for period_num in range(num_periods):
        task_ids_this_period = task_ids_per_period[period_num]
        task_sizes_this_period = task_sizes_per_period[period_num]
        for task_id, task_size in zip(task_ids_this_period, task_sizes_this_period):
            task_id_to_task_size[task_id] = task_size

    task_id_to_worker_id_max_equality = {}
    for worker_idx, worker_id in enumerate(worker_ids):
        for period_num in range(num_periods):
            task_id = task_ids_per_period[period_num][worker_idx]
            task_id_to_worker_id_max_equality[task_id] = worker_id

    return task_id_to_task_size, task_id_to_worker_id_max_equality


def compute_max_efficiency_alloc(
    task_id_to_task_size: dict[str, int],
    worker_id_to_productivities: dict[str, float],
    task_ids: list[str],
    num_periods: int,
) -> dict[str, str]:
    """
    Return mapping of task IDs to worker IDs that results in the most equal allocation
    """
    # Give more productive workers larger tasks, in each period
    task_id_to_worker_id_max_efficiency = {}
    num_workers = len(worker_id_to_productivities)
    for period_num in range(num_periods):
        task_ids_this_period = task_ids[
            period_num * num_workers : (period_num + 1) * num_workers
        ]
        sorted_worker_ids = sorted(
            worker_id_to_productivities.keys(),
            key=lambda worker_id: worker_id_to_productivities[worker_id],
        )
        sorted_task_ids = sorted(
            task_ids_this_period,
            key=lambda task_id: task_id_to_task_size[task_id],
        )
        for worker_idx, worker_id in enumerate(sorted_worker_ids):
            task_id = sorted_task_ids[worker_idx]
            task_id_to_worker_id_max_efficiency[task_id] = worker_id
    return task_id_to_worker_id_max_efficiency


def compute_worker_pay_of_alloc(
    alloc: dict[str, str],
    task_id_to_task_size: dict[str, int],
    worker_id_to_wage: dict[str, float],
) -> dict[str, float]:
    """
    Return a dict mapping worker id to the cumulative pay earned by that worker
    """
    worker_id_to_pay = {worker_id: 0 for worker_id in worker_id_to_wage}
    for task_id, worker_id in alloc.items():
        worker_id_to_pay[worker_id] += (
            task_id_to_task_size[task_id] * worker_id_to_wage[worker_id]
        )
    return worker_id_to_pay


def compute_per_worker_revenue_from_alloc(
    alloc: dict[str, str],
    worker_id_to_productivities: dict[str, float],
    task_id_to_task_size: dict[str, int],
) -> dict[str, float]:
    """
    Return a dict mapping worker id to the revenue earned for that company by that worker
    """
    worker_id_to_revenue = {worker_id: 0 for worker_id in worker_id_to_productivities}
    for task_id, worker_id in alloc.items():
        worker_id_to_revenue[worker_id] += (
            task_id_to_task_size[task_id] * worker_id_to_productivities[worker_id]
        )
    return worker_id_to_revenue


def compute_revenue_from_alloc(
    alloc: dict[str, str],
    worker_id_to_productivities: dict[str, float],
    task_id_to_task_size: dict[str, int],
) -> float:
    """
    Return the total company revenue earned by the alloc
    """
    revenue = sum(
        compute_per_worker_revenue_from_alloc(
            alloc=alloc,
            worker_id_to_productivities=worker_id_to_productivities,
            task_id_to_task_size=task_id_to_task_size,
        ).values()
    )
    return revenue


def compute_greedy_max_equality_alloc(
    task_id_to_task_size: dict[str, float],
    task_ids: list[str],
    num_periods: int,
    num_workers: int,
    worker_id_to_wage: dict[str, float],
) -> dict[str, str]:
    """
    Return the alloc that you get running a naive greedy algorithm that tries to equalize worker earnings
    """
    allocation = {}
    earnings = {worker_id: 0 for worker_id in worker_id_to_wage.keys()}

    for period in range(num_periods):
        period_tasks = task_ids[period * num_workers : (period + 1) * num_workers]

        # Sort tasks from big to small
        period_tasks = sorted(
            period_tasks,
            key=lambda task_id: task_id_to_task_size[task_id],
            reverse=True,
        )

        # Sort workers from low earnings to high earnings
        sorted_workers = sorted(
            earnings.keys(),
            key=lambda worker_id: earnings[worker_id],
        )

        for worker_id, task_id in zip(sorted_workers, period_tasks):
            allocation[task_id] = worker_id
            earnings[worker_id] += (
                task_id_to_task_size[task_id] * worker_id_to_wage[worker_id]
            )

    return allocation


def compute_greedy_max_efficiency_alloc(
    task_id_to_task_size: dict[str, float],
    task_ids: list[str],
    num_periods: int,
    num_workers: int,
    worker_id_to_productivities: dict[str, float],
    num_explore_periods: int,
    my_random: np.random.RandomState,
) -> dict[str, str]:
    """
    Returns an allocation by a reasonably smart algorithm that is trying to maximize efficiency, but doesn't know the tasks in advance.

    Specifically: for first num_explore_periods periods, it plays randomly, and for the rest, it correctly matches the larger tasks to more productive workers.
    Setting num_explore_periods = 0 computes the actual maximally efficient allocation.
    """
    alloc = {}
    # Pre-sort workers once: descending order of productivity.
    sorted_workers = sorted(
        worker_id_to_productivities.items(), key=lambda x: x[1], reverse=True
    )
    sorted_worker_ids = [worker_id for worker_id, _ in sorted_workers]

    for period in range(num_periods):
        start = period * num_workers
        end = start + num_workers
        period_tasks = task_ids[start:end]

        if period < num_explore_periods:
            # Exploration phase: assign tasks randomly.
            # Copy worker ids and tasks to shuffle them.
            worker_ids_copy = sorted_worker_ids.copy()
            tasks_copy = period_tasks.copy()
            my_random.shuffle(worker_ids_copy)
            my_random.shuffle(tasks_copy)
            for worker_id, task_id in zip(worker_ids_copy, tasks_copy):
                alloc[task_id] = worker_id
        else:
            # Exploitation phase: sort tasks in descending order of task_size.
            sorted_tasks = sorted(
                period_tasks,
                key=lambda task_id: task_id_to_task_size[task_id],
                reverse=True,
            )
            for worker_id, task_id in zip(sorted_worker_ids, sorted_tasks):
                alloc[task_id] = worker_id

    return alloc
