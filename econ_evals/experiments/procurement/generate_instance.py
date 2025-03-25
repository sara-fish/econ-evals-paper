import numpy as np
from econ_evals.experiments.procurement.opt_solver import Menu, Entry, evaluate_alloc

ENTRY_TYPES = ["basic", "two_part_tariff", "bulk_discount"]
MIN_PRICE = 1
MAX_PRICE = 20
MIN_MIN_QUANTITY = 2
MAX_MIN_QUANTITY = 10


def generate_instance(
    num_inputs: int,
    num_alternatives_per_input: int,
    num_entries: int,
    my_random: np.random.RandomState = None,
    NUM_ITEMS_PER_ENTRY_P: float = 0.8,
    QUANTITY_PER_ITEM_P: float = 0.5,
    OFFER_QTY_IN_SAMPLE_BUNDLE_P: float = 0.5,
    MIN_EFFECTIVENESS: int = 1,
    MAX_EFFECTIVENESS: int = 1,
) -> tuple[Menu, float, list[list[str]], dict[str, int], dict[str, int]]:
    """
    Generate a menu to use for a procurement problem

    num_inputs: for example if A,B,C then this is 3
    num_alterantives_per_input: for example if A1, A2, B1, B2, C1, C2 then this is 2
    num_entries: total number of entries in the menu (# offers)
    my_random: fixed randomness
    NUM_ITEMS_PER_ENTRY_P:
        Geom(NUM_ITEMS_PER_ENTRY_P) is the law of the # of distinct items each entry gives you
        e.g. if an entry gives 2A1 + 3A2, then this wouldbe 2, because there's 2 distinct items
        Lower = more items per entry (so more bundles in an entry and less singles)
    QUANTITY_PER_ITEM_P:
        Geom(QUANTITY_PER_ITEM_P) is the law of the quantity of each individual item in an entry
        e.g. if an entry gives 2A1, then this would be 2
        Lower = how many of each item included in bundle is higher (e.g. more likely to have 10 A1)
    OFFER_QTY_IN_SAMPLE_BUNDLE_P:
        Geom(OFFER_QTY_IN_SAMPLE_BUNDLE_P) is the law of how much of each offer to buy in the sample alloc
        (that determines the budget)
        e.g. if this is 3, then you buy 3 of Offer_4 in the sample alloc
        Lower = sample alloc has more stuff in it -> budget is bigger -> benchmark is generally harder

    Return:
    - menu, budget, item_groups, start_alloc, item_to_effectiveness
    """

    assert num_entries >= num_inputs * num_alternatives_per_input

    if my_random is None:
        print("Warning: instance generation using global fixed seed")
        my_random = np.random.RandomState(0)

    assert num_inputs >= 1 and num_inputs <= 26

    item_ids = [
        f"{chr(65 + i)}{j+1}"
        for i in range(num_inputs)
        for j in range(num_alternatives_per_input)
    ]

    # Sample effectiveness scores
    item_to_effectiveness = {
        item: int(my_random.choice(range(MIN_EFFECTIVENESS, MAX_EFFECTIVENESS + 1)))
        for item in item_ids
    }

    entries_types = [
        str(entry_type) for entry_type in my_random.choice(ENTRY_TYPES, num_entries)
    ]  # convert from np array

    # num items each entry gives you... we want this to be mostly 1, but sometimes more
    entries_num_items = my_random.geometric(NUM_ITEMS_PER_ENTRY_P, num_entries)

    # For each entry, which item it must include (we do this permutation thing to make sure
    # every item appears at least once)
    entries_item_must_include_start = my_random.permutation(item_ids)
    assert set(entries_item_must_include_start) == set(item_ids)

    entries_item_must_include_rest = my_random.permutation(
        item_ids * (num_entries // len(item_ids))
    )[: num_entries - len(item_ids)]
    entries_item_must_include = my_random.permutation(
        list(entries_item_must_include_start) + list(entries_item_must_include_rest)
    )

    assert set(entries_item_must_include) == set(item_ids)

    # Determine contents of each entry
    entries_contents = []
    for entry_num_items, entry_item_must_include in zip(
        entries_num_items, entries_item_must_include
    ):
        other_items = my_random.choice(item_ids, entry_num_items - 1)
        items = [entry_item_must_include] + list(other_items)
        quantities = my_random.geometric(QUANTITY_PER_ITEM_P, entry_num_items)
        contents = {
            str(item): int(quantity) for item, quantity in zip(items, quantities)
        }
        entries_contents.append(contents)

    # Determine costs of each entry

    entry_cost_args = []
    for entry_type in entries_types:
        if entry_type == "basic":
            entry_cost_args.append(
                {
                    "cost": round(my_random.uniform(MIN_PRICE, MAX_PRICE), 2),
                }
            )
        elif entry_type == "two_part_tariff":
            entry_cost_args.append(
                {
                    "fixed_cost": round(my_random.uniform(MIN_PRICE, MAX_PRICE), 2),
                    "variable_cost": round(my_random.uniform(MIN_PRICE, MAX_PRICE), 2),
                }
            )
        elif entry_type == "bulk_discount":
            entry_cost_args.append(
                {
                    "cost": round(my_random.uniform(MIN_PRICE, MAX_PRICE), 2),
                    "min_quantity": my_random.randint(
                        MIN_MIN_QUANTITY, MAX_MIN_QUANTITY
                    ),
                }
            )
        else:
            raise NotImplementedError

    entries_ids = [f"Offer_{i}" for i in range(1, num_entries + 1)]

    assert (
        len(entries_ids)
        == len(entries_types)
        == len(entry_cost_args)
        == len(entries_contents)
    )

    entries = [
        Entry(id=entry_id, type=entry_type, **entry_arg, contents=entry_contents)
        for entry_id, entry_type, entry_arg, entry_contents in zip(
            entries_ids, entries_types, entry_cost_args, entries_contents
        )
    ]

    menu = Menu(entries=entries)

    # Group A1, A2 together, B1, B2 together, etc
    item_groups = [
        [f"{chr(65 + i)}{j+1}" for j in range(num_alternatives_per_input)]
        for i in range(num_inputs)
    ]

    # To determine budget, pick random alloc that includes something of each item group (A, B, C..)
    # See the cost of that alloc, and then set budget to be that plus a little bit
    # (that way, positive utility is feasible)

    necessary_items = []
    for group in item_groups:
        necessary_items.append(my_random.choice(group))
    entries_with_necessary_items = []
    for item in necessary_items:
        entries_with_item = []
        for entry in entries:
            if item in entry.contents:
                entries_with_item.append(entry)
        entries_with_necessary_items.append(my_random.choice(entries_with_item))
    purchase_quantities = my_random.geometric(
        OFFER_QTY_IN_SAMPLE_BUNDLE_P, len(necessary_items)
    )

    # For all the entries that have minimum quantities, make sure they're hit
    for idx, entry in enumerate(entries_with_necessary_items):
        if entry.type == "bulk_discount":
            # Current value is sampled from Geom(OFFER_QTY_IN_SAMPLE_BUNDLE_P),
            # so some positive integer
            purchase_quantities[idx] += entry.min_quantity - 1
            assert purchase_quantities[idx] >= entry.min_quantity

    assert (
        len(necessary_items)
        == len(purchase_quantities)
        == len(entries_with_necessary_items)
    )

    start_alloc = {
        entry.id: int(quantity)
        for entry, quantity in zip(entries_with_necessary_items, purchase_quantities)
    }

    _, _, total_cost, total_utility = evaluate_alloc(
        menu=menu,
        alloc=start_alloc,
        item_groups=item_groups,
        item_to_effectiveness=item_to_effectiveness,
        budget=0,  # temp value
        agg_type="prod",  # temp value
        group_weights=[1 / len(item_groups) for _ in item_groups],  # temp value
    )

    assert total_utility > 0
    assert total_cost > 0

    # We add a litte bit extra -- positive utility is still feasible, but now
    # LLM won't think it solved it exactly when it finds this particular alloc
    budget = round(total_cost + round(np.random.uniform(0, 1), 2), 2)

    return menu, budget, item_groups, start_alloc, item_to_effectiveness
