from typing import Literal, Optional
from pydantic import BaseModel, model_validator
import math
import gurobipy as gp
import inflect

MAX_SOLS = 1e9

"""
Nomenclature:
- ENTRY is something like "$2 for a A1+A2 bundle"
- ENTRY_ID is a unique ID corresponding to an entry, like "A1_basic" or "A1_A2_bundle"
- MENU is collection of entries
- ITEM is "A1" or "A2" (items are themselves just these strs, they don't have IDs or any properties)
- ITEM GROUPS are [["A1", "A2"], ["B1", "B2"]]
- ALLOC is a dict mapping ENTRY_IDs to quantities (how much of each to buy)
    -> which, in turn, can induce a mapping from ITEMs to quantities, but we only work with this thing implicitly
"""


def alloc_to_str(alloc):
    printed_items = []
    for item, qty in alloc.items():
        if qty > 0:
            printed_items.append("{qty} x {item} ".format(qty=qty, item=item))
    return " + ".join(printed_items)


class Entry(BaseModel):
    id: str
    contents: dict[str, int]
    type: Literal["basic", "two_part_tariff", "bulk_discount"]
    cost: float | None = None
    fixed_cost: float | None = None
    variable_cost: float | None = None
    min_quantity: int | None = None

    @model_validator(mode="before")
    def check_required_fields(cls, values):
        entry_type = values["type"]

        if entry_type == "basic":
            assert (
                values.get("cost")
                and not values.get("fixed_cost")
                and not values.get("variable_cost")
                and not values.get("min_quantity")
            )

        elif entry_type == "two_part_tariff":
            assert (
                values.get("fixed_cost")
                and values.get("variable_cost")
                and not values.get("cost")
                and not values.get("min_quantity")
            )

        elif entry_type == "bulk_discount":
            assert (
                values.get("cost")
                and values.get("min_quantity")
                and not values.get("fixed_cost")
                and not values.get("variable_cost")
            )

        return values

    def to_str(self) -> str:
        # contents is like: {"A1": 2, "A2": 1}
        # contents_list is like: ["2 units of A1", "1 unit of A2"]
        p = inflect.engine()
        contents_list = [
            f"{qty} {p.plural('unit', qty)} of {id}".format(qty=qty, id=id)
            for id, qty in self.contents.items()
        ]
        # contents_str = 2 units of A1 and 1 unit of A2 (or comma-separated for 3+ items)
        contents_str = p.join(contents_list)

        entry_id = self.id

        if self.type == "basic":
            return f"{entry_id}: ${self.cost:.2f} for {contents_str}"
        elif self.type == "two_part_tariff":
            return f"{entry_id}: [additional upfront cost ${self.fixed_cost:.2f}] ${self.variable_cost:.2f} for {contents_str}"
        elif self.type == "bulk_discount":
            return f"{entry_id}: [minimum order quantity {self.min_quantity}] ${self.cost:.2f} for {contents_str}"
        else:
            raise NotImplementedError

    def get_cost(self, quantity: int) -> tuple[float, str]:
        if self.type == "basic":
            return self.cost * quantity, ""
        elif self.type == "two_part_tariff":
            return (
                self.fixed_cost + self.variable_cost * quantity if quantity > 0 else 0,
                "",
            )
        elif self.type == "bulk_discount":
            if quantity > 0 and quantity < self.min_quantity:
                return (
                    0,
                    f"Quantity {quantity} for {self.id} is insufficient (miniumum {self.min_quantity} necessary)",
                )
            return self.cost * quantity, ""
        else:
            raise NotImplementedError


class Menu:
    def __init__(self, entries: list[Entry]):
        self.id_to_entries = {}
        for entry in entries:
            if entry.id in self.id_to_entries:
                raise ValueError(f"Duplicate id {entry.id}")
            self.id_to_entries[entry.id] = entry

    def __getitem__(self, id: str) -> Entry:
        assert id in self.id_to_entries
        return self.id_to_entries[id]

    def __iter__(self):
        return iter(self.id_to_entries.values())

    def __len__(self):
        return len(self.id_to_entries)

    def to_dict(self) -> dict:
        d = {}
        for id, entry in self.id_to_entries.items():
            entry_dict = {
                "type": entry.type,
                "contents": entry.contents,
            }
            if entry.type == "basic":
                entry_dict["cost"] = entry.cost
            elif entry.type == "two_part_tariff":
                entry_dict["fixed_cost"] = entry.fixed_cost
                entry_dict["variable_cost"] = entry.variable_cost
            elif entry.type == "bulk_discount":
                entry_dict["cost"] = entry.cost
                entry_dict["min_quantity"] = entry.min_quantity
            else:
                raise NotImplementedError
            d[id] = entry_dict
        return d

    def to_str(self) -> str:
        s = ""
        for entry in self.id_to_entries.values():
            s += "- " + entry.to_str() + "\n"
        return s

    def get_max_var_value(self, budget: float) -> int:
        """
        Given a menu and a budget, return some number such that setting any entry quantity to be that value is infeasible under the budget.
        (In particular, this serves as an upper bound for all the entry quantities in the ILP solver.)
        """
        max_var_value = 0
        for entry in self:
            if entry.type == "basic":
                max_var_value = max(max_var_value, budget / entry.cost)
            elif entry.type == "two_part_tariff":
                max_var_value = max(max_var_value, budget / entry.variable_cost)
            elif entry.type == "bulk_discount":
                max_var_value = max(max_var_value, budget / entry.cost)
            else:
                raise NotImplementedError
        return math.ceil(max_var_value) + 1

    def get_items(self) -> list[str]:
        """
        Return list of all individual items that are in the menu (like A1, A2, A3, ...., separating bundles)
        """
        items = set()
        for entry in self:
            for id in entry.contents:
                items.add(id)
        return sorted(items)


def evaluate_alloc(
    menu: Menu,
    alloc: dict[str, int],
    item_groups: list[list[str]],
    item_to_effectiveness: dict[str, int],
    budget: float,
    agg_type: Literal["min", "prod"],
    group_weights: list[float],
) -> tuple[bool, str, float, float]:
    """
    Given a menu, an allocation and a budget, return whether:
    - the allocation is feasible (i.e., total cost <= budget,
      and hitting purchase requirement [e.g. hitting min spend for bulk discount])
    - invalid_reason
    - the total cost of the allocation
    - the total utility from the allocation (output of production function)
    """

    if agg_type == "prod":
        assert (
            sum(group_weights) == 1
        )  # maybe replace with more floating point robust thing?

    # Check that the items in menu match the item_groups, and also that alloc is subset of these
    menu_items = menu.get_items()
    item_groups_items = [item for group in item_groups for item in group]

    assert set(menu_items) == set(item_groups_items)
    assert len(item_groups) == len(group_weights)
    assert set(item_groups_items) == set(item_to_effectiveness.keys())

    # Next calculate cost of alloc
    entry_id_to_cost = {}
    is_feasible = True
    invalid_reason = ""

    for entry_id, quantity in alloc.items():
        entry = menu[entry_id]
        entry_cost, invalid_reason_temp = entry.get_cost(quantity)
        if invalid_reason_temp:
            is_feasible = False
            invalid_reason += invalid_reason_temp + "\n"
        entry_id_to_cost[entry_id] = entry_cost

    # Now calculate quantity of each item based on alloc
    item_to_quantity = {item: 0 for item in menu_items}

    for entry_id, quantity in alloc.items():
        entry = menu[entry_id]
        for item, item_quantity in entry.contents.items():
            item_to_quantity[item] += item_quantity * quantity

    total_cost = sum(entry_id_to_cost.values())

    # Final possible way to be invalid is if out of budget
    if total_cost > budget:
        is_feasible = False
        invalid_reason += f"Total cost {total_cost:.2f} exceeds budget {budget:.2f}\n"

    # Finally calculate total utility

    if agg_type == "min":
        total_utility = min(
            [
                sum(
                    item_to_quantity[item] * item_to_effectiveness[item]
                    for item in group
                )
                * group_weight
                for group_weight, group in zip(group_weights, item_groups)
            ]
        )

    elif agg_type == "prod":
        total_utility = 1
        for group_weight, group in zip(group_weights, item_groups):
            total_utility *= (
                sum(
                    item_to_quantity[item] * item_to_effectiveness[item]
                    for item in group
                )
                ** group_weight
            )

    else:
        raise NotImplementedError(f"Unknown agg_type {agg_type}")

    return is_feasible, invalid_reason, total_cost, total_utility


def compute_opt(
    menu: Menu,
    budget: float,
    item_groups: list[list[str]],
    item_to_effectiveness: dict[str, int],
    group_weights: list[float],
    agg_type: Literal["min", "prod"],
    start_alloc: Optional[dict[str, int]] = None,
) -> tuple[dict[str, int], dict[str, object]]:
    """
    This function uses Gurobi
    menu: Menu object, contains entries that describe prices & effectiveness for each item
    budget: float, the total budget available
    item_groups: this specifies the objective function (for now).
                     If you put groups [["a1", "a2"], ["b1", b2"]], then the objective function is min(a1+a2, b1+b2)
    item_to_effectiveness: dict[str, int], effectiveness of each item
    group_weights: global weights for each of the grouped items
                   if min obj: min( w_A * A, w_B * B, ...)
                   if prod obj: A^w_A * B^w_B * ... (Cobb douglass)
                   if min: don't enforce normalization
                   if prod: require that these sum to 1
    start_alloc: dict[str, int], optional, initial allocation to start from

    Return:
    dict[str, int]: a dictionary mapping entry id to the quantity to purchase
    dict: some logging
    """

    if start_alloc is not None:
        is_feasible, _, _, _ = evaluate_alloc(
            menu=menu,
            alloc=start_alloc,
            item_groups=item_groups,
            item_to_effectiveness=item_to_effectiveness,
            budget=budget,
            agg_type=agg_type,
            group_weights=group_weights,
        )
        assert is_feasible, "start_alloc must be feasible"

    if agg_type == "prod":
        assert (
            sum(group_weights) == 1
        )  # maybe replace with more floating point robust thing?

    env = gp.Env(empty=True)
    env.setParam("OutputFlag", 0)  # don't print license info
    env.start()

    model = gp.Model(env=env)

    max_var_value = menu.get_max_var_value(budget)

    # Make vars for each entry -- this is the main thing we are solving for (what to buy)
    vars = {}
    for entry in menu:
        vars[entry.id] = model.addVar(vtype=gp.GRB.INTEGER, name=entry.id)

    aux_vars = {}
    # Additional constraints on quantity
    # (these can be ORs, which make it a ILP and not a LP)
    for entry in menu:
        var_name = entry.id
        if entry.type == "bulk_discount":
            # Quantity can be 0, or min_quantity
            var_name_binary = f"{var_name}_binary"
            aux_vars[var_name_binary] = model.addVar(
                vtype=gp.GRB.BINARY, name=var_name_binary
            )
            model.addConstr(
                vars[var_name] >= entry.min_quantity * aux_vars[var_name_binary],
                f"{var_name}_lower_bound",
            )
            model.addConstr(
                vars[var_name] <= max_var_value * aux_vars[var_name_binary],
                f"{var_name}_upper_bound",
            )
        if entry.type == "two_part_tariff":
            # Quantity can be 0, or >= 1 (matters for cost computation, whether to pay fixed_cost)
            var_name_binary = f"{var_name}_binary"
            aux_vars[var_name_binary] = model.addVar(
                vtype=gp.GRB.BINARY, name=var_name_binary
            )
            model.addConstr(
                vars[var_name] >= aux_vars[var_name_binary],
                f"{var_name}_opt_in",
            )
            model.addConstr(
                vars[var_name] <= max_var_value * aux_vars[var_name_binary],
                f"{var_name}_opt_out",
            )

    cost_vars = {}
    # Compute the costs of each entry
    for entry in menu:
        var_name = entry.id
        cost_vars[var_name] = model.addVar(
            vtype=gp.GRB.CONTINUOUS, name=f"{var_name}_cost"
        )
        if entry.type == "basic":
            model.addConstr(
                cost_vars[var_name] == entry.cost * vars[var_name],
                f"{var_name}_cost",
            )
        elif entry.type == "two_part_tariff":
            # Cost is fixed + var * quantity only if quantity > 0, otherwise it's just 0
            model.addConstr(
                cost_vars[var_name]
                == entry.fixed_cost * aux_vars[f"{var_name}_binary"]
                + entry.variable_cost * vars[var_name],
                f"{var_name}_cost",
            )
        elif entry.type == "bulk_discount":
            model.addConstr(
                cost_vars[var_name] == entry.cost * vars[var_name],
                f"{var_name}_cost",
            )
        else:
            raise NotImplementedError(f"Unknown type {entry.type}")

    # Budget constraint: sum of costs <= budget
    total_cost = model.addVar(vtype=gp.GRB.CONTINUOUS, name="total_cost")
    model.addConstr(
        total_cost == gp.quicksum(cost_var for _, cost_var in cost_vars.items()),
        "total_cost",
    )
    model.addConstr(
        total_cost <= budget,
        "budget_constraint",
    )

    # Item vars: calculate the quantities of each indiv item A1, A2, ... from the entry quantities
    item_vars = {}
    for item in menu.get_items():
        item_vars[item] = model.addVar(vtype=gp.GRB.CONTINUOUS, name=item)

        item_terms = []
        for entry in menu:
            if item in entry.contents:
                item_terms.append(entry.contents[item] * vars[entry.id])

        model.addConstr(
            item_vars[item] == gp.quicksum(item_terms),
            f"{item}_quantity",
        )

    # finally aggregate item_var into objective function

    if agg_type == "min":
        # Objective function: for now hardcoded to be min( sum of some items, sum of some other items, ...)
        aux_vars["obj_min"] = model.addVar(vtype=gp.GRB.INTEGER, name="obj_min")

        for group_weight, group in zip(group_weights, item_groups):
            group_str = "_".join(group)
            model.addConstr(
                aux_vars["obj_min"]
                <= gp.quicksum(
                    item_vars[item] * item_to_effectiveness[item] for item in group
                )
                * group_weight,
                f"obj_min_{group_str}",
            )

        model.setObjective(aux_vars["obj_min"], gp.GRB.MAXIMIZE)

    elif agg_type == "prod":
        # Under the hood, we maximize the log

        # For future reference: the trick is to just make variables for
        # absolutely everything, because you can take logs of Var objects,
        # but not LinExpr objects

        obj_vars = {}

        # First make vars for A1+A2+.., B1+B2+...
        for group in item_groups:
            group_str = "_".join(group)
            obj_vars[f"sum_{group_str}"] = model.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"sum_{group_str}"
            )

            model.addConstr(
                obj_vars[f"sum_{group_str}"]
                == gp.quicksum(
                    item_vars[item] * item_to_effectiveness[item] for item in group
                ),
                f"sum_{group_str}",
            )

        # Now make vars for log(A1+A2+..), log(B1+B2+...)
        for group in item_groups:
            group_str = "_".join(group)
            obj_vars[f"log_sum_{group_str}"] = model.addVar(
                vtype=gp.GRB.CONTINUOUS, name=f"log_sum_{group_str}"
            )

            # model.addGenConstrLog(x,y) means y = logx
            model.addGenConstrLog(
                obj_vars[f"sum_{group_str}"],
                obj_vars[f"log_sum_{group_str}"],
                f"log_{group_str}",
            )

        # Finally make var that is sum of the logs

        obj_vars["obj"] = model.addVar(vtype=gp.GRB.CONTINUOUS, name="obj")

        model.addConstr(
            obj_vars["obj"]
            == gp.quicksum(
                obj_vars[f"log_sum_{'_'.join(group)}"] * group_weight
                for group_weight, group in zip(group_weights, item_groups)
            ),
            "obj",
        )

        model.setObjective(obj_vars["obj"], gp.GRB.MAXIMIZE)

    else:
        raise NotImplementedError(f"Unknown agg_type {agg_type}")

    # Warm start with start_alloc
    if start_alloc is not None:
        for entry_id, quantity in start_alloc.items():
            vars[entry_id].Start = quantity

    model.setParam("PoolSearchMode", 2)
    model.setParam("PoolSolutions", MAX_SOLS)
    model.setParam("PoolGap", 0.0)
    model.setParam("MIPGap", 0.0)
    model.setParam("SolutionLimit", MAX_SOLS)

    try:
        model.optimize()
    except gp.GurobiError as e:
        num_vars = model.getAttr("NumVars")
        print(
            f"ERROR: num_vars={num_vars} too large (max for unrestricted license is 200). Reduce instance size."
        )
        raise e

    if model.status == gp.GRB.INFEASIBLE and agg_type == "prod":
        # Have to manually return 0 alloc in this case
        # (in the min case, it actually computes the 0 alloc, and we don't have to manually specify),
        # but here it can't because of the log constraints
        alloc = {entry_id: 0 for entry_id in vars}
        log = {
            "opt_value": 0,
            "total_cost": 0,
            "total_utility": 0,
            "aux_vars": {},
            "num_solutions": 0,
            "all_solutions": [],
        }
    elif model.status != gp.GRB.OPTIMAL:
        raise ValueError(f"Optimization failed with status {model.status}")
    else:
        alloc = {entry_id: int(var.x) for entry_id, var in vars.items()}
        # Also get lots of other info for logging
        log = {}
        log["opt_value"] = model.objVal
        log["total_cost"] = total_cost.x
        if agg_type == "min":
            log["total_utility"] = model.objVal
        elif agg_type == "prod":
            log["total_utility"] = math.exp(model.objVal)
        else:
            raise NotImplementedError
        log["aux_vars"] = {v.varName: v.x for v in model.getVars()}
        log["num_solutions"] = model.SolCount
        log["all_solutions"] = []
        for i in range(model.SolCount):
            model.params.SolutionNumber = i
            solution = {v.varName: v.Xn for v in model.getVars()}
            log["all_solutions"].append(solution)

    return alloc, log
