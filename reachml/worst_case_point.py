import numpy as np
from cplex import Cplex, SparsePair
import gurobipy as gp
from functools import reduce
from itertools import chain, product
from .action_set import ActionSet
from .reachable_set import ReachableSet
from .cplex_utils import (
    combine,
    get_cpx_variable_types,
    get_cpx_variable_args,
    has_solution,
    CplexGroupedVariableIndices,
    set_mip_time_limit,
    set_mip_node_limit,
    get_mip_stats,
)
from sklearn.linear_model import LogisticRegression
import subprocess
from prettytable import PrettyTable

var_dict = {
    bool: gp.GRB.INTEGER,
    int: gp.GRB.INTEGER,
    float: gp.GRB.CONTINUOUS
}
SCORE_SCALING = 10
class WorstCasePoint:

    SETTINGS = {
        "eps_min": 0.5,
        # todo: set MIP parameters here
    }

    def __init__(self, action_set, print_flag=False, **kwargs):
        """
        :param action_set:
        :param x:
        :param part:
        :param print_flag:
        :param kwargs:
        """
        assert isinstance(action_set, ActionSet)
        self._action_set = action_set

        # set actionable indices
        self.actionable_indices = list(range(len(action_set)))

        # parse remaining settings
        self.print_flag = print_flag

        # build base MIP
        self.model_loaded = False
        
        return 

    @property
    def action_set(self):
        return self._action_set
            

    ### MIP Functions ###
    def build_mip(self, w, feasibility=False):
        """
        build CPLEX mip object of actions
        :return: `cpx` Cplex MIP Object
                 `indices` CplexGroupVariableIndices
        ----
        Variables
        ----------------------------------------------------------------------------------------------------------------
        name                  length              type        description
        ----------------------------------------------------------------------------------------------------------------
        a[j]                  d x 1               real        action on variable j
        a_pos[j]              d x 1               real        absolute value of a[j]
        a_neg[j]              d x 1               real        absolute value of a[j]
        """

        X  = []
        b = []
        m = gp.Model('linear_verifier')
        n = len(self.action_set)

        for con in self.action_set.population_constraints:
            X, _, b = con.add_to_gurobi(X, [], b)

        X = np.array(X)
        b = np.array(b)
        self.X = X
        self.b = b

        x_ub = self.action_set.get_population_bounds('ub')
        x_lb = self.action_set.get_population_bounds('lb')
        var_types = self.action_set.get_variable_types()

        x_var = m.addMVar(shape = X.shape[1], vtype = gp.GRB.CONTINUOUS, lb = x_lb, ub = x_ub, name='x')
        for idx, vtype in enumerate(var_types):
            x_var[idx].VType = var_dict[vtype]
        m.addConstr(X @ x_var <= b)
        m.setObjective(w @ x_var, gp.GRB.MAXIMIZE)
        return m, x_var

        

    
    #def verify_individual(self, x):
    def check_feasibility(self, x):
        m, _ = self.build_mip(x, feasibility=True)
        self.m = m
        m.setParam('OutputFlag', False)
        m.optimize()
        if m.Status == gp.GRB.INFEASIBLE:
            return False
        else:
            return True


    def generate(self, w, print_flag = False):
        """
        Verify the model
        Returns:
        - Status: True if certify recourse for entire population, False otherwise
        - If False, counterexample (data point or region)
        """
        m, x = self.build_mip(w)
        self.m = m
        m.setParam('OutputFlag', print_flag)
        m.optimize()
        if m.Status == gp.GRB.INFEASIBLE:
            if print_flag: print("Action set infeasible.")
            return False, None
        else:
            return True, np.array([var.X for var in x])


def tabulate_datapoint(action_set, x, a):
    """
    prints a table with information about each element in the action set
    :param action_set: ActionSet object
    :return:
    """
    # fmt:off
    TYPES = {bool: "<bool>", int: "<int>", float: "<float>"}
    FMT = {bool: "1.0f", int: "1.0f", float: "1.2f"}
    t = PrettyTable()
    vtypes = [TYPES[v] for v in action_set.variable_type]
    t.add_column("", list(range(len(action_set))), align="r")
    t.add_column("name", action_set.name, align="l")
    t.add_column("type", vtypes, align="c")
    t.add_column("x", np.round(x,2), align="c")
    t.add_column("a", np.round(a,2), align="c")
    print(t)
    return

