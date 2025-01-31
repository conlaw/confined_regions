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
class RecourseVerifier:

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
    def build_mip(self, x, feasibility=False):
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
        A = []
        b = []
        m = gp.Model('linear_verifier')
        n = len(self.action_set)
        u_vars = []
        l_vars = []

        for idx, aj in enumerate(self.action_set):
            # Set upper and lower bounds for x
            ub = aj.get_data_bound('ub')
            lb = aj.get_data_bound('lb')

            vtype = aj.variable_type
            #ub row
            x_row = np.zeros(n)
            a_row = np.zeros(n)
            
            x_row[idx] = 1
            X.append(x_row)
            A.append(a_row)

            #lb row
            x_row = np.zeros(n)
            a_row = np.zeros(n)
            
            x_row[idx] = -1
            X.append(x_row)
            A.append(a_row)

            b.append(ub)
            b.append(-lb)
            
            # Set upper and lower bounds for a
            ub = aj.get_action_bound(None,'ub')
            lb = aj.get_action_bound(None,'lb')

            #ub row
            x_row = np.zeros(n)
            a_row = np.zeros(n)
            
            a_row[idx] = 1
            X.append(x_row)
            A.append(a_row)

            #lb row
            x_row = np.zeros(n)
            a_row = np.zeros(n)
            
            a_row[idx] = -1
            X.append(x_row)
            A.append(a_row)

            b.append(ub)
            b.append(-lb)

            # Set upper and lower bounds for x+a
            ub = aj.get_data_bound('ub')*1.
            lb = aj.get_data_bound('lb')*1.
            x_row = np.zeros(n)
            a_row = np.zeros(n)
            
            x_row[idx] = 1
            a_row[idx] = 1
            X.append(x_row)
            A.append(a_row)

            #lb row
            x_row = np.zeros(n)
            a_row = np.zeros(n)
            x_row[idx] = -1
            a_row[idx] = -1
            X.append(x_row)
            A.append(a_row)
            b.append(ub)
            b.append(-lb)

        for con in self.action_set.constraints:
            X, A, b = con.add_to_gurobi(X, A, b)

        neg_coef = np.clip(self.coefficients, None, 0)
        pos_coef = np.clip(self.coefficients, 0, None)

        x_class_row = -self.coefficients
        a_class_row = -self.coefficients
        X.append(x_class_row)
        A.append(a_class_row)
        b.append(self.intercept)
        A = np.array(A)
        X = np.array(X)
        self.A = A
        self.X = X
        self.b = b

        a_ub = self.action_set.get_bounds(None,'ub')
        a_lb = self.action_set.get_bounds(None,'lb')
        var_types = self.action_set.get_variable_types()

        a_var = m.addMVar(shape = A.shape[1], vtype = gp.GRB.CONTINUOUS, lb = a_lb, ub = a_ub, name='x')
        for idx, vtype in enumerate(var_types):
            a_var[idx].VType = var_dict[vtype]

        b = np.array(b)
        x = np.array(x)

        #rhs = b - X @ x
        if feasibility:
            x_ub = self.action_set.get_population_bounds('ub')
            x_lb = self.action_set.get_population_bounds('lb')
            var_types = self.action_set.get_variable_types()

            x_var = m.addMVar(shape = A.shape[1], vtype = gp.GRB.CONTINUOUS, lb = x_lb, ub = x_ub, name='x')
            for idx, vtype in enumerate(var_types):
                a_var[idx].VType = var_dict[vtype]
            b = np.array(b)
            x = np.array(x)
            m.addConstr(X @ x_var <= b)
            m.setObjective(0)
            return m, x_var

        else:
            a_ub = self.action_set.get_bounds(None,'ub')
            a_lb = self.action_set.get_bounds(None,'lb')
            var_types = self.action_set.get_variable_types()
            a_var = m.addMVar(shape = A.shape[1], vtype = gp.GRB.CONTINUOUS, lb = a_lb, ub = a_ub, name='x')
            for idx, vtype in enumerate(var_types):
                a_var[idx].VType = var_dict[vtype]
            b = np.array(b)
            x = np.array(x)

            m.addConstr(A @ a_var <= b- X @ x)
            m.setObjective(0)

            return m, a_var
        

    def load_model(self, ml_model):
        """
        Load ml model that is to be verified
        - Right now only support scikit-learn logistic regression
        """
        assert isinstance(ml_model, LogisticRegression)
        #todo if model already loaded then remove previous constraints

        self.coefficients = ml_model.coef_[0]
        #coefficients = np.zeros(len(self.action_set))
        #coefficients[0] = 1
        self.intercept = ml_model.intercept_[0]
        #intercept = 1
        self.model_loaded = True

        return
    
    def load_model_manual(self, coefficients, intercept):
        """
        Load ml model that is to be verified
        - Right now only support scikit-learn logistic regression
        """
        #todo if model already loaded then remove previous constraints

        self.coefficients = coefficients
        self.intercept = intercept
        self.model_loaded = True

        return
    
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


    def verify(self, x, print_flag = True):
        """
        Verify the model
        Returns:
        - Status: True if certify recourse for entire population, False otherwise
        - If False, counterexample (data point or region)
        """
        assert self.model_loaded, "Model not loaded"
        m, a = self.build_mip(x)
        self.m = m
        m.setParam('OutputFlag', print_flag)
        m.optimize()
        if m.Status == gp.GRB.INFEASIBLE:
            if print_flag: print("Data point has no recourse.")
            return False, None
        else:
            if print_flag: print("Data point has recourse.")
            if print_flag: tabulate_datapoint(self.action_set, x, [var.X for var in a])
            return True, [var.X for var in a]


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

