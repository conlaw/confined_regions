import numpy as np
from cplex import Cplex, SparsePair
from .abstract import ActionabilityConstraint
from ..utils import parse_attribute_name

class PopulationConstraint(ActionabilityConstraint):
    """
    Constraint to ensure that actions preserve one-hot encoding of a categorical
    attribute. This constraint should be specified over a collection of Boolean
    features produced through a one-hot encoding of an categorical attribute Z.

    Given an categorical attribute Z with `m` categories: `z[0], z[1], .. z[m-1]`,
    the boolean features - i.e., dummies - have the form:

      x[0] := 1[Z = z[0]]
      x[1] := 1[Z = z[1]]
      ...
      x[m-1] := 1[Z = z[m-1]]

    Here z[0], ... z[m-1] denote the different values that Z can take
    and x[k] := 1[Z = k] is a dummy variable set to 1 if and only if Z == k.

    todo: Example:

    """
    VALID_LIMIT_TYPES = ('equal', 'max')
    def __init__(self, names, coefficients, rhs, sense, parent = None):
        """
        :param self:
        :param action_set:
        :param names:
        :param limit: integer value representing number of
        :param limit_type: either `equal` or `max` (at most limit)
        :return:
        """
        self.coefficents = coefficients
        self.rhs = rhs
        self.sense = sense
        self._parameters = ('coefficients', 'rhs', 'sense')
        super().__init__(names = names, parent = parent)

    @property
    def limit(self):
        return self._limit

    @property
    def limit_type(self):
        return self._limit_type
    
    def add_to_gurobi(self, X, A, b):
        """
        adds constraint to Population Verifier
        :param cpx: Cplex object
        :return: x, a row and rhs
        """
        row = np.zeros(len(self.parent))
        row[self.indices] = self.coefficients

        if self.sense == "LE":
            X.append(row)
            A.append(row)
            b.append(self.rhs)
        elif self.sense == "GE":
            X.append(-row)
            A.append(-row)
            b.append(-self.rhs)
        elif self.sense == "EQ":
            X.append(row)
            A.append(row)
            b.append(self.rhs)
            X.append(-row)
            A.append(-row)
            b.append(-self.rhs)

        return X, A, b


    def add_to_individual_population(self, model, x_var):
        row = np.zeros(len(self.parent))
        row[self.indices] = self.coefficents
        if self.sense == "LE":
            model.addConstr(row @ x_var <= self.rhs)
        elif self.sense == "GE":
            model.addConstr(row @ x_var >= self.rhs)
        elif self.sense == "EQ":
            model.addConstr(row @ x_var == self.rhs)
        return

    def add_to_region_population(self,model, u_var, l_var):
        row = np.zeros(len(self.parent))
        row[self.indices] = self.coefficents
        pos_coeff = np.clip(row, 0, None)
        neg_coeff = np.clip(row, None, 0)
        if self.sense == "LE":
            model.addConstr(pos_coeff @ u_var + neg_coeff @ l_var <= self.rhs)
        elif self.sense == "GE":
            model.addConstr(pos_coeff @ l_var + neg_coeff @ u_var >= self.rhs)
        elif self.sense == "EQ":
            model.addConstr(pos_coeff @ u_var + neg_coeff @ l_var <= self.rhs)
            model.addConstr(pos_coeff @ l_var + neg_coeff @ u_var >= self.rhs)
        return
    
    def check_compatibility(self, action_set):
        #todo make this not vacuous
        return True

    def check_feasibility(self, x):
        return True

    def adapt(self, x):
        return

    def add_to_cpx(self, cpx, indices, x):
        """
        adds constraint to ReachableSetEnumeratorMIP
        :param cpx: Cplex object
        :return: nothing
        """
        return
    
    def add_to_gurobi(self, X, A, b):
        """
        adds constraint to Population Verifier
        :param cpx: Cplex object
        :return: x, a row and rhs
        """
        return
