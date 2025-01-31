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
class PopulationVerifierQP:

    SETTINGS = {
        "eps_min": 0.5,
        # todo: set MIP parameters here
    }

    def __init__(self, action_set, verification_type= 'individual', allow_halfspace_regions= False, **kwargs):
        """
        :param action_set:
        :param x:
        :param part:
        :param print_flag:
        :param kwargs:
        """
        assert isinstance(action_set, ActionSet)
        self._action_set = action_set

        self._verification_type = verification_type
        self._allow_halfspace_regions = allow_halfspace_regions
        
        # set actionable indices
        self.actionable_indices = list(range(len(action_set)))

        # build base MIP
        self.model_loaded = False
        
        self.regions = []
        # add non-separable constraints
        #for con in action_set.constraints:
        #    mip, indices = con.add_to_cpx(cpx=cpx, indices=indices, x=self.x)

        # set MIP parameters
        #cpx = self.set_solver_parameters(cpx, print_flag=self.print_flag)
        return 

    @property
    def action_set(self):
        return self._action_set
            

    ### MIP Functions ###
    def build_mip(self):
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
        assert self.model_loaded, "Model not loaded"

        X  = []
        A = []
        b = []
        m = gp.Model('quadratic')
        m.setParam('FeasibilityTol', 1e-9)
        m.setParam('MIPGap', 0)
        n = len(self.action_set)
        u_vars = []
        l_vars = []

        obj = 0

        for idx, aj, xj in zip(range(len(self.action_set)), self.action_set, self.action_set._population_elements.values()):
            # Set upper and lower bounds for x
            ub = xj.get_data_bound('ub')
            lb = xj.get_data_bound('lb')

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

            if self._verification_type == 'population':

                model_variable_type = var_dict[vtype]#gp.GRB.BINARY if vtype == bool else gp.GRB.CONTINUOUS
                uj = m.addVar(vtype = model_variable_type, lb = lb, ub = ub, name = f'u_{idx}')
                lj = m.addVar(vtype = model_variable_type, lb = lb, ub = ub,  name = f'l_{idx}')
                m.addConstr(uj >= lj)
                obj += (uj - lj)/(ub-lb+1)
                u_vars.append(uj)
                l_vars.append(lj)
                b.append(uj)
                b.append(-lj)
            else:
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

            # Set upper and lower bounds for x+a (note this uses the problem bounds, not the population bounds)
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

        if self._allow_halfspace_regions:
            # Add translations of linear classifier as bounds
            min_distance = np.floor(SCORE_SCALING*(pos_coef @ self.action_set.get_data_bounds('lb') + neg_coef @ self.action_set.get_data_bounds('ub')))
            intercept_ub = np.ceil(-1*self.intercept*SCORE_SCALING)
            score_ub = m.addVar(vtype = gp.GRB.INTEGER, lb = min_distance, ub = intercept_ub, name = 'score_ub')
            score_lb = m.addVar(vtype = gp.GRB.INTEGER, lb = min_distance, ub = intercept_ub, name = 'score_lb')
            m.addConstr(score_ub >= score_lb)

            X.append(self.coefficients)
            A.append(np.zeros(n))
            b.append(score_ub/SCORE_SCALING)
            X.append(-self.coefficients)
            A.append(np.zeros(n))
            b.append(-1*score_lb/SCORE_SCALING)

            obj += (score_ub - score_lb)/(SCORE_SCALING*(intercept_ub - min_distance))
            #fixed_features = ~np.array(self.action_set.actionable)
            #fixed_lb = fixed_features * np.array(self.action_set.get_data_bounds('lb'))
            #fixed_ub = fixed_features * np.array(self.action_set.get_data_bounds('ub'))
            #min_fixed_distance = np.floor(SCORE_SCALING*(pos_coef @ fixed_lb + neg_coef @ fixed_ub))
            #max_fixed_distance = np.ceil(SCORE_SCALING*(pos_coef @ fixed_ub + neg_coef @ fixed_lb)) #can tighten this with intercept

            #fixed_score_ub = m.addVar(vtype = gp.GRB.INTEGER, lb = min_fixed_distance, ub = max_fixed_distance, name = 'fixed_score_ub')
            #fixed_score_lb = m.addVar(vtype = gp.GRB.INTEGER, lb = min_fixed_distance, ub = min_fixed_distance, name = 'fixed_score_lb')
            #m.addConstr(fixed_score_ub >= fixed_score_lb)

            #X.append(self.coefficients*fixed_features)
            #A.append(np.zeros(n))
            #b.append(fixed_score_ub/SCORE_SCALING)
            #X.append(-self.coefficients*fixed_features)
            #A.append(np.zeros(n))
            #b.append(-1*fixed_score_lb/SCORE_SCALING)

            #obj += (fixed_score_ub - fixed_score_lb)/(SCORE_SCALING*(max_fixed_distance - min_fixed_distance))




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


        y = m.addMVar(shape = A.shape[0], vtype = gp.GRB.CONTINUOUS, lb = 0)

        if self._verification_type == 'population':
            m.addConstr(X.T @ y  == 0)
            m.addConstr(A.T @ y == 0)
            m.addConstr(y.tolist()[-1] == 1)
            m.addConstr(sum([y.tolist()[i] * b[i] for i in range(len(b))]) <= 0)

            #Valid inequality (box needs to be completely classified in the negative class)
            if not self._allow_halfspace_regions:
                m.addConstr(pos_coef @ u_vars + neg_coef @ l_vars <= -self.intercept)

            # add any population constraints
            #for cons in self.action_set.population_constraints:
            #    cons.add_to_region_population(m, u_vars, l_vars)

            if True:
                #Make sure that we're designing a region that has at least one x in it (important so we dont cheat when we introduce half-spaces)
                x_ub = self.action_set.get_data_bounds('ub')
                x_lb = self.action_set.get_data_bounds('lb')
                var_types = self.action_set.get_variable_types()

                x_var = m.addMVar(shape = X.shape[1], vtype = gp.GRB.CONTINUOUS, lb = x_lb, ub = x_ub, name='x')
                for idx, vtype in enumerate(var_types):
                    x_var[idx].VType = var_dict[vtype]

                for i in range(len(b)-1): #exclude last row (corresponding to after recourse having positive class)
                    if np.sum(np.abs(X[i,:])) > 1e-6:
                        m.addConstr(X[i,:] @ x_var - b[i] <= 0)
                #m.addConstr(sum([y.tolist()[i] * b[i] for i in range(len(b))]) <= 0)

                for cons in self.action_set.population_constraints:
                    cons.add_to_individual_population(m, x_var)


            m.setObjective(obj, gp.GRB.MAXIMIZE)
        elif self._verification_type == 'individual':
            x_ub = self.action_set.get_population_bounds('ub')
            x_lb = self.action_set.get_population_bounds('lb')
            var_types = self.action_set.get_variable_types()

            x_var = m.addMVar(shape = X.shape[1], vtype = gp.GRB.CONTINUOUS, lb = x_lb, ub = x_ub )
            for idx, vtype in enumerate(var_types):
                x_var[idx].VType = var_dict[vtype]

            #m.addConstr(sum([y.tolist()[i] * (b[i] - X[i,:]@ x_var) for i in range(len(b))]) <= 0)
            #Gurobi bug that above doesn't work if X_i @ x == 0
            product_const = 0
            for i in range(len(b)):
                if np.abs(X[i,:]).sum() > 0:
                    product_const += y.tolist()[i] * (b[i] - X[i,:]@ x_var)
                else:
                    product_const += y.tolist()[i] * b[i]
            m.addConstr(product_const <= 0)
            m.addConstr(A.T @ y == 0)
            m.addConstr(y.tolist()[-1] == 1)
            m.addConstr(self.coefficients @ x_var <= -self.intercept)

            # add any population constraints
            for cons in self.action_set.population_constraints:
                cons.add_to_individual_population(m, x_var)

        dv = {}
        if self._verification_type == 'population':
            dv['u'] = u_vars
            dv['l'] = l_vars
        elif self._verification_type == 'individual':
            dv['x'] = x_var.tolist()
        
        if self._allow_halfspace_regions:
            dv['score_ub'] = score_ub
            dv['score_lb'] = score_lb
            #dv['fixed_score_ub'] = fixed_score_ub
            #dv['fixed_score_lb'] = fixed_score_lb


        for region in self.regions:
            region.add_mip_to_model(m, dv, self.coefficients)
            #m, u_vars, l_vars = region.add_qp_to_model(m, u_vars, l_vars)
            #if self._allow_halfspace_regions:
            #    m, u_vars, l_vars = region.add_mip_to_model(m, u_vars, l_vars, self.coefficients, score_ub, score_lb, fixed_score_ub, fixed_score_lb)
            #else:
            #    m, u_vars, l_vars = region.add_mip_to_model(m, u_vars, l_vars)


        # Setup cplex object
        return m, dv
        
    def add_region(self, region_bounds):
        self.regions.append(Region(self.action_set, region_bounds))
        return

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
        

    def verify(self, print_log = False, print_output = True, MIPGap = 0):
        """
        Verify the model
        Returns:
        - Status: True if certify recourse for entire population, False otherwise
        - If False, counterexample (data point or region)
        """
        assert self.model_loaded, "Model not loaded"
        m, dv = self.build_mip()
        self.m = m
        m.setParam('OutputFlag', print_log)
        #m.setParam('TimeLimit', 300)
        m.setParam('MIPGap', 0.1)
        m.optimize()
        if (m.Status == gp.GRB.INFEASIBLE) or (m.Status == gp.GRB.TIME_LIMIT):
            print("All data points have recourse within given population.")
            return True, None
        
        for var in dv.keys():
            if var == 'w':
                continue
            dv[var] = extract_dv_values(dv[var])
        #dv_vals = np.array([var.X for var in dv])
        if self._verification_type == 'population':
            n = len(self.action_set)
            print("All data points in the following region have no recourse:")
            if print_output: tabulate_region(self.action_set, dv)

        elif self._verification_type == 'individual':
            print("The following data point has no recourse:")
            if print_output: tabulate_datapoint(self.action_set, dv)

        return False, dv

def extract_dv_values(dv_list):
    out = []
    if not isinstance(dv_list, list):
        dv_list = [dv_list]
    for element in dv_list:
        if isinstance(element, gp.LinExpr):
            out.append(element.getValue())
        elif isinstance(element, gp.Var):
            out.append(element.X)
        else:
            out.append(element)

    return out

class Region(object):
    def __init__(self, action_set, dv):
        if ('u' not in dv) or ('l' not in dv):
            raise ValueError("Region must have upper and lower bounds")
        
        self.action_set = action_set
        self.u = list(dv['u'])
        self.l = list(dv['l'])

        if 'score_ub' in dv:
            self.score_ub = dv['score_ub'][0]

        if 'score_lb' in dv:
            self.score_lb = dv['score_lb'][0]

        #if 'fixed_score_ub' in dv:
        #    self.fixed_score_ub = dv['fixed_score_ub'][0]
        
        #if 'fixed_score_lb' in dv:
        #    self.fixed_score_lb = dv['fixed_score_lb'][0]

    def add_qp_to_model(self, m, u_vars, l_vars):
        ## DEPRECATED

        n = len(self.u)
        y_region = m.addMVar(shape = n*4, vtype = gp.GRB.CONTINUOUS, lb = 0)
        coefficients = self.u + [-1*val for val in self.l] + u_vars + [-1*var for var in l_vars]
        I = np.eye(n)
        region_matrix = np.concatenate([I, -I, I, -I], axis = 0).T

        print(region_matrix.shape, y_region.shape)
        m.addConstr(region_matrix @ y_region == 0)
        #m.addConstr(y_region @ coefficients == -1)
        m.addConstr(sum([y_region.tolist()[i] * coefficients[i] for i in range(len(coefficients))]) == -1)
        return m, u_vars, l_vars
    
    def add_mip_to_model(self, m, dv, score_classifier = None):

        if ('x' in dv) and ('score_ub' in dv):
            return self.add_mip_to_ind_model(m, dv['x'], score_classifier, dv['score_ub'], dv['score_lb'])
        elif 'x' in dv:
            return self.add_mip_to_ind_model(m, dv['x'])
        elif ('u' in dv) and ('score_ub' in dv):
            return self.add_mip_to_pop_model(m, dv['u'], dv['l'], score_classifier, dv['score_ub'], dv['score_lb'])
        elif 'u' in dv:
            return self.add_mip_to_pop_model(m, dv['u'], dv['l'])
        else:
            raise ValueError("Can't add region to model")
        
    
    def add_mip_to_ind_model(self, m, x_vars, score_classifier = None, score_ub = None, score_lb = None):
        z_vars = []
        data_ub = self.action_set.get_data_bounds('ub')
        data_lb = self.action_set.get_data_bounds('lb')

        for idx, x in enumerate(x_vars):
            if self.u[idx] != data_ub[idx]:
                z = m.addVar(vtype = gp.GRB.BINARY, name = f'z_{idx}')
                m.addConstr(x >= self.u[idx] + 1 + (data_lb[idx] - 1 -self.u[idx]) * (1-z))
                z_vars.append(z)

            if self.l[idx] != data_lb[idx]:
                z = m.addVar(vtype = gp.GRB.BINARY, name = f'z_{idx}')
                m.addConstr(x <= self.l[idx] - 1 - (self.l[idx] - 1 - data_ub[idx]) * (1-z))
                z_vars.append(z)
        
        if score_ub is not None:
            w_pos = np.clip(score_classifier, 0, None)
            w_neg = np.clip(score_classifier, None, 0)

            if (w_pos @ data_lb + w_neg @ data_ub) < self.score_lb/SCORE_SCALING:
                z = m.addVar(vtype = gp.GRB.BINARY, name = 'z')
                z_vars.append(z)
                m.addConstr(score_ub <= self.score_lb - (self.score_lb - np.ceil((w_pos @ data_ub + w_neg @ data_lb)*SCORE_SCALING)) * (1-z))
                m.addConstr(score_lb <= score_ub - 1*z)
        if score_lb is not None:
            w_pos = np.clip(score_classifier, 0, None)
            w_neg = np.clip(score_classifier, None, 0)

            if (w_pos @ data_ub + w_neg @ data_lb) > self.score_ub/SCORE_SCALING:
                z = m.addVar(vtype = gp.GRB.BINARY, name = 'z')
                z_vars.append(z)
                m.addConstr(score_lb >= self.score_ub - (self.score_ub - np.floor((w_pos @ data_lb + w_neg @ data_ub)*SCORE_SCALING)) * (1-z))
                m.addConstr(score_ub >= score_lb + z)
        
        if score_classifier is not None:
            fixed_features = ~np.array(self.action_set.actionable)
            score_classifier = fixed_features*score_classifier 
            data_lb = fixed_features * np.array(self.action_set.get_data_bounds('lb'))
            data_ub = fixed_features * np.array(self.action_set.get_data_bounds('ub'))
        '''
        if fixed_score_ub is not None:
            w_pos = np.clip(score_classifier, 0, None)
            w_neg = np.clip(score_classifier, None, 0)

            if (w_pos @ data_lb - w_neg @ data_ub) < self.fixed_score_lb/SCORE_SCALING:
                z = m.addVar(vtype = gp.GRB.BINARY, name = 'z')
                z_vars.append(z)
                m.addConstr(fixed_score_ub <= self.fixed_score_lb - (self.fixed_score_lb - np.ceil((w_pos @ data_ub + w_neg @ data_lb)*SCORE_SCALING)) * (1-z))
                m.addConstr(fixed_score_lb <= fixed_score_ub - 1*z)
        if fixed_score_lb is not None:
            w_pos = np.clip(score_classifier, 0, None)
            w_neg = np.clip(score_classifier, None, 0)

            if (w_pos @ data_ub - w_neg @ data_lb) > self.fixed_score_ub/SCORE_SCALING:
                z = m.addVar(vtype = gp.GRB.BINARY, name = 'z')
                z_vars.append(z)
                m.addConstr(fixed_score_lb >= self.fixed_score_ub - (self.fixed_score_ub - np.floor((w_pos @ data_lb + w_neg @ data_ub)*SCORE_SCALING)) * (1-z))
                m.addConstr(fixed_score_ub >= fixed_score_lb + z)
        
        if fixed_score_ub is not None:
            w_pos = np.clip(score_classifier, 0, None)
            w_neg = np.clip(score_classifier, None, 0)

            if (w_pos @ data_ub + w_neg @ data_lb) >= self.fixed_score_ub/SCORE_SCALING:
                z = m.addVar(vtype = gp.GRB.BINARY)
                z_vars.append(z)
                m.addConstr(fixed_score_ub >= self.fixed_score_ub + 1 - (self.fixed_score_ub + 1 - np.ceil((w_pos @ data_lb + w_neg @ data_ub)*SCORE_SCALING)) * (1-z))
                #m.addConstr(fixed_score_lb <= fixed_score_ub - 1*z)
        '''
        m.addConstr(sum(z_vars) >= 1)



    def add_mip_to_pop_model(self, m, u_vars, l_vars, score_classifier = None, 
                             score_ub = None, score_lb = None):
        z_vars = []
        data_ub = self.action_set.get_data_bounds('ub')
        data_lb = self.action_set.get_data_bounds('lb')

        for idx, u in enumerate(l_vars):
            if self.u[idx] == data_ub[idx]:
                continue
            z = m.addVar(vtype = gp.GRB.BINARY)
            m.addConstr(u >= self.u[idx] + 1 + (data_lb[idx] - 1 -self.u[idx]) * (1-z))
            z_vars.append(z)
        
        for idx, l in enumerate(u_vars):
            if self.l[idx] == data_lb[idx]:
                continue
            z = m.addVar(vtype = gp.GRB.BINARY)
            m.addConstr(l <= self.l[idx] - 1 - (self.l[idx] - 1 - data_ub[idx]) * (1-z))
            z_vars.append(z)

        
        if score_ub is not None:
            w_pos = np.clip(score_classifier, 0, None)
            w_neg = np.clip(score_classifier, None, 0)

            if (w_pos @ data_lb + w_neg @ data_ub) < self.score_lb/SCORE_SCALING:
                z = m.addVar(vtype = gp.GRB.BINARY)
                z_vars.append(z)
                m.addConstr(score_ub <= self.score_lb - (self.score_lb - np.ceil((w_pos @ data_ub + w_neg @ data_lb)*SCORE_SCALING)) * (1-z))
                m.addConstr(score_lb <= score_ub - 1*z)
        if score_lb is not None:
            w_pos = np.clip(score_classifier, 0, None)
            w_neg = np.clip(score_classifier, None, 0)

            if (w_pos @ data_ub + w_neg @ data_lb) > self.score_ub/SCORE_SCALING:
                z = m.addVar(vtype = gp.GRB.BINARY)
                z_vars.append(z)
                m.addConstr(score_lb >= self.score_ub - (self.score_ub - np.floor((w_pos @ data_lb + w_neg @ data_ub)*SCORE_SCALING)) * (1-z))
                m.addConstr(score_ub >= score_lb + z)
                
        if score_classifier is not None:
            fixed_features = ~np.array(self.action_set.actionable)
            score_classifier = fixed_features*score_classifier 
            data_lb = fixed_features * np.array(self.action_set.get_data_bounds('lb'))
            data_ub = fixed_features * np.array(self.action_set.get_data_bounds('ub'))
        '''
        if fixed_score_ub is not None:
            w_pos = np.clip(score_classifier, 0, None)
            w_neg = np.clip(score_classifier, None, 0)

            if (w_pos @ data_lb - w_neg @ data_ub) < self.fixed_score_lb/SCORE_SCALING:
                z = m.addVar(vtype = gp.GRB.BINARY)
                z_vars.append(z)
                m.addConstr(fixed_score_ub <= self.fixed_score_lb - (self.fixed_score_lb - np.ceil((w_pos @ data_ub + w_neg @ data_lb)*SCORE_SCALING)) * (1-z))
                m.addConstr(fixed_score_lb <= fixed_score_ub - 1*z)
        if fixed_score_lb is not None:
            w_pos = np.clip(score_classifier, 0, None)
            w_neg = np.clip(score_classifier, None, 0)

            if (w_pos @ data_ub - w_neg @ data_lb) > self.fixed_score_ub/SCORE_SCALING:
                z = m.addVar(vtype = gp.GRB.BINARY)
                z_vars.append(z)
                m.addConstr(fixed_score_lb >= self.fixed_score_ub - (self.fixed_score_ub - np.floor((w_pos @ data_lb + w_neg @ data_ub)*SCORE_SCALING)) * (1-z))
                m.addConstr(fixed_score_ub >= fixed_score_lb + z)
        
        if fixed_score_ub is not None:
            w_pos = np.clip(score_classifier, 0, None)
            w_neg = np.clip(score_classifier, None, 0)

            if (w_pos @ data_ub + w_neg @ data_lb) >= self.fixed_score_ub/SCORE_SCALING:
                z = m.addVar(vtype = gp.GRB.BINARY)
                z_vars.append(z)
                m.addConstr(fixed_score_ub >= self.fixed_score_ub + 1 - (self.fixed_score_ub + 1 - np.ceil((w_pos @ data_lb + w_neg @ data_ub)*SCORE_SCALING)) * (1-z))
                #m.addConstr(fixed_score_lb <= fixed_score_ub - 1*z)
        '''
        m.addConstr(sum(z_vars) >= 1)
    


def tabulate_datapoint(action_set, dv):
    """
    prints a table with information about each element in the action set
    :param action_set: ActionSet object
    :return:
    """
    x = dv['x']
    # fmt:off
    TYPES = {bool: "<bool>", int: "<int>", float: "<float>"}
    FMT = {bool: "1.0f", int: "1.0f", float: "1.2f"}
    t = PrettyTable()
    vtypes = [TYPES[v] for v in action_set.variable_type]
    t.add_column("", list(range(len(action_set))), align="r")
    t.add_column("name", action_set.name, align="l")
    t.add_column("type", vtypes, align="c")
    t.add_column("value", np.round(x,2), align="c")
    #t.add_column("lb", [f"{a.lb:{FMT[a.variable_type]}}" for a in action_set], align="c")
    #t.add_column("ub", [f"{a.ub:{FMT[a.variable_type]}}" for a in action_set], align="c")
    #t.add_column("step_direction", action_set.step_direction, align="r")
    #t.add_column("step_ub", [v if np.isfinite(v) else "" for v in action_set.step_ub], align="r")
    #t.add_column("step_lb", [v if np.isfinite(v) else "" for v in action_set.step_lb], align="r")
    print(t)
    return

def tabulate_region(action_set, dv):
    """
    prints a table with information about each element in the action set
    :param action_set: ActionSet object
    :return:
    """
    if ('u' not in dv) or ('l' not in dv):
        raise ValueError("Region must have upper and lower bounds")
    
    u_vars = dv['u']
    l_vars = dv['l']
    TYPES = {bool: "<bool>", int: "<int>", float: "<float>"}
    FMT = {bool: "1.0f", int: "1.0f", float: "1.2f"}

    x_ub = action_set.get_data_bounds('ub')
    x_lb = action_set.get_data_bounds('lb')
    vtypes = [TYPES[v] for v in action_set.variable_type]

    name_output = []
    type_output = []
    lb_output = []
    ub_output = []

    for name, var_type, lb, ub, u, l in zip(action_set.name, action_set.variable_type, x_lb, x_ub, u_vars, l_vars):
        if u == ub and l == lb:
            continue
        name_output.append(name)
        type_output.append(TYPES[var_type])
        lb_output.append(f"{l:{FMT[var_type]}}")
        ub_output.append(f"{u:{FMT[var_type]}}")

    if 'score_ub' in dv:
        name_output.append('Classifier Score')
        type_output.append('-')
        lb_output.append(f"{dv['score_lb'][0]/SCORE_SCALING:{FMT[float]}}")
        ub_output.append(f"{dv['score_ub'][0]/SCORE_SCALING:{FMT[float]}}")

    #if 'fixed_score_ub' in dv:
    #    name_output.append('Fixed Feature Classifier Score')
    #    type_output.append('-')
    #    lb_output.append(f"{dv['fixed_score_lb'][0]/SCORE_SCALING:{FMT[float]}}")
    #    ub_output.append(f"{dv['fixed_score_ub'][0]/SCORE_SCALING:{FMT[float]}}")

    t = PrettyTable()
    t.add_column("", list(range(len(name_output))), align="r")
    t.add_column("name", name_output, align="l")
    t.add_column("type", type_output, align="c")
    t.add_column("lb", lb_output, align="c")
    t.add_column("ub", ub_output, align="c")
    print(t)


    return
