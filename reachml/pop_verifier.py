import numpy as np
from cplex import Cplex, SparsePair
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

class PopulationVerifierMIP:
    SETTINGS = {
        "eps_min": 0.5,
        # todo: set MIP parameters here
    }

    def __init__(self, action_set, verification_type= 'individual', print_flag=False, **kwargs):
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

        # set actionable indices
        self.actionable_indices = list(range(len(action_set)))

        # parse remaining settings
        self.print_flag = print_flag

        # build base MIP
        cpx, indices = self.build_mip()
        self.model_loaded = False

        # add non-separable constraints
        #for con in action_set.constraints:
        #    mip, indices = con.add_to_cpx(cpx=cpx, indices=indices, x=self.x)

        # set MIP parameters
        #cpx = self.set_solver_parameters(cpx, print_flag=self.print_flag)
        self.mip, self.indices = cpx, indices

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

        # Setup cplex object
        cpx = Cplex()
        cpx.set_problem_type(cpx.problem_type.MILP)
        cpx.objective.set_sense(cpx.objective.sense.minimize)
        vars = cpx.variables
        cons = cpx.linear_constraints

        # data variable parameters

        data_lb = self.action_set.get_data_bounds(bound_type="lb")
        data_ub = self.action_set.get_data_bounds(bound_type="ub")

        x_lb = data_lb
        x_ub = data_ub
        x_types = get_cpx_variable_types(self.action_set, relax = False)
        print('x type', x_types)
        x_variable_args = {
            "x": get_cpx_variable_args(
                obj=0,
                name=[f"x({j})" for j in range(len(self.action_set))],
                lb=x_lb,
                ub=x_ub,
                vtype=x_types,
            )
        }
        vars.add(**reduce(combine, x_variable_args.values()))



        # variable parameters
        a_lb = self.action_set.get_bounds(x=None, bound_type="lb")
        a_ub = self.action_set.get_bounds(x=None, bound_type="ub") 
        a_pos_max = np.abs(a_ub)
        a_neg_max = np.abs(a_lb)
        a_types = get_cpx_variable_types(self.action_set, self.actionable_indices, relax=True)
        print(a_types)

        # add variables to CPLEX
        variable_args = {
            "a": get_cpx_variable_args(
                obj=0.0,
                name=[f"a({j})" for j in self.actionable_indices],
                lb=a_lb,
                ub=a_ub,
                vtype=a_types,
            ),
             "a_pos": get_cpx_variable_args(
                obj=1.0,
                name=[f"a({j})_pos" for j in self.actionable_indices],
                lb=0.0,
                ub=a_pos_max,
                vtype=a_types,
            ),
            "a_neg": get_cpx_variable_args(
                obj=1.0,
                name=[f"a({j})_neg" for j in self.actionable_indices],
                lb=0.0,
                ub=a_neg_max,
                vtype=a_types,
            )
        }
        vars.add(**reduce(combine, variable_args.values()))
        '''
            "a_pos": get_cpx_variable_args(
                obj=0.0,
                name=[f"a({j})_pos" for j in self.actionable_indices],
                lb=0.0,
                ub=a_pos_max,
                vtype=a_types,
            ),
            "a_neg": get_cpx_variable_args(
                obj=0.0,
                name=[f"a({j})_neg" for j in self.actionable_indices],
                lb=0.0,
                ub=a_neg_max,
                vtype=a_types,
            ),
            "a_sign": get_cpx_variable_args(
                obj=0.0,
                name=[f"a({j})_sign" for j in self.actionable_indices],
                lb=0.0,
                ub=1.0,
                vtype="B",
            ),
            #
            "c": get_cpx_variable_args(
                obj=0,
                name=[f"c({j})" for j in self.actionable_indices],
                lb=a_lb,
                ub=a_ub,
                vtype=a_types,
            ),
        '''

        # store information about variables for manipulation / debugging
        indices = CplexGroupedVariableIndices()
        indices.append_variables(variable_args)
        indices.append_variables(x_variable_args)
        names = indices.names

        #for j, a_j, a_pos_j, a_neg_j, a_sign_j, c_j, x_j, x_ub, x_lb in zip(
        #    self.actionable_indices,
        #    names["a"],
        #    names["a_pos"],
        #    names["a_neg"],
        #    names["a_sign"],
        #    names["c"],
        #    names["x"],
        #    data_ub,
        #    data_lb
        #):  
        for j, a_j, a_pos_j, a_neg_j, x_j, x_ub, x_lb in zip(
            self.actionable_indices,
            names["a"],
            names["a_pos"],
            names["a_neg"],
            names["x"],
            data_ub,
            data_lb
        ):  
            
            # a_pos_j - a_j ≥ 0
            cons.add(
                names=[f"abs_val_pos_{a_j}"],
                lin_expr=[SparsePair(ind=[a_pos_j, a_j], val=[1.0, -1.0])],
                senses="G",
                rhs=[0.0],
            )

            # a_neg_j + a_j ≥ 0
            cons.add(
                names=[f"abs_val_neg_{a_j}"],
                lin_expr=[SparsePair(ind=[a_neg_j, a_j], val=[1.0, 1.0])],
                senses="G",
                rhs=[0.0],
            )
            '''
            cons.add(
                names=[f"set_{a_j}_sign_pos"],
                lin_expr=[
                    SparsePair(ind=[a_pos_j, a_sign_j], val=[1.0, -a_pos_max[j]])
                ],
                senses="L",
                rhs=[0.0],
            )

            cons.add(
                names=[f"set_{a_j}_sign_neg"],
                lin_expr=[SparsePair(ind=[a_neg_j, a_sign_j], val=[1.0, a_neg_max[j]])],
                senses="L",
                rhs=[a_neg_max[j]],
            )

            cons.add(
                names=[f"set_{a_j}"],
                lin_expr=[
                    SparsePair(ind=[a_j, a_pos_j, a_neg_j], val=[1.0, -1.0, 1.0])
                ],
                senses="E",
                rhs=[0.0],
            )

            cons.add(
                names=[f"set_{c_j}"],
                lin_expr=[SparsePair(ind=[c_j, a_j], val=[1.0, -1.0])],
                senses="E",
                rhs=[0.0],
            )
            '''
            cons.add(
                names=[f"data_ub_{x_j}"],
                lin_expr=[SparsePair(ind=[x_j, a_j], val=[1.0, 1.0])],
                senses="L",
                rhs=[x_ub],
            )

            cons.add(
                names=[f"data_lb_{x_j}"],
                lin_expr=[SparsePair(ind=[x_j, a_j], val=[1.0, 1.0])],
                senses="G",
                rhs=[x_lb],
            )
        
        if self._verification_type == 'population':
            #Create population level problems
            feature_width = 1./np.clip(np.array(data_ub) - np.array(data_lb),1,None)


            ul_variable_args = {
                "u": get_cpx_variable_args(
                    obj=-1*feature_width,
                    name=[f"u[{j}]" for j in range(len(self.action_set))],
                    lb=data_lb,
                    ub=data_ub,
                    vtype=x_types,
                ),
                "l": get_cpx_variable_args(
                    obj=feature_width,
                    name=[f"l[{j}]" for j in range(len(self.action_set))],
                    lb=data_lb,
                    ub=data_ub,
                    vtype=x_types
                )
            }
            vars.add(**reduce(combine, ul_variable_args.values()))
            indices.append_variables(ul_variable_args)
            names = indices.names

            #Add constraints to ensure x comes from population ub, lb
            for j, u_j, l_j, x_j in zip(
                range(len(self.action_set)),
                names["u"],
                names["l"],
                names["x"],
            ):
                cons.add(
                    names=[f"{x_j}_{u_j}"],
                    lin_expr=[SparsePair(ind=[x_j, u_j], val=[1.0, -1.0])],
                    senses="L",
                    rhs=[0.0],
                )
                cons.add(
                    names=[f"{x_j}_{l_j}"],
                    lin_expr=[SparsePair(ind=[x_j, l_j], val=[1.0, -1.0])],
                    senses="G",
                    rhs=[0.0],
                )

        return cpx, indices
    
    def load_model(self, ml_model):
        """
        Load ml model that is to be verified
        - Right now only support scikit-learn logistic regression
        """
        assert isinstance(ml_model, LogisticRegression)
        #todo if model already loaded then remove previous constraints

        coefficients = ml_model.coef_[0]
        #coefficients = np.zeros(len(self.action_set))
        #coefficients[0] = 1
        intercept = ml_model.intercept_[0]
        #intercept = 1

        data_lb = self.action_set.get_data_bounds(bound_type="lb")
        data_ub = self.action_set.get_data_bounds(bound_type="ub")

        #compute lower bound for score by multipling lower bound of data with positive coefficeints and upper bound with negative coefficients
        score_lb = np.dot(data_lb, np.clip(coefficients, 0, None)) + np.dot(data_ub, -1*np.clip(-1*coefficients, 0, None)) + intercept        

        cpx = self.mip
        indices = self.indices

        cons = cpx.linear_constraints
        vars = cpx.variables
        if self._verification_type == 'population':
            variable_args = {'z': get_cpx_variable_args(obj = -(len(self.action_set)+1), name = 'z', vtype = "B", ub = 1.0, lb = 0.0)}
        else:
            variable_args = {'z': get_cpx_variable_args(obj = -1, name = 'z', vtype = "B", ub = 1.0, lb = 0.0)}

        vars.add(**reduce(combine, variable_args.values()))
        indices.append_variables(variable_args)

        #Assuming p=0.5 so want to check w(x+a) + b >= 0
        cons.add(
            names = ['classification'],
            lin_expr = [SparsePair(ind = ['z'] + indices.names['x']+indices.names['a'], val = [1000] + list(coefficients)+list(coefficients))],
            senses = "G",
            rhs = [-intercept]
        )
        if self._verification_type == 'individual':
            cons.add(
                names = ['upper_x_infeasible'],
                lin_expr = [SparsePair(ind = indices.names['x'], val = list(coefficients))],
                senses = "L",
                rhs = [-intercept - 0.001]
            )

        self.mip = cpx
        self.indices = indices
        self.model_loaded = True
        return

    def verify(self):
        """
        Verify the model
        """
        assert self.model_loaded, "Model not loaded"
        cpx = self.mip
        indices = self.indices
        names = indices.names

        cpx.write('model.mps')
        cpx.write('model.lp')

        ll_vars = names['a'] + names['a_pos'] + names['a_neg']# + names['a_sign'] + names['c']
        
        if self._verification_type == 'population':
            ll_vars += names['x']
        #get list of constraints from cplex
        ss_constraints = [name for name in cpx.linear_constraints.get_names() if 'upper_' not in name]

        with open('model.aux', "w") as OUTPUT:
            # Num lower-level variables
            OUTPUT.write("@NUMVARS\n{}\n".format(len(ll_vars)+1))
            # Num lower-level constraints
            OUTPUT.write("@NUMCONSTRS\n{}\n".format(len(ss_constraints)))
            
            # Indices of lower-level variables
            OUTPUT.write("@VARSBEGIN\n")
            #nx_upper = len(x_vars)
            for var in names['a_pos'] + names['a_neg']:
                OUTPUT.write("{} {}\n".format(var,0.0))
            for var in names['a']:
                OUTPUT.write("{} {}\n".format(var,0.0))
            OUTPUT.write("{} {}\n".format(names['z'][0],1000))
            OUTPUT.write("@VARSEND\n")
            
            # Indices of lower-level constraints
            OUTPUT.write("@CONSTRSBEGIN\n")
            for const in ss_constraints:
                OUTPUT.write("{}\n".format(const))
            OUTPUT.write("@CONSTRSEND\n")
            OUTPUT.write("@MPS\n")
            OUTPUT.write("model.mps")


        out = subprocess.run(["mibs", "-instance", "model.mps"], capture_output=True) 
        print(out.stdout.decode('utf-8'))

        return 