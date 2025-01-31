# Understanding Fixed Predictions via Confined Regions

[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the code to reproduce the results in **Understanding Fixed Predictions via Confined Regions**

## Background

*Recourse* is the ability of a decision subject to change the prediction of a machine learning model through actions on their features. *Recourse verification* aims to tell if a decision subject is assigned a prediction that is fixed. *Region Verification* checks whether an entire region of the feature space is confined, meaning that every individual does not have recourse.

## Dependencies

Many of the functions in this codebacke will require [Gurobi](https://www.gurobi.com/) to run properly. Gurobi is a commercial MILP and MIQCP solver, but has free licenses for academic use. 

## Quickstart

The following example shows how to specify actionability constraints using `ActionSet`, which constraints on both the region and actions, and verify recourse over the region using `PopulationVerifierQP`.

```python
import pandas as pd
from reachml import ActionSet, PopulationVerifierQP
from reachml.constraints import OneHotEncoding, DirectionalLinkage

# feature matrix with 3 points
X = pd.DataFrame(
    {
        "age": [32, 19, 52],
        "marital_status": [1, 0, 0],
        "years_since_last_default": [5, 0, 21],
        "job_type_a": [0, 1, 1], # categorical feature with one-hot encoding
        "job_type_b": [1, 0, 0],
        "job_type_c": [0, 0, 0],
    }
)

# Create an action set
action_set = ActionSet(X)

# `ActionSet` infers the type and bounds on each feature from `X`. To see them:
print(action_set)

## print(action_set) should return the following output
##+---+--------------------------+--------+------------+----+----+----------------+---------+---------+
##|   | name                     |  type  | actionable | lb | ub | step_direction | step_ub | step_lb |
##+---+--------------------------+--------+------------+----+----+----------------+---------+---------+
##| 0 | age                      | <int>  |   False    | 19 | 52 |              0 |         |         |
##| 1 | marital_status           | <bool> |   False    | 0  | 1  |              0 |         |         |
##| 2 | years_since_last_default | <int>  |    True    | 0  | 21 |              1 |         |         |
##| 3 | job_type_a               | <bool> |    True    | 0  | 1  |              0 |         |         |
##| 4 | job_type_b               | <bool> |    True    | 0  | 1  |              0 |         |         |
##| 5 | job_type_c               | <bool> |    True    | 0  | 1  |              0 |         |         |
##+---+--------------------------+--------+------------+----+----+----------------+---------+---------+

# Specify constraints on individual features
action_set[["age", "marital_status"]].actionable = False # these features cannot or should not change
action_set["years_since_last_default"].ub = 100 # set maximum value of feature to 100
action_set["years_since_last_default"].step_direction = 1 # actions can only increase value
action_set["years_since_last_default"].step_ub = 1 # limit actions to changes value by 1


# Specify constraint to maintain one hot-encoding on `job_type`
action_set.constraints.add(
    constraint=OneHotEncoding(names=["job_type_a", "job_type_b", "job_type_c"])
)

# Specify deterministic causal relationships
# if `years_since_last_default` increases, then `age` must increase commensurately
# This will force `age` to change even though it is not immediately actionable
action_set.constraints.add(
    constraint=DirectionalLinkage(
        names=["years_since_last_default", "age"], scales=[1, 1]
    )
)
# We can specify bounds to define a region we with to audit as follows (this will construct a region of 20-30 year olds): 
action_set._population_elements["age"].ub = 30
action_set._population_elements["age"].lb = 20

#You can also specify the same non-separable constraints for the region using population constraints:
action_set.population_constraints.add(
    constraint=OneHotEncoding(names=["job_type_a", "job_type_b", "job_type_c"])
)
action_set.population_constraints.add(
    constraint=DirectionalLinkage(
        names=["years_since_last_default", "age"], scales=[1, 1]
    )
)

# Check that `ActionSet` is consistent with observed data
# For example, if features must obey one-hot encoding, this should be the case for X
assert action_set.validate(X)


# Construct the Verifier Object that will run the audit 
# We specify the verification type as 'population' to return confined boxes within our region
pop_verifier = PopulationVerifierQP(action_set,  verification_type= 'population')

#load in a linear classifier (this also works with a logistic regression object from scikit-learn!)
pop_verifier.load_model_manual(w, b)

#Run the audit. The verifier will output a flag indicating if the region is responsive
# If it is not responsive, it will also output an object with bounds on the largest confined box
flag, confined_box = pop_verifier.verify()
