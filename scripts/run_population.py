import sys
import os
import time
import psutil
import argparse
import contextlib
import warnings

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from reachml.qp_pop_verifier import PopulationVerifierQP

from src.paths import *
from src import fileutils


def undo_coefficient_scaling(clf=None, coefficients=None, intercept=0.0, scaler=None):
    """
    given coefficients and data for scaled data, returns coefficients and intercept for unnormalized data

    w = w_scaled / sigma
    b = b_scaled - (w_scaled / sigma).dot(mu) = b_scaled - w.dot(mu)

    :param sklearn linear classifier
    :param coefficients: vector of coefficients
    :param intercept: scalar for the intercept function
    :param scaler: sklearn.Scaler or

    :return: coefficients and intercept for unnormalized data

    """
    if coefficients is None:
        assert clf is not None
        assert intercept == 0.0
        assert hasattr(clf, "coef_")
        coefficients = clf.coef_[0]
        intercept = clf.intercept_[0] if hasattr(clf, "intercept_") else 0.0

    if scaler is None:
        w = np.array(coefficients)
        b = float(intercept)
    else:
        isinstance(scaler, StandardScaler)
        x_shift = np.array(scaler.mean_)
        x_scale = np.sqrt(scaler.var_)
        w = coefficients / x_scale
        w = np.array(w).flatten()
        w[np.isnan(w)] = 0

        b = intercept - np.dot(w, x_shift)
        b = float(b)

    return w, b
    # coefficients_unnormalized = scaler.inverse_transform(coefficients.reshape(1, -1))


# args data name, hard cap, action set name
settings = {
    "data_name": "fico",
    "action_set_name": "complex_nD",
    "model_type": "logreg",
    "method_name": "qp_individual",
    "total_regions": 30,
    "random_seed": 2338,
}

# parse the settings when the script is run from the command line
ppid = os.getppid()  # get parent process id
process_type = psutil.Process(ppid).name()  # e.g. pycharm, bash
if process_type not in ("pycharm"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--action_set_name", type=str, required=True)
    # parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--total_regions", type=int, default=settings["total_regions"])
    parser.add_argument("--random_seed", type=int, default=settings["random_seed"])
    args, _ = parser.parse_known_args()
    settings.update(vars(args))

# load dataset and actionset
data = fileutils.load(get_data_file(**settings))
action_set = fileutils.load(get_action_set_file(**settings))

# load model
model_results = fileutils.load(get_model_file(**settings))
clf = model_results["model"]
scaler = model_results["scaler"]
if scaler is None:
    predictions = clf.predict(data.X)
else:
    predictions = clf.predict(scaler.transform(data.X))
w, b = undo_coefficient_scaling(clf, scaler=scaler)

# setup action set for AR

individual_verifier = PopulationVerifierQP(action_set,  verification_type= 'individual')
individual_verifier.load_model_manual(w, b)
#individual_verifier.verify(x, print_flag = True)
results = []
try:
    start_time = time.process_time()
    with open(os.devnull, "w") as devnull:
        with contextlib.redirect_stdout(devnull):
            flag, x = individual_verifier.verify()
    exec_time = time.process_time() - start_time
    results.append(
                        {
                            "a": x,
                            "found_x": not flag,
                            "certifies_recourse": flag,
                            "exec_time": exec_time,
                        }
                    )

except Exception as e:
    print(e)
print(results)
df = pd.DataFrame(data=results)
stat_names = df.columns.tolist()
df.to_csv(get_benchmark_results_csv_file(**settings), index=False)
fileutils.save(
    results,
    path=get_benchmark_results_file(**settings),
    overwrite=True,
    check_save=False,
)

# audit
#w = np.round(w,1)
#b = np.round(b,1)
settings["method_name"] = "qp_population"
results = []
#individual_verifier = PopulationVerifierQP(action_set,  verification_type= 'population', allow_halfspace_regions = False)
#individual_verifier.load_model_manual(w, b)

#COPIED
pop_verifier = PopulationVerifierQP(action_set,  verification_type= 'population', allow_halfspace_regions=False)
pop_verifier.load_model_manual(w, b)
individual_verifier = PopulationVerifierQP(action_set,  verification_type= 'individual', allow_halfspace_regions=False)
individual_verifier.load_model_manual(w, b)

#flag, outputs = individual_verifier.verify(print_flag = True)

mipgap = 0
for i in [1]:
    try:
        #with open(os.devnull, "w") as devnull:
        #    with contextlib.redirect_stdout(devnull):
        for i in tqdm(range(settings["total_regions"])):
            flag, outputs = individual_verifier.verify(print_output = False)
            if flag: break
            start_time = time.process_time()
            flag, outputs = pop_verifier.verify(print_output = True, MIPGap = mipgap)
            exec_time = time.process_time() - start_time
            results.append(
                                        {   "iteration": i,
                                            "a": outputs,
                                            "found_x": not flag,
                                            "certifies_recourse": flag,
                                            "exec_time": exec_time,
                                        }
                                    )
            if exec_time > 10:
                mipgap = 0.1
            if flag:
                break
            else:
                individual_verifier.add_region(outputs)
                pop_verifier.add_region(outputs)

    except Exception as e:
        print(e)

# convert results to dataframe, prepend settings, and save
print(results)
df = pd.DataFrame(data=results)
stat_names = df.columns.tolist()
df.to_csv(get_benchmark_results_csv_file(**settings), index=False)
fileutils.save(
    results,
    path=get_benchmark_results_file(**settings),
    overwrite=True,
    check_save=False,
)

settings["method_name"] = "qp_population_halfspace"
results = []
#individual_verifier = PopulationVerifierQP(action_set,  verification_type= 'population', allow_halfspace_regions = True)
#individual_verifier.load_model_manual(w, b)
pop_verifier = PopulationVerifierQP(action_set,  verification_type= 'population', allow_halfspace_regions=True)
pop_verifier.load_model_manual(w, b)
individual_verifier = PopulationVerifierQP(action_set,  verification_type= 'individual', allow_halfspace_regions=True)
individual_verifier.load_model_manual(w, b)

mipgap = 0
for i in [1]:
    try:
        #with open(os.devnull, "w") as devnull:
        #    with contextlib.redirect_stdout(devnull):
        for i in tqdm(range(settings["total_regions"])):
            flag, outputs = individual_verifier.verify(print_output = False)
            if flag: break
            start_time = time.process_time()
            flag, outputs = pop_verifier.verify(print_output = True, MIPGap = mipgap)
            exec_time = time.process_time() - start_time
            results.append(
                                        {   "iteration": i,
                                            "a": outputs,
                                            "found_x": not flag,
                                            "certifies_recourse": flag,
                                            "exec_time": exec_time,
                                        }
                                    )
            if exec_time > 10:
                mipgap = 0.1
            if flag:
                break
            else:
                individual_verifier.add_region(outputs)
                pop_verifier.add_region(outputs)

    except Exception as e:
        print(e)
# convert results to dataframe, prepend settings, and save
print(results)
df = pd.DataFrame(data=results)
stat_names = df.columns.tolist()
df.to_csv(get_benchmark_results_csv_file(**settings), index=False)
fileutils.save(
    results,
    path=get_benchmark_results_file(**settings),
    overwrite=True,
    check_save=False,
)
