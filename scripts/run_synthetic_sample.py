import sys
import os
import time
import psutil
import argparse
import contextlib
import warnings
import traceback

warnings.simplefilter(action="ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from reachml.recourse_verifier import RecourseVerifier

from src.paths import *
from src import fileutils
def generate_region_generator(possible_values, reg):
    region_possible_vals = possible_values.copy()
    for col, val in reg.items():
        for key in region_possible_vals.keys():
            if col in region_possible_vals[key].columns:
                region_possible_vals[key] = region_possible_vals[key].query(f"{col} == {val}")
    return region_possible_vals

def generate_region_generator_ul(possible_values, names, u, l):
    region_possible_vals = possible_values.copy()
    for col, uj, lj in zip(names, u, l):
        for key in region_possible_vals.keys():
            if col in region_possible_vals[key].columns:
                region_possible_vals[key] = region_possible_vals[key].query(f"({col} <= {uj}) and ({col} >= {lj})")
    return region_possible_vals


def sample_from_region(region_possible_vals):
    return pd.concat([region_possible_vals[col].sample(1).reset_index(drop=True) for col in region_possible_vals.keys()], axis=1)
def compute_region_size(region_possible_vals):
    return np.prod([region_possible_vals[col].shape[0] for col in region_possible_vals.keys()])


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
    "method_name": "synthetic_sample_region_verification",
    "total_regions": 200,
    "random_seed": 2338,
    "num_samples": 100,
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
regions = fileutils.load(get_region_file(**settings))
generator = fileutils.load(get_generator_file(**settings))

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

individual_verifier = RecourseVerifier(action_set)
individual_verifier.load_model_manual(w, b)

results = []
for idx, region in tqdm(enumerate(regions), total=len(regions)):
    action_set = fileutils.load(get_action_set_file(**settings))
    #query = " and ".join(f"({key} == {val})" for key, val in region.items())
    #global_ub = action_set.get_data_bounds('ub')
    #global_lb = action_set.get_data_bounds('lb')

    #lb = data.X_df.query(query).min().to_list()
    #ub = data.X_df.query(query).max().to_list()
    region_generator = generate_region_generator(generator, region)
    data_sample = pd.concat([sample_from_region(region_generator) for i in range(settings["num_samples"])], axis= 0).to_numpy()
    start_time = time.process_time()
    sample_results = []
    for idx, x in enumerate(data_sample):
        try:
            with open(os.devnull, "w") as devnull:
                with contextlib.redirect_stdout(devnull):
                    flag, a = individual_verifier.verify(x, print_flag = True)
                    sample_results.append(flag)
        except Exception as e:
            print(e)
            traceback.print_exc()
    exec_time = time.process_time() - start_time
    sample_results = np.array(sample_results)
    results.append({"region_idx": idx,
                    "a": None,
                    "found_x": None,
                    "certifies_recourse": np.all(sample_results),
                    "certifies_robustness": np.all(~sample_results),
                    "exec_time": exec_time,
                    }
                    )


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