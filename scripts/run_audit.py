import os
import sys
import psutil
import rich

sys.path.append(os.getcwd())

import numpy as np
import pandas as pd
import argparse
from tqdm.auto import tqdm
from reachml import ReachableSetDatabase

DB_ACTION_SET_NAME = "complex_nD"

settings = {
    "data_name": "german",
    "action_set_name": "complex_nD",
    "model_type": "xgb",
    "method_name": "dice",
}

ppid = os.getppid()
process_type = psutil.Process(ppid).name()
if process_type not in ("pycharm"):
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_name", type=str, required=True)
    parser.add_argument("--action_set_name", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--method_name", type=str, required=True)
    args, _ = parser.parse_known_args()
    settings.update(vars(args))

from src.paths import *
from src import fileutils

# load action set and processed data
# load dataset and actionset
if settings["data_name"] == "givemecreditBias":
    settings["data_name"] = "givemecredit"
    data = fileutils.load(get_data_file(**settings))
    action_set = fileutils.load(get_action_set_file(**settings))
    settings["data_name"] = "givemecreditBias"
else:
    data = fileutils.load(get_data_file(**settings))
    action_set = fileutils.load(get_action_set_file(**settings))


# load processed model
model_results = fileutils.load(get_model_file(**settings))
clf = model_results["model"]
scaler = model_results["scaler"]
reformat = lambda x: x.reshape(1, -1) if x.ndim == 1 else x
if scaler is None:
    rescale = lambda x: reformat(x)
else:
    rescale = lambda x: scaler.transform(reformat(x))

# run audit
null_action = np.zeros(data.d)
nan_action = np.repeat(np.nan, data.d)
results = {}

if settings["method_name"] in ("ip"):
    # load counterfactuals
    raw_results = fileutils.load(get_benchmark_results_file(**settings))

    for info in tqdm(raw_results):

        idx = info["id_for_x"]
        out = results.get(idx, None)

        # skip if we already have an action
        if out is not None and out["actionable"] and out["post_prediction"]:
            continue

        # pull x, y, reachable set
        x = data.df.loc[idx, data.names.X].to_numpy().reshape(1, -1)
        y = data.df.loc[idx, data.names.y]
        #R = db[x]
        fx = clf.predict(rescale(x)).squeeze()

        # build output for this point
        if out is None:
            out = {
                "y_true": y > 0,
                "orig_prediction": fx > 0,
                "post_prediction": False,  # f(x+a)
                "a": info["a"],  # action that was returned;
                "actionable": False,  # set to False by default
                "found_a": info["found_a"],  # True if method found a
                "certifies_infeasibility": info["certifies_infeasibility"],
                "abstains": info["abstains"],
                "recourse_exists": info["found_a"],
            }

        if out["orig_prediction"]:
            a = null_action
        elif info["found_a"]:
            a = np.array(info["a"])
        else:
            a = nan_action

        if np.isfinite(a).all():
            xp = np.add(x, a)
            fxp = clf.predict(rescale(xp)).squeeze()
            # visualize_diff(x, xq)
            # print("feasible:", xq in reachable_set)
            out.update(
                {
                    "post_prediction": fxp > 0,
                    "found_a": info["found_a"],
                    "a": a,
                    "actionable": True,
                }
            )

        results[idx] = out

# save results
fileutils.save(
    results,
    path=get_audit_results_file(**settings),
    overwrite=True,
    check_save=False,
)


# print stats
df = pd.DataFrame.from_dict(results, orient="index")
stat_names = df.columns.tolist()
for key, value in settings.items():
    df[key] = value
df = df[list(settings.keys()) + stat_names].query("orig_prediction == False")
print(
    "\n".join(
        [
            f"method: {settings['method_name']}",
            f"model_type: {settings['model_type']}",
            f"#f(x) = 0: {df.shape[0]}",
            f"#found actions: {df.found_a.sum()}",
            f"#actionable: {df.actionable.sum()}/{df.shape[0]}",
            f"#recourse exists: {df.recourse_exists.sum()}/{df.shape[0]}",
        ]
    )
)
