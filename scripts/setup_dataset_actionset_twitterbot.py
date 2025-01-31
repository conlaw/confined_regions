# todo: update this script for the template

import os
import sys

import numpy as np
import pandas as pd
pd.set_option('display.width', 1000)
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)


from src.paths import *
from src import fileutils
from src.generator_utils import *
from src.data import BinaryClassificationDataset
from reachml import ActionSet, ReachableSetDatabase
from reachml.constraints import *

import pprint as pp
from utils import check_processing_loss, tabulate_actions, tally, tally_predictions
from src.training import extract_predictor
## Specify which dataset to pull
settings = {
    "data_name": "twitterbot",
    "action_set_names": ["complex_nD"],
    "random_seed": 2338,
    "description": "to predict if a twitter account is a bot",
    "fold_id": "K05N01",
    "generate_regions": True,
    "generate_generator_dict": True
}

SOURCE_VALS = [
        "source_automation",
        "source_other",
        "source_branding",
        "source_mobile",
        "source_web",
        "source_app",
    ]
QUANTIZED_FEATURES = {
        "follower_friend_ratio": [1, 10, 100, 1000, 10000, 100000],
        "age_of_account_in_days": [365, 730],
        # "urls_count": [10, 100],
        # "cdn_content_in_kb": [10, 50, 100, 200, 500],
        "user_replied": [10, 100],
        "user_favourited": [1000, 10_000],
        "user_retweeted": [1, 10, 100],
        # "sources_count": [0, 2, 10],
    }

def process_dataset(raw_df):
    df = pd.DataFrame()
    df["is_human"] = raw_df["is_human"]

    for source_val in SOURCE_VALS:
        df[source_val] = raw_df[source_val]

    #df["follower_friend_ratio"] = raw_df["follower_friend_ratio"].astype(int)

    for feature, thresholds in QUANTIZED_FEATURES.items():
        for threshold in thresholds:
            df[f"{feature}_geq_{threshold}"] = raw_df[feature] >= threshold
    
    return df

def simple_1D(data):
    A = ActionSet(data.X_df)
    immutable = [a.name for a in A if 'follower_friend_ratio' in a.name] + SOURCE_VALS
    A[immutable].actionable = False

    return A

def complex_1D(data):
    A = simple_1D(data)
    return A

def complex_nD(data):
    A = simple_1D(data)
    for feature, thresholds in QUANTIZED_FEATURES.items():
        A.constraints.add(
                constraint=ThermometerEncoding(
                        names=[a.name for a in A if feature in a.name],
                        )
                )
        A.population_constraints.add(
                constraint=ThermometerEncoding(
                        names=[a.name for a in A if feature in a.name],
                        )
                )
    return A

def generate_possible_values_dictionary():
    possible_values_df = {}
    source_cols = [a.name for a in action_set if 'source' in a.name]
    for col in source_cols:
        possible_values_df[col] = pd.DataFrame(num_range(0,1), columns = [col])

    therm_cols = ['follower_friend_ratio_geq_', 'age_of_account_in_days_geq_', 'user_replied_geq_', 'user_favourited_geq_', 'user_retweeted_geq_']
    for col in therm_cols:
        col_names = [a.name for a in action_set if col in a.name]
        possible_values_df[col] = pd.DataFrame(therm_lower(len(col_names)), columns = col_names)
    return possible_values_df

### Read in raw format of file
loaded = BinaryClassificationDataset.read_csv(
    data_file= data_dir / settings["data_name"] / f"{settings['data_name']}")

# process dataset
data_df = process_dataset(raw_df=loaded.df)

# create processed dataset
data = BinaryClassificationDataset.from_df(data_df)
data.generate_cvindices(
        strata=data.y,
        total_folds_for_cv=[1, 3, 4, 5],
        replicates=1,
        seed=settings["random_seed"],
        )

# create actionset
for name in settings["action_set_names"]:

    if name == "complex_nD":
        action_set = complex_nD(data)

    print(tabulate_actions(action_set))
    # check that action set matches bounds and constraints
    try:
        assert action_set.validate(data.X)
    except AssertionError:
        violations_df = action_set.validate(data.X, return_df=True)
        violations = ~violations_df.all(axis=0)
        violated_columns = violations[violations].index.tolist()
        print(violated_columns)
        raise AssertionError()

    # save dataset
    fileutils.save(
            data,
            path=get_data_file(settings["data_name"], action_set_name=name),
            overwrite=True,
            check_save=False,
            )

    # save actionset
    fileutils.save(
            action_set,
            path=get_action_set_file(settings["data_name"], action_set_name=name),
            overwrite=True,
            check_save=True,
            )

    
    if settings['generate_regions']:
        A = ActionSet(data.X_df)
        immutable = SOURCE_VALS
        regions = data.X_df[immutable].drop_duplicates().to_dict(orient='records')
        fileutils.save(
                    regions,
                    path=get_region_file(settings["data_name"], action_set_name=name),
                    overwrite=True,
                    check_save=False,
                    )

    if settings['generate_generator_dict']:
        possible_value_dict = generate_possible_values_dictionary()
        fileutils.save(
                    possible_value_dict,
                    path=get_generator_file(settings["data_name"], action_set_name=name),
                    overwrite=True,
                    check_save=False,
                    )