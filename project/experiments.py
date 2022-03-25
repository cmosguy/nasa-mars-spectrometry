#%%
import itertools
from pathlib import Path
from pprint import pprint
from tkinter import W

from matplotlib import legend, pyplot as plt, cm
import numpy as np
import pandas as pd
from pandas_path import path

from sklearn.dummy import DummyClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import make_scorer, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
import XGBClassifier

from tqdm import tqdm

import plotly.express as px
import plotly.graph_objects as go

import os


from argparse import ArgumentParser


# Feature Engineering
class FeatureEngineering():
    def __init__(self) -> None:
        # For our analysis, we will engineer a simple set of features that try to capture some of these characteristics. 
        # We will discretize the overall temperature range into bins (of 100 degrees), and calculate maximum relative 
        # abundance within that temperature bin for each m/z value.
        #
        # There are many ways to describe the shapes of these peaks with higher fidelity than the approach we demonstrate here

        # Create a series of temperature bins
        self.temprange = pd.interval_range(start=-100, end=1500, freq=100)

        # Make dataframe with rows that are combinations of all temperature bins
        # and all m/z values
        allcombs = list(itertools.product(self.temprange, [*range(0, 100)]))

        self.allcombs_df = pd.DataFrame(allcombs, columns=["temp_bin", "m/z"])

    def abun_per_tempbin(self, df):

        """
        Transforms dataset to take the preprocessed max abundance for each
        temperature range for each m/z value

        Args:
            df: dataframe to transform

        Returns:
            transformed dataframe
        """

        # Bin temperatures
        df["temp_bin"] = pd.cut(df["temp"], bins=self.temprange)

        # Combine with a list of all temp bin-m/z value combinations
        df = pd.merge(self.allcombs_df, df, on=["temp_bin", "m/z"], how="left")

        # Aggregate to temperature bin level to find max
        df = df.groupby(["temp_bin", "m/z"]).max("abun_minsub_scaled").reset_index()

        # Fill in 0 for abundance values without information
        df = df.replace(np.nan, 0)

        # Reshape so each row is a single sample
        df = df.pivot_table(columns=["m/z", "temp_bin"], values=["abun_minsub_scaled"])

        return df

class DataModule():
    def __init__(self, metadata, DATA_PATH):

        train_files = metadata[metadata["split"] == "train"]["features_path"].to_dict()
        val_files = metadata[metadata["split"] == "val"]["features_path"].to_dict()
        test_files = metadata[metadata["split"] == "test"]["features_path"].to_dict()

        print("Number of training samples: ", len(train_files))
        print("Number of validation samples: ", len(val_files))
        print("Number of testing samples: ", len(test_files))

        self.metadata = metadata
        self.train_files = train_files
        self.val_files = val_files
        self.test_files = test_files

        self.train_labels = pd.read_csv(os.path.join(DATA_PATH,"train_labels.csv"), index_col="sample_id")

        # Create dict with both validation and test sample IDs and paths
        self.all_test_files = val_files.copy()
        self.all_test_files.update(test_files)
        print("Total test files: ", len(self.all_test_files))


class Preprocessor():
    def __init__(self) -> None:
        pass

    def drop_frac_and_He(self, df):
        """
        Drops fractional m/z values, m/z values > 100, and carrier gas m/z

        Args:
            df: a dataframe representing a single sample, containing m/z values

        Returns:
            The dataframe without fractional an carrier gas m/z
        """

        # drop fractional m/z values
        df = df[df["m/z"].transform(round) == df["m/z"]]
        assert df["m/z"].apply(float.is_integer).all(), "not all m/z are integers"

        # drop m/z values greater than 99
        df = df[df["m/z"] < 100]

        # drop carrier gas
        df = df[df["m/z"] != 4]

        return df

    def scale_abun(self, df):
        """
        Scale abundance from 0-100 according to the min and max values across entire sample

        Args:
            df: dataframe containing abundances and m/z

        Returns:
            dataframe with additional column of scaled abundances
        """

        df["abun_minsub_scaled"] = minmax_scale(df["abundance_minsub"].astype(float))

        return df

    # Removing background ion presences
    # As mentioned in the project description, scientists may remove background noise more carefully. 
    # They may take an average of an area early in the experiment to subtract. Or, if the background 
    # noise varies over time, they may fit a function to it and subtract according to this function.
    def remove_background_abundance(self, df):
        """
        Subtracts minimum abundance value

        Args:
            df: dataframe with 'm/z' and 'abundance' columns

        Returns:
            dataframe with minimum abundance subtracted for all observations
        """

        df["abundance_minsub"] = df.groupby(["m/z"])["abundance"].transform(
            lambda x: (x - x.min())
        )

        return df

    # Preprocess function
    def preprocess_sample(self, df):
        df = self.drop_frac_and_He(df)
        df = self.remove_background_abundance(df)
        df = self.scale_abun(df)
        return df


def cli_main():

    def logloss_cross_val(clf, X, y, skf):

        # Generate a score for each label class
        log_loss_cv = {}
        for col in y.columns:

            y_col = y[col]  # take one label at a time
            log_loss_cv[col] = np.mean(
                cross_val_score(clf, X.values, y_col, cv=skf, scoring=log_loss_scorer)
            )

        avg_log_loss = np.mean(list(log_loss_cv.values()))

        return log_loss_cv, avg_log_loss

    RANDOM_SEED = 42  # For reproducibility

    # ------------
    # args
    # ------------
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--classifier', type=str, default="dummy")
    parser.add_argument('--data_dir', type=str, default=Path.cwd() / "data/final/public/")

    parser = parser.parse_args()

    metadata = pd.read_csv(parser.data_dir + "metadata.csv", index_col="sample_id")

    # ------------
    # Load data
    # ------------

    data_module = DataModule(metadata, DATA_PATH=parser.data_dir)

    # ------------
    # Preprocess data
    # ------------

    preprocessor = Preprocessor()

    # ------------
    # Feature Engineering
    # ------------
    fe = FeatureEngineering()

    #%% Assembling preprocessed and transformed training set

    train_features_dict = {}
    print("Total number of train files: ", len(data_module.train_files))

    for i, (sample_id, filepath) in enumerate(tqdm(data_module.train_files.items())):

        # Load training sample
        temp = pd.read_csv(os.path.join(parser.data_dir, filepath))

        # Preprocessing training sample
        train_sample_pp = preprocessor.preprocess_sample(temp)

        # Feature engineering
        train_sample_fe = fe.abun_per_tempbin(train_sample_pp).reset_index(drop=True)
        train_features_dict[sample_id] = train_sample_fe

    train_features = pd.concat(
        train_features_dict, names=["sample_id", "dummy_index"]
    ).reset_index(level="dummy_index", drop=True)

    # Make sure that all sample IDs in features and labels are identical
    assert train_features.index.equals(data_module.train_labels.index)

    # Define stratified k-fold validation
    skf = StratifiedKFold(n_splits=10, random_state=RANDOM_SEED, shuffle=True)

    # Define log loss
    log_loss_scorer = make_scorer(log_loss, needs_proba=True)

    if parser.classifier == 'dummy':
        # Dummy classifier
        clf = DummyClassifier(strategy="prior")

    elif parser.classifier == 'benchmark':
        clf = LogisticRegression(
            penalty="l1", solver="liblinear", C=10, random_state=RANDOM_SEED
        )
    elif parser.classifier == 'bagging':
        clf = BaggingClassifier(
            DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1
        )
    elif parser.classifier == 'DecisionTree':
        clf = DecisionTreeClassifier()
    elif parser.classifier == 'XGBClassifier':
        clf = XGBClassifier()
    else:
        assert False, "Classifier not recognized"


    # Check log loss score for baseline dummy model
    print("{} model log-loss:".format(parser.classifier))
    logloss = logloss_cross_val(clf, train_features, data_module.train_labels, skf)
    pprint(logloss[0])
    print("\nAverage log-loss")
    pprint(logloss[1])

    # ------------
    # Train model
    # ------------
    # Train logistic regression model with l1 regularization, where C = 10

    # Initialize dict to hold fitted models
    fitted_dict = {}

    # Split into binary classifier for each class
    for col in data_module.train_labels.columns:

        y_train_col = data_module.train_labels[col]  # Train on one class at a time

        # output the trained model, bind this to a var, then use as input
        # to prediction function
        fitted_dict[col] = clf.fit(train_features.values, y_train_col)  # Train


    # ------------
    # PREPARING A SUBMISSION
    # ------------
    #%%
    # Import submission format
    submission_template_df = pd.read_csv(
        os.path.join(parser.data_dir,"submission_format.csv"), index_col="sample_id"
    )
    compounds_order = submission_template_df.columns
    sample_order = submission_template_df.index

    #%%
    def predict_for_sample(sample_id, fitted_model_dict):

        # Import sample
        temp_sample = pd.read_csv(os.path.join(parser.data_dir, data_module.all_test_files[sample_id]))

        # Preprocess sample
        temp_sample = preprocessor.preprocess_sample(temp_sample)

        # Feature engineering on sample
        temp_sample = fe.abun_per_tempbin(temp_sample)

        # Generate predictions for each class
        temp_sample_preds_dict = {}

        for compound in compounds_order:
            clf = fitted_model_dict[compound]
            temp_sample_preds_dict[compound] = clf.predict_proba(temp_sample.values)[:, 1][0]
        
        return temp_sample_preds_dict


    #%% SUBMIT PREDICTION
    # Dataframe to store submissions in
    final_submission_df = pd.DataFrame(
        [
            predict_for_sample(sample_id, fitted_dict)
            for sample_id in tqdm(sample_order)
        ],
        index=sample_order,
    )

    #%%
    # Check that columns and rows are the same between final submission and submission format
    assert final_submission_df.index.equals(submission_template_df.index)
    assert final_submission_df.columns.equals(submission_template_df.columns)

    #%%
    final_submission_df.to_csv("{}_submission.csv".format(parser.classifier))


if __name__ == '__main__':
    cli_main()


# %%

#%%