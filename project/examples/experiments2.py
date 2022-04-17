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
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline

from tqdm import tqdm

import plotly.express as px
import plotly.graph_objects as go

import os


from argparse import ArgumentParser


#%%
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

        self.train_labels = pd.read_csv(os.path.join(DATA_PATH,"train_labels.csv"), index_col="sample_id")
        self.test_labels = pd.read_csv(os.path.join(DATA_PATH,"test_labels.csv"), index_col="sample_id")

        # Create dict with both validation and test sample IDs and paths
        self.all_test_files = val_files.copy()
        self.all_test_files.update(test_files)
        print("Total test files: ", len(self.all_test_files))


class DropFracAndHe(BaseEstimator, TransformerMixin):
    """
    Drops fractional m/z values, m/z values > 100, and carrier gas m/z

    Args:
        df: a dataframe representing a single sample, containing m/z values

    Returns:
        The dataframe without fractional an carrier gas m/z
    """
    def __init__(self, drop_less_value=100, carrier_gas=4):
        self.drop_less_value = drop_less_value
        self.carrier_gas = carrier_gas

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # drop fractional m/z values
        X = X[X["m/z"].transform(round) == X["m/z"]]

        assert X["m/z"].apply(float.is_integer).all(), "not all m/z are integers"

        # drop m/z values greater than 99
        X = X[X["m/z"] < self.drop_less_value]

        # drop carrier gas
        X = X[X["m/z"] != self.carrier_gas]

        return X

class ScaleAbundance(BaseEstimator, TransformerMixin):
    """
    Scale abundance from 0-100 according to the min and max values across entire sample

    Args:
        df: dataframe containing abundances and m/z

    Returns:
        dataframe with additional column of scaled abundances
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X["abun_minsub_scaled"] = (X
        .assign(abundance_minsub=X.abundance_minsub.astype(float))
        .groupby('sample_id')
        .abundance_minsub
        .transform(minmax_scale)
        )

        return X

    # Removing background ion presences
    # As mentioned in the project description, scientists may remove background noise more carefully. 
    # They may take an average of an area early in the experiment to subtract. Or, if the background 
    # noise varies over time, they may fit a function to it and subtract according to this function.
class RemoveBackgroundAbundance(BaseEstimator, TransformerMixin):
    """
    Subtracts minimum abundance value

    Args:
        df: dataframe with 'm/z' and 'abundance' columns

    Returns:
        dataframe with minimum abundance subtracted for all observations
    """
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X["abundance_minsub"] = X.groupby(['sample_id',"m/z"])["abundance"].transform(
            lambda x: (x - x.min())
        )

        return X

# Feature Engineering
# For our analysis, we will engineer a simple set of features that try to capture some of these characteristics. 
# We will discretize the overall temperature range into bins (of 100 degrees), and calculate maximum relative 
# abundance within that temperature bin for each m/z value.
#
# There are many ways to describe the shapes of these peaks with higher fidelity than the approach we demonstrate here

# Create a series of temperature bins
class AbundancePerTemperatureBinFeature(BaseEstimator, TransformerMixin):
    """
    Transforms dataset to take the preprocessed max abundance for each
    temperature range for each m/z value

    Args:
        df: dataframe to transform

    Returns:
        transformed dataframe
    """

    def __init__(self) -> None:
        self.temprange = pd.interval_range(start=-100, end=1500, freq=100)

        # Make dataframe with rows that are combinations of all temperature bins
        # and all m/z values
        allcombs = list(itertools.product(self.temprange, [*range(0, 100)]))

        self.allcombs_df = pd.DataFrame(allcombs, columns=["temp_bin", "m/z"])

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        # Bin temperatures
        X["temp_bin"] = pd.cut(X["temp"], bins=self.temprange)

        X['m/z'] = X[['m/z']].astype(int)

        # Combine with a list of all temp bin-m/z value combinations
        X = X.reset_index().merge(a.allcombs_df, on=["temp_bin", "m/z"], how="left" ).set_index('sample_id')

        # Aggregate to temperature bin level to find max
        X = X.groupby(['sample_id',"temp_bin", "m/z"]).max("abun_minsub_scaled").reset_index()

        # Fill in 0 for abundance values without information
        X = X.replace(np.nan, 0)

        # Reshape so each row is a single sample
        X = X.pivot_table(columns=['sample_id',"m/z", "temp_bin"], values=["abun_minsub_scaled"])

        return X


#%%
data_dir = './data/final/public/'

RANDOM_SEED = 42  # For reproducibility

metadata = pd.read_csv(data_dir + "metadata.csv", index_col="sample_id")

# ------------
# Load data
# ------------

data_module = DataModule(metadata, DATA_PATH=data_dir)

#%% Assembling preprocessed and transformed training set

train_features_dict = {}
print("Total number of train files: ", len(data_module.train_files))

for i, (sample_id, filepath) in enumerate(tqdm(data_module.train_files.items())):

    # Load training sample
    temp = pd.read_csv(os.path.join(data_dir, filepath))

    train_features_dict[sample_id] = temp

train_features = pd.concat(
    train_features_dict, names=["sample_id", "dummy_index"]
).reset_index(level="dummy_index", drop=True)

#%%
# data_module.train_labels.head(10)
# Make sure that all sample IDs in features and labels are identical
# assert train_features.index.equals(data_module.train_labels.index)

# Define stratified k-fold validation
skf = StratifiedKFold(n_splits=10, random_state=RANDOM_SEED, shuffle=True)

# Define log loss
log_loss_scorer = make_scorer(log_loss, needs_proba=True)

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

scoring = {'AUC': 'roc_auc', 'Accuracy': log_loss_scorer}

#%%
feature_pipeline = Pipeline(
    steps = [
        ("drop_frac_and_He", DropFracAndHe(drop_less_value=100, carrier_gas=4)),
        ("remove_background_abundance", RemoveBackgroundAbundance()),
        ('scale_abun', ScaleAbundance()),
        ('feature_engineering_abun_per_tempbin', AbundancePerTemperatureBinFeature()),
        ('model', LogisticRegression(penalty="l1", solver="liblinear", C=10, random_state=RANDOM_SEED))
    ]
)

#%%
train_features.head()
#%%

# Initialize dict to hold fitted models
fitted_dict = {}

# Split into binary classifier for each class
for col in data_module.train_labels.columns:

    y_train_col = data_module.train_labels[col]  # Train on one class at a time

    # output the trained model, bind this to a var, then use as input
    # to prediction function
    fitted_dict[col] = feature_pipeline.fit_transform(train_features.values, y_train_col)  # Train
    print("model score {}")

#%%
train
#%%
grid_search = GridSearchCV(
    DecisionTreeClassifier(random_state=RANDOM_SEED),
    param_grid={'min_samples_split': range(2, 403, 10)},
    scoring=scoring,
    refit='AUC',
    return_train_score=True
)


#%%
def poo:
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


cli_main(data_dir=data_dir)


#%%
if __name__ == '__main__':
    # ------------
    # args
    # ------------
    parser = ArgumentParser(add_help=True)
    parser.add_argument('--classifier', type=str, default="dummy")
    parser.add_argument('--data_dir', type=str, default=Path.cwd() / "data/final/public/")

    parser = parser.parse_args()

    cli_main(data_dir=parser.data_dir)


# %%

# %%

#%%