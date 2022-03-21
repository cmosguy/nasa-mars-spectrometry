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
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, log_loss
from sklearn.model_selection import StratifiedKFold, cross_val_score
from tqdm import tqdm

import plotly.express as px
import plotly.graph_objects as go

pd.set_option("max_colwidth", 80)
RANDOM_SEED = 42  # For reproducibility


#%% Importing data set
DATA_PATH = Path.cwd() / "data/final/public/"
metadata = pd.read_csv(DATA_PATH / "metadata.csv", index_col="sample_id")
metadata.head()

#%%
train_files = metadata[metadata["split"] == "train"]["features_path"].to_dict()
val_files = metadata[metadata["split"] == "val"]["features_path"].to_dict()
test_files = metadata[metadata["split"] == "test"]["features_path"].to_dict()

print("Number of training samples: ", len(train_files))
print("Number of validation samples: ", len(val_files))
print("Number of testing samples: ", len(test_files))

# %% EXPLORATORY DATA ANALYSIS
# Share of samples from commercial instruments vs. SAM testbed
meta_instrument = (
    metadata.reset_index()
    .groupby(["split", "instrument_type"])["sample_id"]
    .aggregate("count")
    .reset_index()
)
meta_instrument.head()


#%%
fig = px.bar(meta_instrument, x="sample_id", y="split", color="instrument_type", text_auto=True)
fig.update_layout(title="Instrument type by dataset split", height=500, width=800)
fig.show()
#%%
train_labels = pd.read_csv(DATA_PATH / "train_labels.csv", index_col="sample_id")
train_labels.head()

#%%
sumlabs = train_labels.aggregate("sum").sort_values()
fig = px.bar(sumlabs, orientation='h', text_auto=True)
fig.update_layout(title='Compounds represented in training set', height=500, width=800, showlegend=False)
fig.update_xaxes(title_text='Count in training set')
fig.update_yaxes(title_text='Compounds')
fig.show()


#%%
# We can use some plots to understand how a few different variables relate to each other. 
# We have only four variables available to us - time, which is the time from the start of the 
# experiment, temp, the temperature that the sample is heated to at that point in time, m/z, 
# which is a "type" of ion detected, and abundance, which is the amount of the ion type 
# detected at the temperature and point in time.
#
# First, we can observe the relationship between temperature and time. Temperature 
# is supposed to be a function of time in these experiments, but the patterns of 
# ion abundance may vary as a function of both. Further, the commercial and testbed 
# samples may contain a different range of times and temperatures, so let's examine 
# a few samples of each type.

# Select sample IDs for five commercial samples and five testbed samples
sample_id_commercial = (
    metadata.query("instrument_type=='commercial'")
    .index
    .values[0:5]
)
sample_id_testbed = (
    metadata.query("instrument_type=='sam_testbed'")
    .index
    .values[0:5]
)

sample_id_commercial
#%%
# Import sample files for EDA
sample_commercial_dict = {}
sample_testbed_dict = {}

sample_data_arr = []

for i in range(0, 5):
	comm_lab = sample_id_commercial[i]
	data = pd.read_csv(DATA_PATH / train_files[comm_lab])
	data['sample_id'] = comm_lab
	data['instrument_type'] = 'commercial'
	sample_data_arr.append(data)

	test_lab = sample_id_testbed[i]
	data = pd.read_csv(DATA_PATH / train_files[test_lab])
	data['sample_id'] = test_lab
	data['instrument_type'] = 'sam_testbed'
	sample_data_arr.append(data)

sample_data = pd.concat(sample_data_arr)

sample_data.tail()

#%%
sample_data.instrument_type.unique()
#%%
fig = px.scatter(sample_data.query("instrument_type=='commercial'"), 
x="time", y="temp", facet_col="sample_id", facet_col_wrap=5, title="Commercial Samples")
fig.update_layout(width=1200)
fig.update_xaxes(title_text="Time (seconds)")
fig.update_yaxes(title_text="")
fig.show()
fig = px.scatter(sample_data.query("instrument_type=='sam_testbed'"), 
x="time", y="temp", facet_col="sample_id", facet_col_wrap=5, title="SAM testbed Samples")
fig.update_layout(width=1200)
fig.update_xaxes(title_text="Time (seconds)")
fig.update_yaxes(title_text="Temp (C)")
fig.show()

#%%
# Let's also look at the two other key values - m/z, which indicates the type of an 
# ion, and abundance, which indicates that ion type's levels across temperature and time. 
# We can visualize changes in abundance across temperature and time, which can help us 
# identify relationships within the data that chemists might consider when identifying compounds.

fig = px.scatter(sample_data.query("instrument_type=='commercial'"), 
x="temp", y="abundance", facet_col="sample_id", facet_col_wrap=5, color='m/z', title="Commercial Samples")
fig.update_layout(width=1200, showlegend=False)
fig.update_xaxes(title_text="Temp (C)")
fig.update_yaxes(title_text="Abundance")
fig.show()
fig = px.scatter(sample_data.query("instrument_type=='sam_testbed'"),
x="temp", y="abundance", facet_col="sample_id", facet_col_wrap=5, color='m/z', title="SAM testbed Samples")
fig.update_layout(width=1200, showlegend=False)
fig.update_xaxes(title_text="Temp (C)")
fig.update_yaxes(title_text="Abundance")
fig.show()
#%% Select a sample to analyze
sample_data['m/z_cat'] = sample_data['m/z'].astype('category')
sample_lab = sample_id_testbed[1]
sample_df = sample_data.query('sample_id==@sample_lab')

#%% PREPROCESSING
# Standarizing which m/z values to include
def drop_frac_and_He(df):
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

sample_df = drop_frac_and_He(sample_df)

fig = px.histogram(sample_data.query('sample_id==@sample_lab'), x='m/z', color='instrument_type', title="Before dropping selected m/x values")
fig.update_layout(width=1200, showlegend=False)
fig.update_xaxes(range=[0, 300])
fig.show()

fig = px.histogram(sample_df, color='instrument_type', x='m/z', title="After dropping selected m/x values")
fig.update_layout(width=1200, showlegend=False)
fig.update_xaxes(range=[0, 300])
fig.show()

# %% Removing background ion presences
# As mentioned in the project description, scientists may remove background noise more carefully. 
# They may take an average of an area early in the experiment to subtract. Or, if the background 
# noise varies over time, they may fit a function to it and subtract according to this function.
def remove_background_abundance(df):
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

#%% Abundance values before subtracting minimum
sample_df = remove_background_abundance(sample_df)

#%%
fig = px.scatter(sample_data.query("sample_id==@sample_lab"),
x="temp", y="abundance", color='m/z', title="Commercial Samples")
fig.update_layout(width=800, height=500, showlegend=False)
fig.update_xaxes(title_text="Temp (C)")
fig.update_yaxes(title_text="Abundance")
fig.show()
 
fig = px.scatter(sample_df,
x="temp", y="abundance_minsub", color='m/z', title="Commercial Samples")
fig.update_layout(width=800, height=500, showlegend=False)
fig.update_xaxes(title_text="Temp (C)")
fig.update_yaxes(title_text="Abundance")
fig.show()

#%% Putting it all together
def scale_abun(df):
    """
    Scale abundance from 0-100 according to the min and max values across entire sample

    Args:
        df: dataframe containing abundances and m/z

    Returns:
        dataframe with additional column of scaled abundances
    """

    df["abun_minsub_scaled"] = minmax_scale(df["abundance_minsub"].astype(float))

    return df

# Preprocess function
def preprocess_sample(df):
    df = drop_frac_and_He(df)
    df = remove_background_abundance(df)
    df = scale_abun(df)
    return df


#%%
sample_data['m/z'] = sample_data['m/z'].astype(float)
sample_data_preprocessed = sample_data.groupby('sample_id').apply(preprocess_sample)

fig = px.scatter(sample_data_preprocessed.query("instrument_type=='commercial'"), 
x="temp", y="abun_minsub_scaled", facet_col="sample_id", facet_col_wrap=5, color='m/z', title="Commercial Samples")
fig.update_layout(width=1200, showlegend=False)
fig.update_xaxes(title_text="Temp (C)")
fig.update_yaxes(title_text="Abundance")
fig.show()
fig = px.scatter(sample_data_preprocessed.query("instrument_type=='sam_testbed'"),
x="temp", y="abun_minsub_scaled", facet_col="sample_id", facet_col_wrap=5, color='m/z', title="SAM testbed Samples")
fig.update_layout(width=1200, showlegend=False)
fig.update_xaxes(title_text="Temp (C)")
fig.update_yaxes(title_text="Abundance")
fig.show()

#%% Feature Engineering
# For our analysis, we will engineer a simple set of features that try to capture some of these characteristics. 
# We will discretize the overall temperature range into bins (of 100 degrees), and calculate maximum relative 
# abundance within that temperature bin for each m/z value.
#
# There are many ways to describe the shapes of these peaks with higher fidelity than the approach we demonstrate here

# Create a series of temperature bins
temprange = pd.interval_range(start=-100, end=1500, freq=100)
temprange

# Make dataframe with rows that are combinations of all temperature bins
# and all m/z values
allcombs = list(itertools.product(temprange, [*range(0, 100)]))

allcombs_df = pd.DataFrame(allcombs, columns=["temp_bin", "m/z"])
allcombs_df.sample(20)

#%%
def abun_per_tempbin(df):

    """
    Transforms dataset to take the preprocessed max abundance for each
    temperature range for each m/z value

    Args:
        df: dataframe to transform

    Returns:
        transformed dataframe
    """

    # Bin temperatures
    df["temp_bin"] = pd.cut(df["temp"], bins=temprange)

    # Combine with a list of all temp bin-m/z value combinations
    df = pd.merge(allcombs_df, df, on=["temp_bin", "m/z"], how="left")

    # Aggregate to temperature bin level to find max
    df = df.groupby(["temp_bin", "m/z"]).max("abun_minsub_scaled").reset_index()

    # Fill in 0 for abundance values without information
    df = df.replace(np.nan, 0)

    # Reshape so each row is a single sample
    df = df.pivot_table(columns=["m/z", "temp_bin"], values=["abun_minsub_scaled"])

    return df


#%% Assembling preprocessed and transformed training set

train_features_dict = {}
print("Total number of train files: ", len(train_files))

for i, (sample_id, filepath) in enumerate(tqdm(train_files.items())):

    # Load training sample
    temp = pd.read_csv(DATA_PATH / filepath)

    # Preprocessing training sample
    train_sample_pp = preprocess_sample(temp)

    # Feature engineering
    train_sample_fe = abun_per_tempbin(train_sample_pp).reset_index(drop=True)
    train_features_dict[sample_id] = train_sample_fe

train_features = pd.concat(
    train_features_dict, names=["sample_id", "dummy_index"]
).reset_index(level="dummy_index", drop=True)

#%%
train_features.head()

#%%
# Make sure that all sample IDs in features and labels are identical
assert train_features.index.equals(train_labels.index)

#%% PERFORM MODELING
# This competition's task is a multi-label classification problem 
# with 10 label classes—each observation can belong to any number 
# of the label classes. One simple modeling approach for multi-label 
# classification is "one vs. all", in which we create a binary classifier 
# for each label class independently. Then, each binary classifier's 
# predictions are simply concatenated together at the end for the overall 
# prediction. For this benchmark, we will use logistic regression for each 
# classifier as a first-pass modeling approach.

# Define stratified k-fold validation
skf = StratifiedKFold(n_splits=10, random_state=RANDOM_SEED, shuffle=True)

# Define log loss
log_loss_scorer = make_scorer(log_loss, needs_proba=True)

#%% BASELINE DUMMY CLASSIFIER
# Check log loss score for baseline dummy model
def logloss_cross_val(clf, X, y):

    # Generate a score for each label class
    log_loss_cv = {}
    for col in y.columns:

        y_col = y[col]  # take one label at a time
        log_loss_cv[col] = np.mean(
            cross_val_score(clf, X.values, y_col, cv=skf, scoring=log_loss_scorer)
        )

    avg_log_loss = np.mean(list(log_loss_cv.values()))

    return log_loss_cv, avg_log_loss

# Dummy classifier
dummy_clf = DummyClassifier(strategy="prior")

print("Dummy model log-loss:")
dummy_logloss = logloss_cross_val(dummy_clf, train_features, train_labels)
pprint(dummy_logloss[0])
print("\nAverage log-loss")
dummy_logloss[1]

#%% LOGISTIC REGRESSION
# Define logistic regression model
logreg_clf = LogisticRegression(
    penalty="l1", solver="liblinear", C=10, random_state=RANDOM_SEED
)
print("Logistic regression model log-loss:\n")
logreg_logloss = logloss_cross_val(logreg_clf, train_features, train_labels)
pprint(logreg_logloss[0])
print("Average log-loss")
logreg_logloss[1]

#%% Training the model on all the data
# Train logistic regression model with l1 regularization, where C = 10

# Initialize dict to hold fitted models
def logistic_regression_fit(train_features, train_labels):
    fitted_logreg_dict = {}

    # Split into binary classifier for each class
    for col in train_labels.columns:

        y_train_col = train_labels[col]  # Train on one class at a time

        # output the trained model, bind this to a var, then use as input
        # to prediction function
        clf = LogisticRegression(
            penalty="l1", solver="liblinear", C=10, random_state=RANDOM_SEED
        )
        fitted_logreg_dict[col] = clf.fit(train_features.values, y_train_col)  # Train

        return fitted_logreg_dict

fitted_logreg_dict = logistic_regression_fit(train_features, train_labels)

#%%
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

bag_clf = BaggingClassifier(
    DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1
)
bagging_logloss = logloss_cross_val(bag_clf, train_features, train_labels)
pprint(logreg_logloss[0])
print("Average bagging log-loss")
bagging_logloss[1]


#%%
forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)

forest_logloss = logloss_cross_val(forest_clf, train_features, train_labels)
pprint(forest_logloss[0])
print("Average random forest log-loss")
forest_logloss[1]
#%%
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC

voting_clf = VotingClassifier(
    estimators=[
        ("lr", logreg_clf), 
        ("rf", RandomForestClassifier(n_estimators=100)),
        ("svc", SVC(probability=True, random_state=RANDOM_SEED)),
        ("bag", bag_clf)
        ],
    voting="soft",
)
print("Logistic regression model log-loss:\n")
voting_logloss = logloss_cross_val(voting_clf, train_features, train_labels)
pprint(voting_logloss[0])
print("Average voting log-loss")
voting_logloss[1]

#%%

def bagging_classifier_fit(train_features, train_labels):
    fitted_baggging_dict = {}

    # Split into binary classifier for each class
    for col in train_labels.columns:

        y_train_col = train_labels[col]  # Train on one class at a time

        # output the trained model, bind this to a var, then use as input
        # to prediction function
        bag_clf = BaggingClassifier(
            DecisionTreeClassifier(), n_estimators=500, max_samples=100, bootstrap=True, n_jobs=-1
        )
        fitted_baggging_dict[col] = bag_clf.fit(train_features.values, y_train_col)  # Train

    return fitted_baggging_dict

fitted_baggging_dict  = bagging_classifier_fit(train_features, train_labels)

#%%
def classifier_fit(clf, train_features, train_labels):
    fitted_classifier_dict = {}

    # Split into binary classifier for each class
    for col in train_labels.columns:

        y_train_col = train_labels[col]  # Train on one class at a time

        # output the trained model, bind this to a var, then use as input
        # to prediction function
        fitted_classifier_dict[col] = clf.fit(train_features.values, y_train_col)  # Train

    return fitted_classifier_dict

fitted_voting_dict  = classifier_fit(voting_clf, train_features, train_labels)
#%% PREPARING A SUBMISSION
# Create dict with both validation and test sample IDs and paths
all_test_files = val_files.copy()
all_test_files.update(test_files)
print("Total test files: ", len(all_test_files))

#%%
# Import submission format
submission_template_df = pd.read_csv(
    DATA_PATH / "submission_format.csv", index_col="sample_id"
)
compounds_order = submission_template_df.columns
sample_order = submission_template_df.index

#%%
def predict_for_sample(sample_id, fitted_model_dict):

    # Import sample
    temp_sample = pd.read_csv(DATA_PATH / all_test_files[sample_id])

    # Preprocess sample
    temp_sample = preprocess_sample(temp_sample)

    # Feature engineering on sample
    temp_sample = abun_per_tempbin(temp_sample)

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
        predict_for_sample(sample_id, fitted_voting_dict)
        for sample_id in tqdm(sample_order)
    ],
    index=sample_order,
)
#%%
fitted_voting_dict.keys()

#%%
# Check that columns and rows are the same between final submission and submission format
assert final_submission_df.index.equals(submission_template_df.index)
assert final_submission_df.columns.equals(submission_template_df.columns)

#%%
final_submission_df.to_csv("voting_submission.csv")
