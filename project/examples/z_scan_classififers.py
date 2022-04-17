#%%
from experiments import *

pd.set_option("max_colwidth", 80)

RANDOM_SEED = 42  # For reproducibility
DATA_PATH = Path.cwd() / "data/final/public/"
metadata = pd.read_csv(DATA_PATH + "metadata.csv", index_col="sample_id")
data_module = DataModule(metadata, DATA_PATH=DATA_PATH)

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

sample_df = drop_frac_and_He(sample_df)

fig = px.histogram(sample_data.query('sample_id==@sample_lab'), x='m/z', color='instrument_type', title="Before dropping selected m/x values")
fig.update_layout(width=1200, showlegend=False)
fig.update_xaxes(range=[0, 300])
fig.show()

fig = px.histogram(sample_df, color='instrument_type', x='m/z', title="After dropping selected m/x values")
fig.update_layout(width=1200, showlegend=False)
fig.update_xaxes(range=[0, 300])
fig.show()


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