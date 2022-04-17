#%%
from cProfile import label
import itertools
from pathlib import Path
from pprint import pprint
from time import time
from unicodedata import name

from matplotlib import pyplot as plt, cm
import numpy as np
import pandas as pd
from pandas_path import path
from sklearn import datasets
from sklearn.cluster import KMeans
from sklearn.datasets import make_multilabel_classification
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.manifold import TSNE
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import minmax_scale
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, hamming_loss, make_scorer, log_loss
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.svm import SVC
from skmultilearn.model_selection import iterative_train_test_split
from skmultilearn.dataset import load_dataset

# Multi Label Pkgs
from skmultilearn.problem_transform import BinaryRelevance
from skmultilearn.problem_transform import ClassifierChain
from skmultilearn.problem_transform import LabelPowerset
from skmultilearn.adapt import MLkNN

import plotly.express as px

from tqdm import tqdm

pd.set_option("max_colwidth", 80)
#%%
train_labels_classes.query('compounds=="oxychlorine"')
#%%
samples_df = train_features.head()
samples_df

#%%
samples_df_m = pd.melt(
        samples_df.reset_index(), 
        id_vars=[('sample_id','')], 
        value_vars=list(samples_df.columns), 
        value_name='abundance')

samples_df_m = (samples_df_m
    .assign(temp_bin=samples_df_m.temp_bin.astype(str))
    .rename(columns={('sample_id',''):'sample_id'})
)
samples_df_m.sample(20)
#%%
fig = px.bar(
    samples_df_m[samples_df_m.sample_id.isin(['S0000','S0001','S0002']) & 
    samples_df_m['m/z'].isin(range(10))], 
    x="temp_bin", y="abundance", color="sample_id", facet_row='m/z', barmode='group',)
fig.update_layout(title='Abundance per temperature bin', height=1200)
fig.show()

#%%

#%%
samples_df_m[
    samples_df_m.sample_id.isin(['S0000','S0001']) & 
    samples_df_m['m/z'].isin([0,1])]

#%%
train_labels.head()


# %% PERFORM MODELING

# %%
#%%
X,Y = make_multilabel_classification(n_samples=10, n_classes=3, n_labels=10)
X_train, y_train, X_test, y_test = iterative_train_test_split(X,Y,test_size=0.20)
X_test
#%%
# Loading the Digits dataset
digits = datasets.load_digits()

#%%
# X_train, X_test, y_train, y_test = train_test_split(train_features.to_numpy(),train_labels,test_size=0.20, random_state=RANDOM_SEED)
X_train, y_train, X_test, y_test = iterative_train_test_split(train_features.to_numpy(),train_labels.to_numpy(),test_size=0.20)
#%%
binary_rel_clf = BinaryRelevance(SVC())
binary_rel_clf.fit(X_train, y_train)

br_prediction = binary_rel_clf.predict(X_test)
accuracy_score(y_test,br_prediction)

#%%
parameters = [
    {
        'classifier': [MultinomialNB()],
        'classifier__alpha': [0.7, 1.0],
    },
    {
        'classifier': [SVC()],
        'classifier__kernel': ['rbf', 'linear'],
    },
]

clf = GridSearchCV(BinaryRelevance(), parameters, scoring='accuracy')
clf.fit(X_train, y_train)

print (clf.best_params_, clf.best_score_)

#%%
grid_prediction = clf.predict(X_test)
accuracy_score(y_test,grid_prediction)

#%%
#%%
def build_model(model,mlb_estimator,xtrain,ytrain,xtest,ytest):
    # Create an Instance
    clf = mlb_estimator(model)
    clf.fit(xtrain,ytrain)
    # Predict
    clf_predictions = clf.predict(xtest)
    
    # Check For Accuracy
    acc = accuracy_score(ytest,clf_predictions)
    ham = hamming_loss(ytest,clf_predictions)
    log_loss_score = log_loss(ytest,clf_predictions.toarray())
    result = {"accuracy:":acc,"hamming_score":ham, "log_loss":log_loss_score}
    return result

clf_chain_model = build_model(MultinomialNB(),ClassifierChain,X_train,y_train,X_test,y_test)

clf_chain_model
#%%
clf_labelP_model = build_model(MultinomialNB(),LabelPowerset,X_train,y_train,X_test,y_test)
clf_labelP_model

#%%
def make_class(t, cols):
    res = [i for i, val in enumerate(t) if val]
    return ','.join([cols[i] for index, i in enumerate(res)])
classes = pd.DataFrame(y_train, columns=list(train_labels.columns))
classes['compounds'] = classes.apply(lambda row: make_class(row, list(train_labels.columns)), axis=1)
classes.reset_index(inplace=True)
classes.drop(columns=classes.columns[0:11], inplace=True)
classes['compounds'] = classes['compounds'].apply(lambda x: 'unknown' if x == '' else x)
classes.info()
#%%
classes.compounds.unique()

#%%
train_labels_classes = pd.DataFrame(train_labels ,columns=list(train_labels.columns))
train_labels_classes['compounds'] = train_labels_classes.apply(lambda row: make_class(row, list(train_labels.columns)), axis=1)
train_labels_classes.query('compounds == "basalt"')


#%%
pca2 = PCA(n_components=2)
principalComponents = pca2.fit_transform(X_train)
principalDf2 = pd.DataFrame(data = principalComponents, columns = ['principal component 1', 'principal component 2'])
final_pca2 = pd.concat([principalDf2, classes], axis = 1)
final_pca2.head()


#%%
fig = px.scatter(final_pca2, x="principal component 1", y="principal component 2", color="compounds",)
fig.show()

#%%
pca3 = PCA(n_components=3)
principalComponents = pca3.fit_transform(X_train)
principalDf3 = pd.DataFrame(data = principalComponents, columns = ['comp1', 'comp2', 'comp3'])
final_pca3 = pd.concat([principalDf3, classes], axis = 1)
final_pca3.head()
#%%
fig = px.scatter_3d(final_pca3, x="comp1", y="comp2", z="comp3", color="compounds",)
fig.update_layout(height=1000)
fig.show()
#%%
svd = TruncatedSVD(n_components=50, n_iter=7, random_state=RANDOM_SEED)
svd_result = svd.fit_transform(X_train)
print("Explained variation per principal component: \n {}".format(
    svd.explained_variance_ratio_))
print("Total Explained variation by the first {} components: \n{}".format(
    50, svd.explained_variance_ratio_.sum()))
# plot the first 3 component
plt.plot(svd.components_[0, :])
plt.plot(svd.components_[1, :])
plt.plot(svd.components_[2, :])
#%%
time_start = time()
tsne = TSNE(init='pca', n_components=2, verbose=1, perplexity=40, n_iter=300, learning_rate='auto')
tsne_results = tsne.fit_transform(svd_result)
print('t-SNE done! Time elapsed: {} seconds'.format(time()-time_start))

#%%
tsne_results.shape
classes.shape

#%%
fig = px.scatter((classes
.assign(tsne_pca_one=tsne_results[:,0])
.assign(tsne_pca_two=tsne_results[:,1])
), 
x='tsne_pca_one', y='tsne_pca_two', color="compounds",)
fig.update_layout(height=1000)
fig.show()

#%%
len(classes.compounds.unique())

#%%
kmeans = KMeans(n_clusters=3, random_state=0).fit(X_train)
kmeans.cluster_centers_
# label infos
group_label = kmeans.labels_
#%%
group_label = pd.Series(group_label)
group_label.value_counts()

#%%
plt.plot(kmeans.cluster_centers_[0])
center1 = [train_features.columns[i]
           for i in np.where(kmeans.cluster_centers_[0] > 0.1)[0].tolist()]
center1
# What are these kmeans.cluster_centers_
# for cluster_center 2:
plt.plot(kmeans.cluster_centers_[1])
# feature_names=v.get_feature_names()
center2 = [train_features.columns[i]
           for i in np.where(kmeans.cluster_centers_[1] > 0.4)[0].tolist()]
center2
plt.plot(kmeans.cluster_centers_[2])

#%%
group_label.unique()
#%%
kmeans = KMeans(n_clusters=38, random_state=0).fit(tsne_results)
fig = px.scatter((classes
.assign(tsne_pca_one=tsne_results[:,0])
.assign(tsne_pca_two=tsne_results[:,1])
.assign(kmeans_cluster=pd.Series(kmeans.labels_).astype('category'))
), 
x='tsne_pca_one', y='tsne_pca_two', color="kmeans_cluster",)
fig.update_layout(height=1000)
fig.show()
#%%
px.data.medals_wide().head()

#%%
list(final_df.columns[2:])




# %%
# Train logistic regression model with l1 regularization, where C = 10

# Initialize dict to hold fitted models
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
# %%
# Create dict with both validation and test sample IDs and paths
all_test_files = val_files.copy()
all_test_files.update(test_files)
print("Total test files: ", len(all_test_files))
# %%
# Import submission format
submission_template_df = pd.read_csv(
    DATA_PATH / "submission_format.csv", index_col="sample_id"
)
compounds_order = submission_template_df.columns
sample_order = submission_template_df.index
# %%
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
# %%
# Dataframe to store submissions in
final_submission_df = pd.DataFrame(
    [
        predict_for_sample(sample_id, fitted_logreg_dict) for sample_id in tqdm(sample_order)
    ],
    index=sample_order,
)
# %%
final_submission_df.head()
#%%
# Check that columns and rows are the same between final submission and submission format
assert final_submission_df.index.equals(submission_template_df.index)
assert final_submission_df.columns.equals(submission_template_df.columns)
#%%
# Save submission
final_submission_df.to_csv("benchmark_logreg_submission.csv")
