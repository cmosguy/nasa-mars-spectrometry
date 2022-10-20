#%%
from pathlib import Path
import pickle
from pprint import pprint
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.multioutput import MultiOutputClassifier
from sklearn.svm import SVC
from tqdm import tqdm
import xgboost as xgb
from xgboost import XGBClassifier
import config
from sklearn.dummy import DummyClassifier

from dataset import Dataset
from preproccess import (
	get_processed_splits
)

from loss import logloss_cross_val

from util import (
	train_classifier,
	make_submission_file,
	predict_for_sample,
	get_compounds_order)


#%%
dataset = Dataset()
processed_splits = get_processed_splits(dataset)

#%%
xgb_clf = XGBClassifier(objective='binary:logistic', n_estimators=1000, use_label_encoder=False)
multilabel_model = MultiOutputClassifier(xgb_clf)
X_train = processed_splits['train']
X_train.columns = range(len(X_train.columns))
multilabel_model.fit(X_train, dataset.train_labels, eval_metric='logloss')

# evaluate on test data
print('Accuracy on test data: {:.1f}%'.format(accuracy_score(dataset.val_labels, multilabel_model.predict(processed_splits['val']))*100))

#%%
def plot_predt(y: np.ndarray, y_predt: np.ndarray, name: str) -> None:
    s = 25
    plt.scatter(y[:, 0], y[:, 1], c="navy", s=s, edgecolor="black", label="data")
    plt.scatter(
        y_predt[:, 0], y_predt[:, 1], c="cornflowerblue", s=s, edgecolor="black"
    )
    plt.xlim([-1, 2])
    plt.ylim([-1, 2])
    plt.show()

# Train a regressor on it
reg = xgb.XGBRegressor(tree_method="hist", n_estimators=64)
reg.fit(X_train, dataset.train_labels, eval_set=[(X_train, dataset.train_labels)], eval_metric=['logloss'], early_stopping_rounds=10, verbose=True)

y_predt = reg.predict(X_train)
plot_predt(dataset.train_labels, y_predt, "multi")

#%%
reg.predict_proba(X_train)

#%%
compounds_order, sample_order =  get_compounds_order()

def predict_for_sample(sample_id: str, clf, processed_splits: dict, compounds_order: list):
	temp_sample_preds_dict = {}

	temp_sample = None
	for split, data in processed_splits.items():
		if sample_id in data.index:
			temp_sample = data.loc[sample_id]

	if temp_sample is None:
		assert 'Sample not found: ' | sample_id

	return dict(zip(compounds_order, clf.predict(temp_sample.values.reshape(1, -1))[0]))

print ("Generating predictions...")
final_submission_df = pd.DataFrame(
	[
		predict_for_sample(sample_id, reg, processed_splits, compounds_order) for sample_id in tqdm(sample_order)
	],
	index=sample_order,
)
filename = "xgboost_submission.csv"
print("output PREDICTIONS file: ", filename)
final_submission_df.to_csv(filename)
