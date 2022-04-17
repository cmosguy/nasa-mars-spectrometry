#%%
from pathlib import Path
import pickle
from pprint import pprint
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from tqdm import tqdm
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
if __name__ == "__main__":
	dataset = Dataset()

	processed_splits = get_processed_splits(dataset)

	# %% Dummy classifier
	# Dummy classifier
	def dummy_classifier(train_features, train_labels):
		dummy_clf = DummyClassifier(strategy="prior")

		dummy_logloss = logloss_cross_val(dummy_clf, train_features, train_labels)
		print("Dummy model log-loss:")
		pprint(dummy_logloss[0])
		print("\nAverage log-loss")
		dummy_logloss[1]

		return dummy_clf

	compounds_order, sample_order =  get_compounds_order()

	dummy_clf = dummy_classifier(processed_splits['train'], dataset.train_labels)
	fitted_dict = train_classifier(dummy_clf, processed_splits, dataset.train_labels)
	make_submission_file(fitted_dict, processed_splits, 'dummy', compounds_order, sample_order)

	sample_id = 'S0000'
	pprint(predict_for_sample(sample_id, fitted_dict, processed_splits, compounds_order))
	pprint(dataset.train_labels.loc[sample_id])

	#%%
	def logreg_classifier(train_features, train_labels):

		clf = LogisticRegression(
			penalty="l1", solver="liblinear", C=10, random_state=config.RANDOM_SEED
		)
		logloss = logloss_cross_val(clf, train_features, train_labels)
		print("LogisticRegression model log-loss:")
		pprint(logloss[0])
		print("\nAverage log-loss")
		logloss[1]

		return clf

	logreg_clf = logreg_classifier(processed_splits['train'], dataset.train_labels)
	fitted_dict = train_classifier(logreg_clf, processed_splits, dataset.train_labels)
	logreg_sub_df = make_submission_file(fitted_dict, processed_splits, 'logreg', compounds_order, sample_order)

# %%
def train_logreg(processed_splits, train_labels):

	print('\nLogisticRegression cross-validation check')

	fitted_dict = {}

	# Split into binary classifier for each class
	for col in train_labels.columns:

		y_train_col = train_labels[col]  # Train on one class at a time

		clf = LogisticRegression(
			penalty="l1", solver="liblinear", C=10, random_state=config.RANDOM_SEED
			# penalty="l2", solver="newton-cg", C=60, random_state=config.RANDOM_SEED
		)
		fitted_dict[col] = clf.fit(processed_splits['train'].values, y_train_col)  # Train

	logloss = logloss_cross_val(fitted_dict, processed_splits['train'], train_labels)
	print("LogisticRegression model log-loss:")
	pprint(logloss[0])
	print("\nAverage log-loss: ", logloss[1])

	return fitted_dict

logreg_fitted = train_logreg(processed_splits, dataset.train_labels)
logreg_sub_df = make_submission_file(logreg_fitted, processed_splits, 'logreg', compounds_order, sample_order)
logreg_sub_df.head()


#%%
sample_id = 'S0001'
pprint(predict_for_sample(sample_id, fitted_dict, processed_splits, compounds_order))
print('\n original training label:')
pprint(dataset.train_labels.loc[sample_id])

#%%
import warnings 
warnings.filterwarnings("ignore")

def train_logreg_grid_cv(processed_splits, train_labels):

	# print('\nLogisticRegression cross-validation check')

	# clf = GridSearchCV(LogisticRegression(), param_grid, cv=5, scoring='log_loss')
	# print(clf.best_params_)
	# logloss = logloss_cross_val(clf, processed_splits['train'], train_labels)
	# print("LogisticRegression model log-loss:")
	# pprint(logloss[0])
	# print("\nAverage log-loss")
	# logloss[1]

	param_grid = {
		'C': np.arange(0.01, 100, 10),
		'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
		'penalty': ['none', 'l1', 'l2'],
		'random_state': [config.RANDOM_SEED]
	}
	param_grid = {
		'C': np.arange(0.01, 20, 1),
		'penalty': ['none', 'l1', 'l2'],
		'random_state': [config.RANDOM_SEED], 
		'solver': ['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'],
		}

	fitted_dict = {}

	log_loss_scorer = make_scorer(log_loss, greater_is_better=False, needs_proba=True)

	# Split into binary classifier for each class
	for col in train_labels.columns:

		y_train_col = train_labels[col]  # Train on one class at a time

		# Gridsearch to determine value of C
		logreg_cv = GridSearchCV(LogisticRegression(), param_grid, cv=5, return_train_score=True, scoring=log_loss_scorer)
		logreg_cv.fit(processed_splits['train'].values, y_train_col)  # Train
		print('class:', col )
		print(logreg_cv.best_params_)

		bestlogreg = logreg_cv.best_estimator_
		bestlogreg.fit(processed_splits['train'].values, y_train_col)  # Train
		# bestlogreg.coef_ = bestlogreg.named_steps['logreg'].coef_
		bestlogreg.score(processed_splits['train'].values, y_train_col)

		fitted_dict[col] = bestlogreg

	return fitted_dict

logreg_grid = train_logreg_grid_cv(processed_splits, dataset.train_labels)
logreg_grid_sub_df  = make_submission_file(logreg_grid, processed_splits, 'logreg_grid_cv', compounds_order, sample_order)
pickle.dump(logreg_grid_sub_df, open('logreg_grid_sub_df.pkl', 'wb'))


def logloss_cross_val(clf, X, y):
	skf = StratifiedKFold(n_splits=10, random_state=config.RANDOM_SEED, shuffle=True)

	# Define log loss
	log_loss_scorer = make_scorer(log_loss, needs_proba=True)

	# Generate a score for each label class
	log_loss_cv = {}
	for col in y.columns:

		y_col = y[col]  # take one label at a time
		log_loss_cv[col] = np.mean(
			cross_val_score(clf[col], X.values, y_col, cv=skf, scoring=log_loss_scorer)
		)

	avg_log_loss = np.mean(list(log_loss_cv.values()))

	return log_loss_cv, avg_log_loss
logloss = logloss_cross_val(logreg_grid, processed_splits['train'], dataset.train_labels)
print("LogisticRegression model log-loss:")
pprint(logloss[0])
print("\nAverage log-loss")
logloss[1]
#%%
logreg_grid['basalt'].get_params()
#%%
sample_id = 'S0006'
pprint(predict_for_sample(sample_id, logreg_grid, processed_splits, compounds_order))
print('\n original training label:')
pprint(dataset.train_labels.loc[sample_id])

# %%

sample_id = 'S0768'
pprint(predict_for_sample(sample_id, logreg_grid, processed_splits, compounds_order))
# print('\n original training label:')
# pprint(dataset.train_labels.loc[sample_id])