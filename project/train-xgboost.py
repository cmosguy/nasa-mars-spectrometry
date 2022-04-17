#%%
from pathlib import Path
import pickle
from pprint import pprint
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
forest = RandomForestClassifier(random_state=config.RANDOM_SEED)
multi_forest_target = MultiOutputClassifier(forest, n_jobs=-1)
multi_forest_target.fit(processed_splits['train'], dataset.train_labels)

print('Accuracy on test data: {:.1f}%'.format(accuracy_score(dataset.val_labels, multi_forest_target.predict(processed_splits['val']))*100))

#%%
svc = SVC(gamma="scale")
multi_svc = MultiOutputClassifier(estimator=svc, n_jobs=-1)
multi_svc.fit(processed_splits['train'], dataset.train_labels)
print('Accuracy on test data: {:.1f}%'.format(accuracy_score(dataset.val_labels, multi_svc.predict(processed_splits['val']))*100))
