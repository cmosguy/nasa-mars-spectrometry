#%%
# adapted from: http://www.codiply.com/blog/hyperparameter-grid-search-across-multiple-models-in-scikit-learn/

import numpy as np
import pandas as pd

from sklearn.model_selection import GridSearchCV

class EstimatorSelectionHelper:
    
    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
    
    def fit(self, X, y, **grid_kwargs):
        for key in self.keys:
            print('Running GridSearchCV for %s.' % key)
            model = self.models[key]
            params = self.params[key]
            grid_search = GridSearchCV(model, params, **grid_kwargs)
            grid_search.fit(X, y)
            self.grid_searches[key] = grid_search
        print('Done.')
    
    def score_summary(self, sort_by='mean_test_score'):
        frames = []
        for name, grid_search in self.grid_searches.items():
            frame = pd.DataFrame(grid_search.cv_results_)
            frame = frame.filter(regex='^(?!.*param_).*$')
            frame['estimator'] = len(frame)*[name]
            frames.append(frame)
        df = pd.concat(frames)
        
        df = df.sort_values([sort_by], ascending=False)
        df = df.reset_index()
        df = df.drop(['rank_test_score', 'index'], 1)
        
        columns = df.columns.tolist()
        columns.remove('estimator')
        columns = ['estimator']+columns
        df = df[columns]
        return df
# %%
from sklearn import datasets

breast_cancer = datasets.load_breast_cancer()
X_cancer = breast_cancer.data
y_cancer = breast_cancer.target

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

#%%
bc_d = pd.DataFrame(breast_cancer.data)
bc_d.columns = breast_cancer.feature_names
bc_t = pd.DataFrame(breast_cancer.target)
bc_t.columns = ['target']
bc = pd.concat([bc_d, bc_t], axis=1)
bc.head()


#%%
models1 = { 
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier()
}

params1 = { 
    'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
    'RandomForestClassifier': [
        { 'n_estimators': [16, 32] },
        {'criterion': ['gini', 'entropy'], 'n_estimators': [8, 16]}],
    'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
    'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] }
}

#%%
helper1 = EstimatorSelectionHelper(models1, params1)
helper1.fit(X_cancer, y_cancer, scoring='f1', n_jobs=2)
helper1.score_summary()
# %%
