# - one vs all approach where a binary classifier is trained for each label class independently

# Define stratified k-fold validation
import numpy as np
from sklearn.metrics import log_loss, make_scorer
import config
from sklearn.model_selection import StratifiedKFold, cross_val_score


# Check log loss score for baseline dummy model
def logloss_cross_val(clf, X, y):
	skf = StratifiedKFold(n_splits=10, random_state=config.RANDOM_SEED, shuffle=True)

	# Define log loss
	log_loss_scorer = make_scorer(log_loss, needs_proba=True)

	# Generate a score for each label class
	log_loss_cv = {}
	for col in y.columns:

		y_col = y[col]  # take one label at a time
		log_loss_cv[col] = np.mean(
			cross_val_score(clf, X.values, y_col, cv=skf, scoring=log_loss_scorer)
		)

	avg_log_loss = np.mean(list(log_loss_cv.values()))

	return log_loss_cv, avg_log_loss

