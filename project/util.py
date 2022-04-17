
import pandas as pd
from tqdm import tqdm
from feature_engineering import abun_per_tempbin
from preproccess import preprocess_sample
import config
import plotly.express as px


def plot_sample(sample_id, compounds):
	sample_data = pd.read_csv(
		config.DATA_PATH / 'train_features'/ '{}.csv'.format(sample_id))

	t = abun_per_tempbin(preprocess_sample(sample_data)
	.assign(temp=lambda df_: df_.temp.astype('category'))
	.assign(time=lambda df_: df_.time.astype('category'))
	.assign(m_z=lambda df_: df_['m/z'].astype('category'))
	)

	t = (t
	.unstack()
	.reset_index()
	.rename(columns={'level_0': 'm/z', 'level_1': 'temp_bin', 0: 'abun_minsub_scaled'})
	.drop(columns=['level_2'])
	.assign(temp_bin=lambda df_: df_.temp_bin.astype(str))
	)
	t['m/z'] = t['m/z'].astype('category')

	fig = px.scatter(t, x="m/z", y="abun_minsub_scaled", color="temp_bin") 
	# fig = px.scatter_3d(t, x="m/z", y="abun_minsub_scaled", z="temp_bin", color="temp_bin") 
	fig.update_layout(height=300, title_text="Sample: {} - Compounds: {}".format(sample_id, compounds))
	fig.show()

def train_classifier(clf, processed_splits, train_labels):
	fitted_dict = {}

	# Split into binary classifier for each class
	for col in train_labels.columns:

		y_train_col = train_labels[col]  # Train on one class at a time

		fitted_dict[col] = clf.fit(processed_splits['train'].values, y_train_col)  # Train

	return fitted_dict

# Generate predictions for each class
def predict_for_sample(sample_id, fitted_model_dict, processed_splits, compounds_order):
	temp_sample_preds_dict = {}

	temp_sample = None
	for split, data in processed_splits.items():
		if sample_id in data.index:
			temp_sample = data.loc[sample_id]

	if temp_sample is None:
		assert 'Sample not found: ' | sample_id

	for compound in compounds_order:
		clf = fitted_model_dict[compound]
		temp_sample_preds_dict[compound] = clf.predict_proba(temp_sample.values.reshape(1, -1))[:, 1][0]

	return temp_sample_preds_dict

def get_compounds_order():
	submission_template_df = pd.read_csv(
		config.DATA_PATH / "submission_format.csv", index_col="sample_id"
	)
	compounds_order = submission_template_df.columns
	sample_order = submission_template_df.index

	return compounds_order, sample_order

def make_submission_file(fitted_dict, processed_splits, classifier_name, compounds_order, sample_order):

	# Dataframe to store submissions in
	print ("Generating predictions...")
	final_submission_df = pd.DataFrame(
		[
			predict_for_sample(sample_id, fitted_dict, processed_splits, compounds_order) for sample_id in tqdm(sample_order)
		],
		index=sample_order,
	)
	filename = "{}_submission.csv".format(classifier_name)
	print("output PREDICTIONS file: ", filename)
	final_submission_df.to_csv(filename)

	return final_submission_df

