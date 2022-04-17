import itertools
from pathlib import Path
import pickle
import numpy as np

import pandas as pd
from sklearn.preprocessing import minmax_scale
from tqdm import tqdm
import config

from feature_engineering import (
	abun_per_tempbin,
	temprange,
	allcombs_df
)


# %%
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

# Assembling preprocessed and transformed training set
def process_features(dataset):
	print("Processing features...")
	splits = {
		"train": dataset.train_files,
		"val": dataset.val_files,
		"test": dataset.test_files
	}
	processed_splits = {}
	for split, split_files in splits.items():
		print("Total number of {} files: {}".format(split, len(split_files)))

		features_dict = {}

		for i, (sample_id, filepath) in enumerate(tqdm(split_files.items())):

			# Load training sample
			temp = pd.read_csv(config.DATA_PATH / filepath)

			# Preprocessing training sample
			train_sample_pp = preprocess_sample(temp)

			# Feature engineering
			sample_fe = abun_per_tempbin(train_sample_pp, allcombs_df, temprange).reset_index(drop=True)
			features_dict[sample_id] = sample_fe

		processed_splits[split] = pd.concat(
			features_dict, names=["sample_id", "dummy_index"]
		).reset_index(level="dummy_index", drop=True)

	# Make sure that all sample IDs in features and labels are identical 
	assert processed_splits['train'].index.equals(dataset.train_labels.index)

	return processed_splits

def get_processed_splits(dataset):
	processed_splits = {}

	if Path(Path.cwd() / "data" / "features_extracted_train.pkl").is_file():
		print("Features already extracted loading existing files") 
		for split in ['train', 'val', 'test']:
			filename = Path.cwd() / "data" / "features_extracted_{}.pkl".format(split)
			processed_splits[split] = pickle.load(open(filename, "rb"))
	else:
		processed_splits = process_features(dataset)
		for split, split_features in processed_splits.items():
			filename = Path.cwd() / "data" / "features_extracted_{}.pkl".format(split)
			print('Writing {} to {}'.format(split, filename))
			picklefile = open(filename, 'wb')
			pickle.dump(split_features, picklefile)
			picklefile.close()

	return processed_splits