import config
import pandas as pd


class Dataset():
	def __init__(self) -> None:
		self.metadata = pd.read_csv(config.DATA_PATH / "metadata.csv", index_col="sample_id")

		self.train_files = self.metadata[self.metadata["split"] == "train"]["features_path"].to_dict()
		self.val_files = self.metadata[self.metadata["split"] == "val"]["features_path"].to_dict()
		self.test_files = self.metadata[self.metadata["split"] == "test"]["features_path"].to_dict()

		print("Number of training samples: ", len(self.train_files))
		print("Number of validation samples: ", len(self.val_files))
		print("Number of testing samples: ", len(self.test_files))
		# %%
		self.train_labels = pd.read_csv(config.DATA_PATH / "train_labels.csv", index_col="sample_id")
		self.val_labels = pd.read_csv(config.DATA_PATH / "val_labels.csv", index_col="sample_id")

		self.all_test_files = self.val_files.copy()
		self.all_test_files.update(self.test_files)
		print("Total test files: ", len(self.all_test_files))

def test():
	dataset = Dataset()
	print(dataset.metadata.head())
	print(dataset.train_labels.head())

if __name__ == "__main__":
    test()