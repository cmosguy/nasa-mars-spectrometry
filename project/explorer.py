#%%
from importlib_metadata import metadata
import pandas as pd
from dataset import Dataset
import config
from project.util import plot_sample

dataset = Dataset()
metadata = dataset.metadata

#%%
# Select sample IDs for five commercial samples and five testbed samples

sample_id_commercial = (
    metadata[metadata["instrument_type"] == "commercial"]
    .index
    .values[0:5]
)
sample_id_testbed = (
    metadata[metadata["instrument_type"] == "sam_testbed"]
    .index
    .values[0:5]
)
# Import sample files for EDA
sample_commercial_dict = {}
sample_testbed_dict = {}

for i in range(0, 5):
    comm_lab = sample_id_commercial[i]
    sample_commercial_dict[comm_lab] = pd.read_csv(config.DATA_PATH / dataset.train_files[comm_lab])

    test_lab = sample_id_testbed[i]
    sample_testbed_dict[test_lab] = pd.read_csv(config.DATA_PATH / dataset.train_files[test_lab])
# Selecting a testbed sample to demonstrate preprocessing steps
sample_lab = sample_id_testbed[1]
sample_df = sample_testbed_dict[sample_lab]

#%%
# basalt only samples
# S0021, S0028, S0030, S0035
plot_sample('S0021', 'basalt')
plot_sample('S0028', 'basalt')
plot_sample('S0030', 'basalt')
# plot_sample('S0035')

# basalt,oxychlorine
#S0084, S0107, S0223, S0753
plot_sample('S0084', 'basalt,oxychlorine')
plot_sample('S0107', 'basalt,oxychlorine')
plot_sample('S0223', 'basalt,oxychlorine')

# oxychlorine only
# S0002, S0009, S00024
plot_sample('S0002', 'oxychlorine')
plot_sample('S0009', 'oxychlorine')
plot_sample('S0024', 'oxychlorine')
# %%
