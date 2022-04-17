#%%
import pandas as pd
from sklearn.manifold import TSNE
from pyDeepInsight import ImageTransformer
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

import torch
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import accuracy_score

import warnings; 
warnings.simplefilter('ignore')

import config
from dataset import Dataset
from preproccess import (
	get_processed_splits
)
#%%
dataset = Dataset()

processed_splits = get_processed_splits(dataset)

#%%
def plot_embed_2D(X, title=None):
    sns.set(style="darkgrid")

    # Create subplots
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), squeeze=False)
    ax[0, 0].scatter(X[:, 0],
                     X[:, 1],
                     cmap=plt.cm.get_cmap("jet", 10),
                     marker="x",
                     alpha=1.0)
    plt.gca().set_aspect('equal', adjustable='box')

    if title is not None:
        ax[0, 0].set_title(title, fontsize=20)

    plt.rcParams.update({'font.size': 14})
    plt.show()


def tsne_transform(data, perplexity=30, plot=True):
    # Transpose to get (n_features, n_samples)
    data = data.T

    tsne = TSNE(n_components=2,
                metric='cosine',
                perplexity=perplexity,
                n_iter=1000,
                method='exact',
                random_state=config.RANDOM_SEED,
                n_jobs=-1)
    # Transpose to get (n_features, n_samples)
    transformed = tsne.fit_transform(data)

    if plot:
        plot_embed_2D(
            transformed,
            f"All Feature Location Matrix of Training Set (Perplexity: {perplexity})"
        )
    return transformed

#%%
# train_all_tsne = tsne_transform(processed_splits['train'], perplexity=5)
# train_all_tsne = tsne_transform(processed_splits['train'], perplexity=15)
# train_all_tsne = tsne_transform(processed_splits['train'], perplexity=30)
# train_all_tsne = tsne_transform(processed_splits['train'], perplexity=50)
# %%
tsne = TSNE(
    n_components=2,
    random_state=config.RANDOM_SEED,
	perplexity=5,
    n_jobs=-1)

# %%
# Plot image matrix with feature counts per pixel
def plot_feature_density(it, pixels=100, show_grid=True, title=None):
    # Update image size
    it.pixels = pixels

    fdm = it.feature_density_matrix()
    fdm[fdm == 0] = np.nan

    # Create subplots
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), squeeze=False)

    if show_grid:
        sns.heatmap(fdm,
                    cmap="viridis",
                    linewidths=0.01,
                    linecolor="lightgrey",
                    square=False,
                    ax=ax[0, 0])
        for _, spine in ax[0, 0].spines.items():
            spine.set_visible(True)
    else:
        sns.heatmap(fdm,
                    cmap="viridis",
                    linewidths=0,
                    square=False,
                    ax=ax[0, 0])

    if title is not None:
        ax[0, 0].set_title(title, fontsize=20)

    plt.rcParams.update({'font.size': 14})
    plt.show()

    # Feature Overlapping Counts
    dim_overlap = (
        pd.DataFrame(it._coords.T).assign(count=1).groupby(
            [0, 1],  # (x1, y1)
            as_index=False).count())
    print(dim_overlap["count"].describe())
    print(dim_overlap["count"].hist())
    plt.suptitle("Feauture Overlap Counts")


#%%
def plot_embed_2D(X, title=None):
    sns.set(style="darkgrid")

    # Create subplots
    fig, ax = plt.subplots(1, 1, figsize=(10, 7), squeeze=False)
    ax[0, 0].scatter(X[:, 0],
                     X[:, 1],
                     cmap=plt.cm.get_cmap("jet", 10),
                     marker="x",
                     alpha=1.0)
    plt.gca().set_aspect('equal', adjustable='box')

    if title is not None:
        ax[0, 0].set_title(title, fontsize=20)

    plt.rcParams.update({'font.size': 14})
    plt.show()


def tsne_transform(data, perplexity=30, plot=True, type='train'):
    # Transpose to get (n_features, n_samples)
    data = data.T

    tsne = TSNE(n_components=2,
                metric='cosine',
                perplexity=perplexity,
                n_iter=1000,
                method='exact',
                random_state=config.RANDOM_SEED,
                n_jobs=-1)
    # Transpose to get (n_features, n_samples)
    transformed = tsne.fit_transform(data)

    if plot:
        plot_embed_2D(
            transformed,
            f"All Feature Location Matrix of {type} Set (Perplexity: {perplexity})"
        )
    return transformed

#%%
train_tsne = tsne_transform(processed_splits['train'], perplexity=5, type='train')
test_tsne = tsne_transform(processed_splits['test'], perplexity=5, type='test')
val_tsne = tsne_transform(processed_splits['val'], perplexity=5, type='val')

#%%
perplexity = 5

tsne = TSNE(n_components=2,
			metric='cosine',
			perplexity=perplexity,
			n_iter=1000,
			method='exact',
			random_state=config.RANDOM_SEED,
			n_jobs=-1)

#%%
pixels = 50
it = ImageTransformer( feature_extractor=tsne, pixels=pixels)

# plot_feature_density(
#     it,
#     pixels=pixels,
#     title=
#     f"All Feature Density Matrix of Training Set (Resolution: {pixels}x{pixels})"
# )


X_train_img = it.fit_transform(processed_splits['train'])
X_val_img = it.transform(processed_splits['val'])
X_test_img = it.transform(processed_splits['test'])

fig, ax = plt.subplots(1, 3, figsize=(15, 5))
for i in range(0,3):
    ax[i].imshow(X_train_img[i])
    ax[i].title.set_text("Train[{}] - class '{}'".format(i, dataset.train_labels['basalt'][i]))
plt.tight_layout()

#%%squeeze net
compounds = dataset.train_labels.columns

preprocess = transforms.Compose([
	transforms.ToTensor()
])

nets = {}
X_train_tensor = torch.stack([preprocess(img) for img in X_train_img])
X_val_tensor = torch.stack([preprocess(img) for img in X_val_img])
X_test_tensor = torch.stack([preprocess(img) for img in X_test_img])

train_predictions = {}
val_predictions = {}
test_predictions = {}


model = torch.hub.load(
	'pytorch/vision:v0.6.0', 'squeezenet1_1', 
	pretrained=False, verbose=False).double()

model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1,1), stride=(1,1)).double()

#%%
list(model.children())[-3:]

#%%

y_train_tensor = torch.from_numpy(dataset.train_labels.values)
y_val_tensor = torch.from_numpy(dataset.val_labels.values)

batch_size = 1

trainset = TensorDataset(X_train_tensor, y_train_tensor)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-4, momentum=0.9)

for epoch in range(20):

	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = model(inputs)
		loss = criterion(outputs, labels.float())
		loss.backward()
		optimizer.step()

		running_loss += loss.item()
	# print epoch statistics
	print('[%d] loss: %.3f' %
		(epoch + 1, running_loss / len(X_train_tensor) * batch_size))

train_outputs = model(X_train_tensor)
_, train_prediction = torch.max(train_outputs, 1)

val_outputs = model(X_val_tensor)
_, val_prediction = torch.max(val_outputs, 1)

test_outputs = model(X_test_tensor)
_, test_prediction = torch.max(test_outputs, 1)

print("The train accuracy was {:.3f}".format(accuracy_score(train_prediction, y_train_tensor)))
print("The val accuracy was {:.3f}".format(accuracy_score(val_predictions, y_val_tensor)))
# %%
print(compounds)
compound = 'basalt'
accuracy_score(train_predictions[compound], torch.from_numpy(dataset.train_labels[compound].values))

#%%
for compound in compounds:
	print(torch.max(nets[compound](X_train_tensor), 1)[1].sum())

#%%
y_train_tensor[0:20]
#%%
train_predictions[compound][0:20]
#%%
dataset.train_labels[compound][0:10]


#%%
from tensorflow import keras
from tensorflow.keras import layers
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])