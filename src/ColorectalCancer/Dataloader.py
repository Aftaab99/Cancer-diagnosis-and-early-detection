from sklearn.model_selection import train_test_split
import pandas as pd
from src.ColorectalCancer.Model import Net
from torch.optim import Adam
import torch.nn as nn
from torch.utils.data import Dataset


data = pd.read_csv('/home/aftaab/MylanDatasets/colorectal-histology-mnist/hmnist_64_64_L.csv')
train, test = train_test_split(data, test_size=0.1, shuffle=True)

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

train_x = train.drop(['label'], axis=1).values.reshape(-1, 1, 64, 64)
train_y = train['label'].values.reshape(-1, 1)

test_x = test.drop(['label'], axis=1).values.reshape(-1, 1, 64, 64)
test_y = test['label'].values.reshape(-1, 1)


class TrainDataset(Dataset):

	def __getitem__(self, index):
		return train_x[index]/255.0, train_y[index]-1

	def __len__(self):
		return test_x.shape[0]


class TestDataset(Dataset):

	def __getitem__(self, index):
		return test_x[index]/255.0, test_y[index]-1

	def __len__(self):
		return test_x.shape[0]