from torch.utils.data import Dataset
import numpy as np
from torch import Tensor
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('preprocessed_data.csv')
data.columns = list(range(0, 80)) + ['Target']
train, test = train_test_split(data, test_size=0.20)
X_train = train.drop(['Target'], axis=1)
Y_train = train['Target']
Y_train_bin = []

for x in Y_train:
	if x != 9:
		Y_train_bin.append(1)
	else:
		Y_train_bin.append(0)

X_test = test.drop(['Target'], axis=1)
Y_test = test['Target']

Y_test_bin = []
for x in Y_test:
	if x != 9:
		Y_test_bin.append(1)
	else:
		Y_test_bin.append(0)

Y_train = np.array(Y_train).reshape(-1, 1)
Y_test = np.array(Y_test).reshape(-1, 1)

# Converting to torch tensors
X_train = Tensor(X_train.values)
Y_train = Tensor(Y_train).view(-1)
Y_train_bin = Tensor(Y_train_bin).reshape(-1, 1)
X_test = Tensor(X_test.values)
Y_test = Tensor(Y_test).view(-1)
Y_test_bin = Tensor(Y_test_bin).reshape(-1, 1)


class TrainDataset(Dataset):
	def __init__(self, dataset_type):
		self.dataset_type = dataset_type

	def __getitem__(self, index):
		if self.dataset_type == 'Multiclass':
			return X_train[index], Y_train[index]
		if self.dataset_type == 'Binary':
			return X_train[index], Y_train_bin[index]

	def __len__(self):
		return len(X_train)


class TestDataset(Dataset):
	def __init__(self, dataset_type):
		self.dataset_type = dataset_type

	def __getitem__(self, index):
		if self.dataset_type == 'Multiclass':
			return X_test[index], Y_test[index]
		if self.dataset_type == 'Binary':
			return X_test[index], Y_test_bin[index]

	def __len__(self):
		return len(X_test)
