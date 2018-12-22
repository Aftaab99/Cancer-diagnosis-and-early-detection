import torch.nn.functional as F
from torch.nn import Module, Linear, Dropout, BCELoss, CrossEntropyLoss
import numpy as np
import torch
from sklearn.metrics import f1_score, accuracy_score
from torch.optim import Adam
from torch import Tensor, save
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split


data = pd.read_csv('preprocessed_data.csv')
data.columns = list(range(0, 80)) + ['Target']
train, test = train_test_split(data, test_size=0.25)
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

enc = OneHotEncoder()
Y_train = np.array(Y_train).reshape(-1, 1)
Y_test = np.array(Y_test).reshape(-1, 1)

# Converting to torch tensors
X_train = Tensor(X_train.values)
Y_train = Tensor(Y_train).view(-1)
Y_train_bin = Tensor(Y_train_bin).reshape(-1,1)
X_test = Tensor(X_test.values)
Y_test = Tensor(Y_test).view(-1)
Y_test_bin = Tensor(Y_test_bin).reshape(-1,1)

print(Y_train.size())

class MultiClassNet(Module):
	def __init__(self):
		super().__init__()
		self.layer1 = Linear(80, 96)
		self.dropout1 = Dropout(0.4)
		self.layer2 = Linear(96, 64)
		self.dropout2 = Dropout(0.2)
		self.layer3 = Linear(64, 10)

	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.dropout1(x))
		x = F.relu(self.layer2(x))
		x = F.relu(self.dropout2(x))
		x = F.relu(self.layer3(x))
		return F.softmax(x)

model_multi_class = MultiClassNet()
criterion_multi = CrossEntropyLoss()
optim_multi = Adam(model_multi_class.parameters())

class BinaryNet(Module):
	def __init__(self):
		super().__init__()
		self.layer1 = Linear(80, 96)
		self.dropout1 = Dropout(0.4)
		self.layer2 = Linear(96, 64)
		self.dropout2 = Dropout(0.2)
		self.layer3 = Linear(64, 1)

	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.dropout1(x))
		x = F.relu(self.dropout2(x))
		x = F.relu(self.layer2(x))
		x = F.relu(self.dropout2(x))
		x = F.relu(self.layer3(x))
		return torch.sigmoid(x)

model_binary = BinaryNet()
criterion_bin = BCELoss()
optim_bin = Adam(model_binary.parameters())

# Multiclass model training
N_epochs = 10000
for i in range(1, N_epochs+1):
	optim_multi.zero_grad()
	Y_hat = model_multi_class.forward(X_train)
	loss = criterion_multi(Y_hat, Y_train.long())
	print('Epoch = {}, loss = {}'.format(i, loss.item()))
	loss.backward()
	optim_multi.step()

save(model_multi_class.state_dict(), 'drug_test_multi.pt')

# Binary model training
N_epochs = 250
for i in range(1, N_epochs+1):
	optim_bin.zero_grad()
	Y_hat = model_binary.forward(X_train)
	loss = criterion_bin(Y_hat, Y_train_bin)
	print('Epoch = {}, loss = {}'.format(i, loss.item()))
	loss.backward()
	optim_bin.step()

save(model_binary.state_dict(), 'drug_test_binary.pt')


# Testing
accuracy_multi = accuracy_score(y_true=Y_test, y_pred=np.argmax(model_multi_class.forward(X_test).detach().numpy(), axis=1))
accuracy_bin = f1_score(y_true=Y_test_bin, y_pred=np.round(model_binary.forward(X_test).detach().numpy()))
print('Accuracy multiclass: {}'.format(accuracy_multi))
print('Accuracy binary: {}'.format(accuracy_bin))