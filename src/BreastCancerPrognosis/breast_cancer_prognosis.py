import torch.nn.functional as F
from torch.nn import Linear
from torch.nn import BCELoss, Module
from torch.optim import Adam
import pandas as pd
from torch import Tensor, save
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import f1_score
from Model import BreastCancerPrognosisModel


model = BreastCancerPrognosisModel()
criterion = BCELoss()
optimizer = Adam(model.parameters())

# Headers for the dataset
columns = ['ID', 'CLUMP_THICKNNESS', 'UNIFORMITY_OF_CELL_SIZE', 'UNIFORMITY_OF_CELL_SHAPE', 'MARGINAL_ADHESION',
		   'SINGLE_EPITHELIAL_CELL_SIZE', 'BARE_NUCLEI', 'BLAND_CHROMATIN', 'NORMAL_NUCLEI', 'MITOSIS', 'TARGET_CLASS']

raw_data = pd.read_csv('breast-cancer-wisconsin.data', header=None)
raw_data.columns = columns

# Fill missing values
raw_data = raw_data.replace('?', np.nan)
raw_data = raw_data.fillna(raw_data.median())

# Map benign to 0, malignant to 1
raw_data['TARGET_CLASS'] = raw_data['TARGET_CLASS'].map({2: 0, 4: 1})
print(raw_data.tail())
# Drop the ID column
raw_data = raw_data.loc[:, raw_data.columns != 'ID']
columns.remove('ID')

# Scale the values between 0-1
scaler = MinMaxScaler()
y = raw_data['TARGET_CLASS']
raw_data = scaler.fit_transform(raw_data.drop(['TARGET_CLASS'], axis=1))
columns.remove('TARGET_CLASS')
raw_data = pd.DataFrame(raw_data, columns=columns)
raw_data = pd.concat([raw_data, y], axis=1)
columns.append('TARGET_CLASS')
train, test = train_test_split(raw_data, test_size=0.15)
train = pd.DataFrame(data=train, columns=columns)
test = pd.DataFrame(data=test, columns=columns)

# Model training and test data
X_train = train.loc[:, train.columns != 'TARGET_CLASS']
Y_train = train['TARGET_CLASS']

X_test = test.loc[:, test.columns != 'TARGET_CLASS']
Y_test = test['TARGET_CLASS']
print('Shape={}'.format(X_train.values.shape))
n_features_tr = X_train.values.shape[1]
n_samples_tr = X_train.values.shape[0]

X_train = Tensor(X_train.values)
Y_train = Tensor(Y_train.values).reshape(-1, 1)
X_test = Tensor(X_test.values)
Y_test = Tensor(Y_test.values).reshape(-1, 1)

N_epochs = 500

for i in range(1, N_epochs + 1):
	optimizer.zero_grad()

	Y_hat = model.forward(X_train)

	loss = criterion(Y_hat, Y_train)
	loss.backward()
	optimizer.step()
	print('Loss = {}, epoch = {}'.format(loss.item(), i))

# Testing
model.eval()
print(X_test.shape)
Y_hat = np.round(model.forward(X_test).detach().numpy())
accuracy = f1_score(y_pred=Y_hat, y_true=Y_test)
print('Accuracy: {}'.format(accuracy))

# Saving model and scaler
save(model.state_dict(), 'breast_cancer_prognosis.pt')
with open('scaler.pkl', 'wb') as f:
	pickle.dump(scaler, f)
