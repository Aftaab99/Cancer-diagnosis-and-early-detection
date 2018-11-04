import pandas as pd
import numpy as np
import keras.backend as K
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
import pickle

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
raw_data = scaler.fit_transform(raw_data)

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

# Some reshaping
X_train = X_train.values
Y_train = Y_train.values


def f1(y_true, y_pred):
	def recall(y_true, y_pred):
		"""Recall metric.

		Only computes a batch-wise average of recall.

		Computes the recall, a metric for multi-label classification of
		how many relevant items are selected.
		"""
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
		recall = true_positives / (possible_positives + K.epsilon())
		return recall

	def precision(y_true, y_pred):
		"""Precision metric.

		Only computes a batch-wise average of precision.

		Computes the precision, a metric for multi-label classification of
		how many selected items are relevant.
		"""
		true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
		predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
		precision = true_positives / (predicted_positives + K.epsilon())
		return precision

	precision = precision(y_true, y_pred)
	recall = recall(y_true, y_pred)
	return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# Create a model, train, evaluate and save model
# Sample model
model = Sequential()
model.add(Dense(10, activation='relu', input_dim=n_features_tr))
model.add(Dropout(0.5))  # For Regularization
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1])
model.summary()

model.fit(X_train, Y_train, epochs=30)
scores = model.evaluate(X_test, Y_test)
print('Test loss={}'.format(scores[0]))
print('Test f1 score={}'.format(scores[1]))

model = Sequential()
model.add(Dense(10, activation='relu', input_dim=n_features_tr))
model.add(Dropout(0.5))  # For Regularization
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[f1])
model.summary()
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, Y_train, epochs=30)
scores = model.evaluate(X_test, Y_test)
print('Test loss={}'.format(scores[0]))
print('Test accuracy score={}'.format(scores[1]))
model.save('breast_cancer_prognosis.h5')
pickle.dump(scaler, open('scaler.pkl', 'wb'), pickle.HIGHEST_PROTOCOL)
# F1 score 0.971. Accuracy not calculated as data is skewed and accuracy won't be a reliable measure
