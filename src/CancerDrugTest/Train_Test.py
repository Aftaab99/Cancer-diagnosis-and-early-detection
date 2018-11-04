from keras.models import Sequential
from keras.layers import Dropout, Dense
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('preprocessed_data.csv')
data.columns = list(range(0, 80)) + ['Target']
print(data.columns)
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
Y_train = enc.fit_transform(np.array(Y_train).reshape(-1, 1))
Y_test = enc.fit_transform(np.array(Y_test).reshape(-1, 1))

# Multiclass model
model = Sequential()
model.add(Dense(80, activation='relu', input_dim=80))
model.add(Dense(96, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(9, activation='softmax'))
model.compile(optimizer='adam', metrics=['accuracy'], loss='categorical_crossentropy')
model.summary()
model.fit(X_train, Y_train, epochs=500)
scores = model.evaluate(X_test, Y_test)

# Binary model
model_bin = Sequential()
model_bin.add(Dense(80, activation='relu', input_dim=80))
model_bin.add(Dense(96, activation='relu'))
model_bin.add(Dense(64, activation='relu'))
model_bin.add(Dense(1, activation='sigmoid'))
model_bin.compile(optimizer='adam', metrics=['accuracy'], loss='binary_crossentropy')
model_bin.summary()
model_bin.fit(X_train, Y_train_bin, epochs=25)
scores_bin = model_bin.evaluate(X_test, Y_test_bin)

print('Test loss={}, Test accuracy={}'.format(scores[0], scores[1]))
print('Test loss={}, Test accuracy={}'.format(scores_bin[0], scores_bin[1]))

model.save('cancer_drug_multiclass.h5')
model_bin.save('cancer_drug_binary.h5')

# Accuracy 57.59%, 99.32%