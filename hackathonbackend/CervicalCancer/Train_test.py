import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import pickle

# Read dataset
proc_data = pd.read_csv('feature_dataset.csv')
columns = proc_data.columns
# Scale features betweeen 0-1
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(proc_data)

# Split the data into train and test sets.
train, test = train_test_split(scaled_data)

# Split train and test into features and targets
train = pd.DataFrame(data=train, columns=columns)
X_train = train.drop(['Hinselmann', 'Schiller', 'Citology', 'Biopsy'], axis=1)
Y_train_hinselmann = train['Hinselmann']
Y_train_schiller = train['Schiller']
Y_train_citology = train['Citology']
Y_train_biopsy = train['Biopsy']

test = pd.DataFrame(data=test, columns=columns)
X_test = test.drop(['Hinselmann', 'Schiller', 'Citology', 'Biopsy'], axis=1)
Y_test_hinselmann = test['Hinselmann']
Y_test_schiller = test['Schiller']
Y_test_citology = test['Citology']
Y_test_biopsy = test['Biopsy']

model_hinselmann = SVC(kernel='linear', class_weight={0: 1, 1: 12})
model_schiller = SVC(kernel='linear', class_weight={0: 1, 1: 10})
model_citology = SVC(kernel='linear', class_weight={0: 1, 1: 3})
model_biopsy = SVC(kernel='linear', class_weight={0: 1, 1: 12})

model_hinselmann.fit(X_train, Y_train_hinselmann)
model_schiller.fit(X_train, Y_train_schiller)
model_citology.fit(X_train, Y_train_citology)
model_biopsy.fit(X_train, Y_train_biopsy)

Y_pred_hinselmann = model_hinselmann.predict(X_test)
Y_pred_schiller = model_schiller.predict(X_test)
Y_pred_citology = model_citology.predict(X_test)
Y_pred_biopsy = model_biopsy.predict(X_test)

# Measure accuracy
print('Accuracy Hinselmann={}, f1={}'.format(accuracy_score(Y_test_hinselmann, Y_pred_hinselmann)
											 , f1_score(Y_test_hinselmann, Y_pred_hinselmann)))
print('Accuracy Schiller={}, f1={}'.format(accuracy_score(Y_test_schiller, Y_pred_schiller)
										   , f1_score(Y_test_schiller, Y_pred_schiller)))
print('Accuracy Citoloy={}, f1={}'.format(accuracy_score(Y_test_citology, Y_pred_citology)
										  , f1_score(Y_test_citology, Y_pred_citology)))
print('Accuracy Biopsy={}, f1={}'.format(accuracy_score(Y_test_biopsy, Y_pred_biopsy)
										 , f1_score(Y_test_biopsy, Y_pred_biopsy)))

f_hinselmann = open('model_hins.pkl', 'wb')
pickle.dump(model_hinselmann, f_hinselmann)
f_schiller = open('model_sch.pkl', 'wb')
pickle.dump(model_schiller, f_schiller)
f_citology = open('model_cit.pkl', 'wb')
pickle.dump(model_citology, f_citology)
f_biopsy = open('model_bio.pkl', 'wb')
pickle.dump(model_biopsy, f_biopsy)
# Accuracy 87-94% for each model