from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv('Dataset/combined_data.csv')
Y = data.iloc[:, -1]
X = data.iloc[:, 0:6117]
del data
pca_comp = PCA(n_components=80, svd_solver='full')
X = pca_comp.fit_transform(X, Y)
print('PCA variance=%.3f' % pca_comp.explained_variance_ratio_[0])
scaler = MinMaxScaler()
X = scaler.fit_transform(X)
joined_data = np.concatenate((X, np.array(Y).reshape(-1, 1)), axis=1)
data = pd.DataFrame(joined_data)
data.to_csv('preprocessed_data.csv', index=False)
pickle.dump(pca_comp, open('pca_compressor.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))
