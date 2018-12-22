import pandas as pd
import numpy as np
import os

data = pd.read_csv('Dataset/risk_factors_cervical_cancer.csv')
data.describe()
data=data.replace('?', np.nan)
print(data.isna().sum())

cols=data.columns
featured_cols=[]
for c in list(cols):
    if data[c].isna().sum()<100:
        featured_cols.append(c)
print(featured_cols)
cols_to_be_dropped = list(data.columns)
featured_cols.remove('Smokes')
featured_cols.remove('Smokes (years)')
featured_cols.remove('Smokes (packs/year)')
for c in featured_cols:
    cols_to_be_dropped.remove(c)
data = data.drop(cols_to_be_dropped, axis=1)
data = data.drop('Dx:Cancer', axis=1)
data = data.drop('Dx', axis=1)
print(data.head())
data=data.fillna(data.median())
print(data.isna().sum())

data.to_csv('feature_dataset.csv', index=False)