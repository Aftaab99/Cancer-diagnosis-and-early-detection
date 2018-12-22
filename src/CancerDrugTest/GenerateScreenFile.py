import pandas as pd

data = pd.read_csv('/home/aftaab/MylanDatasets/inhibitors/train_test/cdk2_train.csv')
first = data.iloc[0]
print(first)
first.to_csv('generated_screening_file.csv', index=False)
