from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import ntpath
from torch import Tensor
from glob import glob

all_files = glob('/home/aftaab/MylanDatasets/Skin Cancer/*.jpg')
ground_truth = pd.read_csv('/home/aftaab/MylanDatasets/Skin Cancer/ground_truth.csv')
meta_data = pd.read_csv('/home/aftaab/MylanDatasets/Skin Cancer/ISIC-2017_Training_Data_metadata.csv')

train, test = train_test_split(all_files, test_size=0.15)
melanoma_dict = {}

n_samples = sum(ground_truth['melanoma'])
n_negatives = 0
n_positives = 0

for (index, row), (index1, row1) in zip(ground_truth.iterrows(), meta_data.iterrows()):
	if n_negatives<=n_samples and row['melanoma']==0:
		melanoma_dict[row['image_id']] = {'target': row['melanoma'], 'gender': row1['sex'], 'age': row1['age_approximate']}
		n_negatives+=1
	elif n_positives<=n_samples and row['melanoma']==1:
		melanoma_dict[row['image_id']] = {'target': row['melanoma'], 'gender': row1['sex'], 'age': row1['age_approximate']}
		n_positives+=1
	else:
		all_files_temp = all_files
		for file in all_files:
			file_name = ntpath.basename(ntpath.splitext(file)[0])

			if file_name == row['image_id']:
				all_files_temp.remove(file)
				break
		all_files = all_files_temp
print('Undersampling to {} positive, {} negative samples.'.format(n_negatives, n_positives))
train, test = train_test_split(all_files, test_size=0.15)

class TrainDataset(Dataset):

	def __init__(self):
		super().__init__()

	def __getitem__(self, index):
		file_path = train[index]
		img = np.array(Image.open(file_path).convert('RGB').resize([128, 128]))
		img = Tensor(img).view(3, 128, 128)

		file_name = ntpath.basename(ntpath.splitext(train[index])[0])
		y = 0
		age = -1
		sex = -1
		if melanoma_dict.get(file_name) is not None:
			y = melanoma_dict.get(file_name)['target']
			g = melanoma_dict.get(file_name)['gender']
			if g == 'female':
				g = 0
			elif g == 'male':
				g = 1
			else:
				g = -1
			sex = g
			age = melanoma_dict.get(file_name)['age']
			if type(age) == str:
				age = -80

		return [img, y, sex, age]

	def __len__(self):
		return len(train)


class TestDataset(Dataset):

	def __init__(self):
		super().__init__()

	def __getitem__(self, index):
		file_path = test[index]
		img = np.array(Image.open(file_path).convert('RGB').resize([128, 128]))
		img = Tensor(img).view(3, 128, 128)
		file_name = ntpath.basename(ntpath.splitext(test[index])[0])
		y = 0
		age = -1
		sex = -1
		if melanoma_dict.get(file_name) is not None:
			y = melanoma_dict.get(file_name)['target']
			g = melanoma_dict.get(file_name)['gender']
			if g == 'female':
				g = 0
			elif g == 'male':
				g = 1
			else:
				g = -1
			sex = g
			age = melanoma_dict.get(file_name)['age']
			if type(age) == str:
				age = -80

		return [img, y, sex, age]

	def __len__(self):
		return len(test)
