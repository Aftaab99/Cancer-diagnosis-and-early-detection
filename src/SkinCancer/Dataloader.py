from torchvision import transforms
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

train, test = train_test_split(all_files)
melanoma_dict = {}

for (index, row), (index1, row1) in zip(ground_truth.iterrows(), meta_data.iterrows()):
	melanoma_dict[row['image_id']] = {'target': row['melanoma'], 'gender': row1['sex'], 'age': row1['age_approximate']}


class TrainDataset(Dataset):

	def __init__(self):
		super().__init__()
		self.tranform = transforms.Compose([transforms.ToPILImage(),
											transforms.ToTensor(),
											transforms.Normalize(mean=[0], std=[1])])

	def __getitem__(self, index):
		file_path = train[index]
		img = Image.open(file_path).convert('L').resize([128, 128])
		img_t = self.tranform(Tensor(np.array(img)).view(1, 128, 128))
		file_name = ntpath.basename(ntpath.splitext(train[index])[0])
		y = 0
		age = -1
		sex = -1
		if melanoma_dict.get(file_name) != None:
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

		return [img_t, y, sex, age]

	def __len__(self):
		return len(train)


class TestDataset(Dataset):

	def __init__(self):
		super().__init__()
		self.tranform = transforms.Compose([transforms.ToPILImage(),
											transforms.ToTensor(),
											transforms.Normalize(mean=[0], std=[1])])

	def __getitem__(self, index):
		file_path = test[index]
		img = Image.open(file_path).convert('L').resize([128, 128])
		img_t = self.tranform(Tensor(np.array(img)).view(1, 128, 128))
		file_name = ntpath.basename(ntpath.splitext(test[index])[0])
		y = 0;
		age = -1;
		sex = -1
		if melanoma_dict.get(file_name) != None:
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

		return [img_t, y, sex, age]

	def __len__(self):
		return len(test)
