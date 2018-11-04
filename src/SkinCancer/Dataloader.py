from torchvision import transforms
from torch.utils.data import Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from PIL import Image
import ntpath

from glob import glob

all_files = glob('/home/aftaab/MylanDatasets/Skin Cancer/*.jpg')
ground_truth = pd.read_csv('/home/aftaab/MylanDatasets/Skin Cancer/ground_truth.csv')
train, test = train_test_split(all_files)
melanoma_dict = {}

for index, row in ground_truth.iterrows():
	melanoma_dict[row['image_id']]=row['melanoma']

class TrainDataset(Dataset):

	def __init__(self):
		super().__init__()
		self.tranform = transforms.Compose([transforms.ToPILImage(),
											transforms.ToTensor(),
											transforms.Normalize(mean=[0], std=[1])])

	def __getitem__(self, index):
		file_path = train[index]
		img = Image.open(file_path).convert('RGB').resize([300, 300])
		img_t = self.tranform(np.array(img)).view(3, 300, 300)
		file_name = ntpath.basename(ntpath.splitext(train[index])[0])
		y = 0
		if melanoma_dict.get(file_name) == None:
			y = 0
		else:
			y = melanoma_dict.get(file_name)

		return [img_t, y]

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
		img = Image.open(file_path).convert('RGB').resize([300, 300])
		img_t = self.tranform(np.array(img)).view(3, 300, 300)
		file_name = ntpath.basename(ntpath.splitext(test[index])[0])
		y = 0
		if melanoma_dict.get(file_name) == None:
			y = 0
		else:
			y = melanoma_dict.get(file_name)

		return [img_t, y]

	def __len__(self):
		return len(test)