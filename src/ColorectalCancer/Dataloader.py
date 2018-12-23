from PIL import Image
from torch import Tensor
import numpy as np
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from glob import glob, fnmatch

data = glob('/home/aftaab/MylanDatasets/colorectal-histology-mnist/Patches/*/*.tif')
class1 = fnmatch.filter(data, '/home/aftaab/MylanDatasets/colorectal-histology-mnist/Patches/01_TUMOR/*.tif')
class2 = fnmatch.filter(data, '/home/aftaab/MylanDatasets/colorectal-histology-mnist/Patches/02_STROMA/*.tif')
class3 = fnmatch.filter(data, '/home/aftaab/MylanDatasets/colorectal-histology-mnist/Patches/03_COMPLEX/*.tif')
class4 = fnmatch.filter(data, '/home/aftaab/MylanDatasets/colorectal-histology-mnist/Patches/04_LYMPHO/*.tif')
class5 = fnmatch.filter(data, '/home/aftaab/MylanDatasets/colorectal-histology-mnist/Patches/05_DEBRIS/*.tif')
class6 = fnmatch.filter(data, '/home/aftaab/MylanDatasets/colorectal-histology-mnist/Patches/06_MUCOSA/*.tif')
class7 = fnmatch.filter(data, '/home/aftaab/MylanDatasets/colorectal-histology-mnist/Patches/07_ADIPOSE/*.tif')
class0 = fnmatch.filter(data, '/home/aftaab/MylanDatasets/colorectal-histology-mnist/Patches/08_EMPTY/*.tif')
train, test = train_test_split(data, test_size=0.05, shuffle=True)


class TrainDataset(Dataset):

	def __init__(self):
		super().__init__()
		self.train = train

	def __getitem__(self, index):
		img = Image.open(train[index]).convert('RGB')
		img = img.resize([150, 150])
		img = np.array(img).reshape(3, 150, 150)
		if train[index] in class1:
			y = 1
		elif train[index] in class2:
			y = 2
		elif train[index] in class3:
			y = 3
		elif train[index] in class4:
			y = 4
		elif train[index] in class5:
			y = 5
		elif train[index] in class6:
			y = 6
		elif train[index] in class7:
			y = 7
		else:
			y = 0

		img = Tensor(img/255.0).view(3, 150, 150)
		return img, y

	def __len__(self):
		return len(train)


class TestDataset(Dataset):

	def __init__(self):
		super().__init__()
		self.test = test

	def __getitem__(self, index):
		img = Image.open(train[index]).convert('RGB')
		img = img.resize([150, 150])
		img = np.array(img).reshape(3, 150, 150)
		if test[index] in class1:
			y = 1
		elif test[index] in class2:
			y = 2
		elif test[index] in class3:
			y = 3
		elif test[index] in class4:
			y = 4
		elif test[index] in class5:
			y = 5
		elif test[index] in class6:
			y = 6
		elif test[index] in class7:
			y = 7
		else:
			y = 0

		img = Tensor(img/255.0).view(3, 150, 150)
		return img, y

	def __len__(self):
		return len(test)
