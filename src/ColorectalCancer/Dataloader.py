from PIL import Image
import numpy as np
from torchvision import transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from glob import glob, fnmatch

data = glob('/home/aftaab/MylanDatasets/colorectal-histology-mnist/Patches/*/*.tif')
positives = fnmatch.filter(data, '/home/aftaab/MylanDatasets/colorectal-histology-mnist/Patches/01_TUMOR/*.tif')
negatives = []
for file in data:
	if file not in positives:
		negatives.append(file)
print('Negatives=%d' % len(negatives))
print('Positives=%d' % len(positives))

train, test = train_test_split(data, test_size=0.2)

class TrainDataset(Dataset):

	def __init__(self):
		super().__init__()
		self.train = train
		self.transform = transforms.Compose([transforms.ToPILImage(),
											 transforms.Normalize([0, 0, 0], [1, 1, 1]),
											 transforms.ToTensor()])

	def __getitem__(self, index):
		img = Image.open(train[index], 'r').convert('RGB')
		img = img.resize([150, 150])
		img = np.array(img).reshape(3, 150, 150)
		y = 0
		if train[index] in positives:
			y = 1
		else:
			y = 0
		return (img, y)

	def __len__(self):
		return len(train)


class TestDataset(Dataset):

	def __init__(self):
		super().__init__()
		self.test = test
		self.transform = transforms.Compose([transforms.ToPILImage(),
											 transforms.Normalize([0, 0, 0], [1, 1, 1]),
											 transforms.ToTensor()])

	def __getitem__(self, index):
		img = Image.open(train[index], 'r').convert('RGB')
		img = img.resize([150, 150])
		img = np.array(img).reshape(3, 150, 150)
		y = 0
		if test[index] in positives:
			y = 1
		else:
			y = 0
		return (img, y)

	def __len__(self):
		return len(test)
