from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
from glob import glob, fnmatch
from sklearn.model_selection import train_test_split
from PIL import Image
data_path = '/home/aftaab/MylanDatasets/breast-histopathology-images/IDC_histopathology_data/**/*.png'
all_files = glob(data_path, recursive=True)

positives = fnmatch.filter(all_files, '*class1.png')
negatives = fnmatch.filter(all_files, '*class0.png')
print("No of files total={}".format(len(all_files)))
print("No of positives={}".format(len(positives)))
print("No of negatives={}".format(len(negatives)))

train_files, test_files = train_test_split(all_files, test_size=0.05)

class TrainDataset(Dataset):
	def __init__(self):
		self.train_files = train_files
		self.transform=transforms.Compose([transforms.ToPILImage(),
										   transforms.ToTensor(),
										   transforms.Normalize(mean=[0], std=[1])])

	def __getitem__(self, index):
		file_name = self.train_files[index]
		img = np.array(Image.open(file_name).convert('RGB').resize([32, 32]))
		y=0
		if file_name in positives:
			y=1
		if file_name in negatives:
			y=0
		img=self.transform(img).view(3, 32, 32)
		return (img, y)

	def __len__(self):
		return len(train_files)


class TestDataset(Dataset):
	def __init__(self):
		self.test_files = test_files
		self.transform = transforms.Compose([transforms.ToPILImage(),
											 transforms.ToTensor(),
											 transforms.Normalize(mean=[0], std=[1])])

	def __getitem__(self, index):
		file_name = self.test_files[index]
		img = np.array(Image.open(file_name).convert('RGB').resize([32,32]))
		y = 0
		if file_name in positives:
			y = 1
		if file_name in negatives:
			y = 0
		img = self.transform(img).view(3, 32, 32)
		return (img, y)

	def __len__(self):
		return len(test_files)
