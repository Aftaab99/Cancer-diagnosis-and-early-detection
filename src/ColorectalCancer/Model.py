import torch.nn as nn
from torch.nn.functional import relu

class Net(nn.Module):

	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(1, 8, 3, stride=1, padding=1)
		self.conv2 = nn.Conv2d(8, 8, 3, stride=1, padding=1)
		self.pool1 = nn.MaxPool2d(kernel_size=3)
		self.conv3 = nn.Conv2d(8, 16, 3, stride=1, padding=1)
		self.conv4 = nn.Conv2d(16, 16, 3, stride=1, padding=1)
		self.pool2 = nn.MaxPool2d(3)
		self.conv5 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
		self.pool3 = nn.MaxPool2d(3)
		self.dropout1 = nn.Dropout2d(0.3)
		self.dense1 = nn.Linear(2*2*32, 8)

	def forward(self, x):
		x = self.conv1(x)
		x = self.conv2(x)
		x = relu(self.pool1(x))
		x = self.conv3(x)
		x = self.conv4(x)
		x = relu(self.pool2(x))
		x = self.conv5(x)
		x = relu(self.pool3(x))
		x = self.dropout1(x)
		x = x.view(-1, 2*2*32)
		x = self.dense1(x)
		return x
