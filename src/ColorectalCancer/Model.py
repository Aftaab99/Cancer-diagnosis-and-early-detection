import torch.nn as nn
from torch.nn.functional import sigmoid, relu


class Net(nn.Module):

	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 10, 3)
		self.pool1 = nn.MaxPool2d(3, 2)
		self.conv2 = nn.Conv2d(10, 12, 3)
		self.pool2 = nn.MaxPool2d(3, 2)
		self.fc1 = nn.Linear(35 * 35 * 12, 256)
		self.dropout1 = nn.Linear(256, 8)
		self.fc2 = nn.Linear(8, 1)

	def forward(self, x):
		x = self.conv1(x)
		x = self.pool1(x)
		x = relu(x)
		x = self.conv2(x)
		x = self.pool2(x)
		x = relu(x)
		x = x.view(-1, 35 * 35 * 12)
		x = self.fc1(x)
		x = relu(x)
		x = self.dropout1(x)
		x = self.fc2(x)
		x = sigmoid(x)
		return x
