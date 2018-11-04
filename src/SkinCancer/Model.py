import torch.nn as nn
from torch.nn.functional import relu, sigmoid


class Net(nn.Module):

	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 10, 3, stride=2)
		self.pool1 = nn.MaxPool2d(3, stride=2)
		self.conv2 = nn.Conv2d(10, 12, 5, stride=2)
		self.pool2 = nn.AvgPool2d(3, stride=2)
		self.fc1 = nn.Linear(12*17*17, 24)
		self.dropout = nn.Dropout(0.4)
		self.fc2 = nn.Linear(24, 1)


	def forward(self, x):
		x = self.conv1(x)
		x = self.pool1(x)
		x = relu(x)
		x = self.conv2(x)
		x = self.pool2(x)
		x = relu(x)
		x = x.view(-1, 12*17*17)
		x = self.fc1(x)
		x = self.dropout(x)
		x = self.fc2(x)
		return sigmoid(x)