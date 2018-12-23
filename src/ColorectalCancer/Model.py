import torch.nn as nn
from torch.nn.functional import relu, softmax


class Net(nn.Module):

	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 16, 3, stride=1, padding=1)
		self.pool1 = nn.MaxPool2d(kernel_size=3)
		self.dropout1 = nn.Dropout2d(0.3)
		self.conv5 = nn.Conv2d(16, 32, 3, stride=1, padding=1)
		self.pool2 = nn.MaxPool2d(kernel_size=3)
		self.dropout2 = nn.Dropout2d(0.3)
		self.conv8 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
		self.pool3 = nn.MaxPool2d(kernel_size=3)
		self.conv11 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
		self.conv12 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
		self.pool4 = nn.MaxPool2d(kernel_size=3)
		self.dropout3 = nn.Dropout2d(0.5)
		self.dense1 = nn.Linear(256, 8)

	def forward(self, x):
		x = self.conv1(x)
		x = relu(self.pool1(x))
		x = self.dropout1(x)
		x = self.conv5(x)
		x = relu(self.pool2(x))
		x = self.dropout2(x)
		x = self.conv8(x)
		x = relu((self.pool3(x)))
		x = self.conv11(x)
		x = self.conv12(x)
		x = relu(self.pool4(x))
		x = self.dropout3(x)
		x = x.view(-1, 256)
		x = self.dense1(x)
		x = softmax(x, dim=1)
		return x
