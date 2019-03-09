from torch import nn
from torch.nn.functional import relu
import torch

class BreastCancerModel(nn.Module):
	def __init__(self):
		super(BreastCancerModel, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, kernel_size=(5, 5))
		self.pool1 = nn.MaxPool2d(kernel_size=(2, 2))
		self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
		self.pool2 = nn.AdaptiveAvgPool2d(output_size=(5, 5))
		self.fc1 = nn.Linear(16 * 5 * 5, 64)
		self.fc2 = nn.Linear(64, 1)

	def forward(self, x):
		x = self.pool1(relu(self.conv1(x)))
		x = self.pool2(relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = relu(self.fc1(x))
		x = self.fc2(x)
		return torch.sigmoid(x)
