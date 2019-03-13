import torch.nn as nn
from torch.nn.functional import relu, sigmoid, leaky_relu
import numpy as np
from torch import Tensor
import torch


class SkinCancerModel(nn.Module):

	def __init__(self):
		super().__init__()
		self.conv1 = nn.Conv2d(3, 3, 3, stride=2)
		self.pool1 = nn.MaxPool2d(3, stride=2)
		self.conv2 = nn.Conv2d(3, 4, 5, stride=2)
		self.pool2 = nn.MaxPool2d(3, stride=2)
		self.fc1 = nn.Linear(4 * 6 * 6, 24)
		self.dropout = nn.Dropout(0.4)
		self.fc2 = nn.Linear(24, 1)
		self.meta_fc3 = nn.Linear(3, 2)
		self.out = nn.Linear(2, 1)

	def forward(self, x, g, a):
		x = self.conv1(x)
		x = self.pool1(x)
		x = relu(x)
		x = self.conv2(x)
		x = self.pool2(x)
		x = leaky_relu(x)
		x = x.view(-1, 4 * 6 * 6)
		x = self.fc1(x)
		x = self.dropout(x)
		x = leaky_relu(self.fc2(x))
		a = a.view(-1, 1)
		g = g.view(-1, 1)
		x = torch.cat((x, g, a), 1)
		x = self.meta_fc3(x)
		x = leaky_relu(x)
		x = self.out(x)
		return sigmoid(x)
