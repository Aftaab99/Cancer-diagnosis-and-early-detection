import torch.nn.functional as F
from torch.nn import Module, Linear, Dropout
import torch


class MultiClassNet(Module):

	def __init__(self):
		super().__init__()
		self.layer1 = Linear(80, 96)
		self.dropout1 = Dropout(0.4)
		self.layer2 = Linear(96, 64)
		self.dropout2 = Dropout(0.3)
		self.layer3 = Linear(64, 10)

	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.dropout1(x))
		x = F.relu(self.layer2(x))
		x = F.relu(self.dropout2(x))
		x = F.relu(self.layer3(x))
		return F.softmax(x, dim=1)


class BinaryNet(Module):
	def __init__(self):
		super().__init__()
		self.layer1 = Linear(80, 96)
		self.dropout1 = Dropout(0.4)
		self.layer2 = Linear(96, 64)
		self.dropout2 = Dropout(0.2)
		self.layer3 = Linear(64, 1)

	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.dropout1(x))
		x = F.relu(self.dropout2(x))
		x = F.relu(self.layer2(x))
		x = F.relu(self.dropout2(x))
		x = F.relu(self.layer3(x))
		return torch.sigmoid(x)
