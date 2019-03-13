from torch.nn import Module, Linear
import torch.nn.functional as f


class BreastCancerPrognosisModel(Module):

	def __init__(self):
		super().__init__()
		self.layer_1 = Linear(9, 12)
		self.layer_2 = Linear(12, 4)
		self.layer_3 = Linear(4, 1)

	def forward(self, x):
		x = f.relu(self.layer_1(x))
		x = f.relu(self.layer_2(x))
		x = self.layer_3(x)
		return f.sigmoid(x)
