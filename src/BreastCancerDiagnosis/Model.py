from torch import nn, optim
from torch.nn.functional import relu, sigmoid


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 64)
		self.fc2 = nn.Linear(64, 1)


	def forward(self, x):
		x = self.pool(relu(self.conv1(x)))
		x = self.pool(relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = relu(self.fc1(x))
		x = self.fc2(x)
		return sigmoid(x)


net = Net()
