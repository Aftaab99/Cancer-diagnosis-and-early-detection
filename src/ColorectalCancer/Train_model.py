from src.ColorectalCancer.Model import Net
import numpy as np
from torch.autograd import Variable
from src.ColorectalCancer.Dataloader import TrainDataset
from torch.optim import Adam
import torch.nn as nn
from torch import Tensor
from torch import save
from torch.utils.data import DataLoader

model = Net()
criterion = nn.BCELoss()
optimizer = Adam(model.parameters())

train_dataset = TrainDataset()
train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


def train(epoch):
	for step, data in enumerate(train_data_loader):
		train_x, train_y = data
		train_x = train_x.type('torch.FloatTensor')
		optimizer.zero_grad()
		y_pred = model.forward(train_x)
		y_true = Tensor(np.array(train_y)).type('torch.FloatTensor')
		loss = criterion(y_pred, y_true)
		loss.backward()
		optimizer.step()
		if step == len(train_dataset)//64:
			print('Epoch {}, loss={}'.format(epoch, loss.item()))

for epoch in range(1, 26):
	train(epoch)

save(model.state_dict(), 'ColorectalCancer.pt')