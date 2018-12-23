from src.ColorectalCancer.Model import Net
import numpy as np
from src.ColorectalCancer.Dataloader import TrainDataset
from torch.optim import Adam
import torch.nn as nn
from torch import Tensor
from torch import save
from torch.utils.data import DataLoader

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters())

train_dataset = TrainDataset()
train_data_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)


def checkpoint():
	save(model.state_dict(), 'ColorectalCancer.pt')


def train(epoch):
	epoch_loss = 0
	for step, data in enumerate(train_data_loader):
		train_x, train_y = data
		train_y = train_y.view(-1)
		train_x = train_x.type('torch.FloatTensor')
		optimizer.zero_grad()
		y_pred = model.forward(train_x)
		y_true = Tensor(np.array(train_y)).type('torch.LongTensor')
		loss = criterion(y_pred, y_true)
		epoch_loss += loss.item()
		loss.backward()
		optimizer.step()
		if step == len(train_dataset) // 64:
			print('Epoch {}, loss={}'.format(epoch, epoch_loss/64))
	return epoch_loss


prev_epoch_loss = 1e20
for epoch in range(1, 50):
	epoch_loss = train(epoch)
	if epoch_loss < prev_epoch_loss:
		prev_epoch_loss = epoch_loss
		checkpoint()
