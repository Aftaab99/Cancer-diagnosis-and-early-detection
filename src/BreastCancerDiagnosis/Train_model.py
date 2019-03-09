from torch.optim import Adam
import numpy as np
from torch.utils.data import DataLoader
from src.BreastCancerDiagnosis.Model import BreastCancerModel
from src.BreastCancerDiagnosis.Dataloader import TrainDataset
from torch import Tensor
import torch.nn as nn
from torch import save
import time

train_dataset = TrainDataset()
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=2, shuffle=True)

model = BreastCancerModel()
criterion = nn.BCELoss()
optim = Adam(model.parameters())

loss_history = []


def train(epoch):
	time1 = time.time()
	running_loss = 0
	for step, data in enumerate(train_loader, 0):
		train_x, train_y = data
		train_y=train_y.view(-1, 1)
		optim.zero_grad()

		y_hat = model.forward(train_x)
		y_true = Tensor(np.array(train_y)).type('torch.FloatTensor')
		loss = criterion(y_hat, y_true)
		loss.backward()
		optim.step()

		running_loss += loss.item()
		if step % 50 == 0 and step != 0:
			print('Epoch %d, batch=%d loss: %.8f' %
				  (epoch, step, running_loss / 50))
			running_loss = 0.0
	time2 = time.time()
	print('Epoch time:{}'.format(time2 - time1))


for epoch in range(1, 11):
	train(epoch)

save(model.state_dict(), 'breast_cancer_diagnosis.pt')
