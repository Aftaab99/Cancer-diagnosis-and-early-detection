from torch.optim import Adam
import numpy as np
from torch.autograd import Variable
from torch.utils.data import DataLoader
from BreastCancerDiagnosis.Model import Net
from BreastCancerDiagnosis.Dataloader import TrainDataset

from torch import Tensor
import torch.nn as nn
from torch import save
import time
train_dataset = TrainDataset()
train_loader = DataLoader(train_dataset, batch_size=32, num_workers=2, shuffle=True)

model = Net()
criterion = nn.BCELoss()
optim = Adam(model.parameters())

loss_history = []


def train(epoch):
	time1=time.time()
	running_loss=0
	for step, data in enumerate(train_loader, 0):
		train_x, train_y = data

		optim.zero_grad()

		y_hat = model.forward(train_x)
		y_true = Tensor(np.array(train_y)).type('torch.FloatTensor')
		loss = criterion(y_hat, y_true)
		loss.backward()
		optim.step()

		running_loss+=loss.item()
		if step % 32 == 31:  # print every 2000 mini-batches
			print('[%d, %5d] loss: %.8f' %
				  (epoch, step + 1, running_loss / 32))
			running_loss = 0.0
	time2=time.time()
	print('Epoch time:{}'.format(time2-time1))

for epoch in range(1, 11):
	train(epoch)

save(model.state_dict(), 'breast_cancer_diagnosis.pt')
