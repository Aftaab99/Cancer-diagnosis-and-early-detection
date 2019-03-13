from src.SkinCancer.Model import SkinCancerModel
from src.SkinCancer.Dataloader import TrainDataset
from torch import save
import torch.nn as nn
import pandas as pd
from torch.optim import Adam
from torch.utils.data import DataLoader

model = SkinCancerModel()
optimizer = Adam(model.parameters())
criterion = nn.BCELoss()

train_dataset = TrainDataset()
ground_truth = pd.read_csv('/home/aftaab/MylanDatasets/Skin Cancer/ground_truth.csv')
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)


def train(epoch):
	for step, data in enumerate(train_loader):
		train_x, train_y, sm, am = data
		sex_t = sm.float()
		age_t = am.float()
		train_y = train_y.float().view(-1, 1)
		optimizer.zero_grad()
		y_pred = model.forward(train_x, sex_t, age_t)
		loss = criterion(y_pred, train_y)
		loss.backward()
		optimizer.step()
		print('Epoch {}, batch={}, loss={}'.format(epoch, step, loss.item()))


for i in range(1, 41):
	train(i)
	if i % 5 == 0:
		save(model.state_dict(), 'model_skin_cancer_epoch{}.pt'.format(i))
