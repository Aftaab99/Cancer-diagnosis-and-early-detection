from SkinCancer.Model import Net
from SkinCancer.Dataloader import TrainDataset
from torch import save
from torch.tensor import Tensor
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader

model = Net()
optimizer = Adam(model.parameters())
criterion = nn.BCELoss()

train_dataset = TrainDataset()
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)


def train(epoch):
	for step, data in enumerate(train_loader):
		train_x, train_y = data
		train_y = train_y.type('torch.FloatTensor')
		optimizer.zero_grad()
		y_pred = model.forward(train_x)
		loss = criterion(y_pred, train_y)
		loss.backward()
		optimizer.step()
		if step == len(train_dataset) // 32 - 1:
			print('Epoch {}, loss={}'.format(epoch, loss.item()))


for i in range(1, 11):
	train(i)

save(model.state_dict(), 'model_skin_cancer.pt')
