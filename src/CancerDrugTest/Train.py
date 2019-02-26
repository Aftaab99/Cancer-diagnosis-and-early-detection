from torch.nn import BCELoss, CrossEntropyLoss
from torch.optim import Adam
from torch import save
from src.CancerDrugTest.Dataloader import TrainDataset
from torch.utils.data import DataLoader
from src.CancerDrugTest.Model import MultiClassNet, BinaryNet

model_multi_class = MultiClassNet()
criterion_multi = CrossEntropyLoss()
optim_multi = Adam(model_multi_class.parameters())

model_binary = BinaryNet()
criterion_bin = BCELoss()
optim_bin = Adam(model_binary.parameters())

# Multiclass model training
train_dataset = TrainDataset('Multiclass')
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=True)

N_epochs = 10000
for batch_idx, data in enumerate(train_loader):
	X_train, Y_train = data
	for i in range(1, N_epochs + 1):
		optim_multi.zero_grad()
		Y_hat = model_multi_class.forward(X_train)
		loss = criterion_multi(Y_hat, Y_train.long())
		print('Epoch = {}, loss = {}'.format(i, loss.item()))
		loss.backward()
		optim_multi.step()

save(model_multi_class.state_dict(), 'drug_test_multi.pt')

# Binary model training
train_dataset = TrainDataset('Binary')
train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

N_epochs = 250
for batch_idx, data in enumerate(train_loader):
	X_train, Y_train_bin = data
	for i in range(1, N_epochs + 1):
		optim_bin.zero_grad()
		Y_hat = model_binary.forward(X_train)
		loss = criterion_bin(Y_hat, Y_train_bin)
		print('Epoch = {},  loss = {}'.format(i, loss.item()))
		loss.backward()
		optim_bin.step()

save(model_binary.state_dict(), 'drug_test_binary.pt')
