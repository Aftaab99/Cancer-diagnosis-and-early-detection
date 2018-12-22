from src.CancerDrugTest.Dataloader import TestDataset
from torch.utils.data import DataLoader
import numpy as np
from torch import load
from src.CancerDrugTest.Model import MultiClassNet, BinaryNet
from sklearn.metrics import accuracy_score, f1_score

# Testing Multiclass model
test_dataset = TestDataset('Multiclass')
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
model = MultiClassNet()
model.load_state_dict(load('drug_test_multi.pt'))
model.eval()
for _, data in enumerate(test_loader):
	X_test, Y_test = data
	accuracy = accuracy_score(y_true=Y_test, y_pred=np.argmax(model.forward(X_test).detach().numpy(), axis=1))
	print('Accuracy(multiclass)={}'.format(accuracy))

# Testing binary model
test_dataset = TestDataset('Binary')
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
model = BinaryNet()
model.load_state_dict(load('drug_test_binary.pt'))
model.eval()
for _, data in enumerate(test_loader):
	X_test, Y_test_bin = data
	accuracy = f1_score(y_true=Y_test_bin, y_pred=np.round(model.forward(X_test).detach().numpy()))
	print('Accuracy(binary)={}'.format(accuracy))
