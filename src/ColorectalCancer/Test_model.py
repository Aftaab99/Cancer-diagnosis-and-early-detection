from ColorectalCancer.Dataloader import TestDataset
from ColorectalCancer.Model import Net
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
import numpy as np

model = Net()
test_datatset = TestDataset()
test_loader = DataLoader(test_datatset, batch_size=32, shuffle=True)

batch_acc = 0
batch_f1 = 0
n_batches = 0
print(len(test_datatset))
for step, data in enumerate(test_loader):
	test_x, test_y = data
	test_x = test_x.type('torch.FloatTensor')
	n_batches+=1
	y_pred = model.forward(test_x)
	y_pred = np.round(y_pred.detach().numpy())
	y_true = np.array(test_y)
	batch_acc+=accuracy_score(y_true, y_pred)
	batch_f1+=f1_score(y_true, y_pred)

print('Accuracy=%.4f' %(batch_acc/n_batches))
# Accuracy 79%