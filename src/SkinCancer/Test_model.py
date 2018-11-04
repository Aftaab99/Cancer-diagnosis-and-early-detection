from SkinCancer.Dataloader import TestDataset
from SkinCancer.Model import Net
from torch import load
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

model = Net()
model.load_state_dict(load('model_skin_cancer.pt'))

test_dataset = TestDataset()
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

acc = 0
f1 = 0
n_test_batches = 0
y_pred_sum = 0

for _, data in enumerate(test_loader):
	test_x, test_y = data
	test_y.type('torch.FloatTensor')
	y_pred = model.forward(test_x)
	y_pred = np.round(y_pred.detach().numpy())
	acc += accuracy_score(test_y.numpy(), y_pred)
	f1 += f1_score(test_y.numpy(), y_pred)
	n_test_batches += 1
	y_pred_sum+=np.sum(y_pred)

print('Accuracy={}'.format(acc/n_test_batches))
print('Number predicted as 1={}'.format(y_pred_sum))
print('F1={}'.format(f1))
# Accuracy 81%