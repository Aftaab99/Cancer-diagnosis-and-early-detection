from src.SkinCancer.Dataloader import TestDataset
from src.SkinCancer.Model import Net
from torch import load
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

model = Net()
model.load_state_dict(load('model_skin_cancer_epoch3.pt'))
model.eval()
test_dataset = TestDataset()
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

acc = 0
f1 = 0
n_test_batches = 0
y_pred_sum = 0

for _, data in enumerate(test_loader):
	test_x, test_y, sex, age = data
	test_x = test_x.float()
	test_y = test_y.float()
	sex = sex.float()
	age = age.float()
	y_pred = model.forward(test_x, sex, age)
	y_pred = y_pred.detach().numpy() >= 0.5
	acc += accuracy_score(test_y.numpy(), y_pred)
	n_test_batches += 1

print('Accuracy={}'.format(acc / n_test_batches))
# Accuracy 82%
