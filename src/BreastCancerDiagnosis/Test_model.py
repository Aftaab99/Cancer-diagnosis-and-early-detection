from sklearn.metrics import f1_score, accuracy_score
from torch import load
from src.BreastCancerDiagnosis.Model import BreastCancerModel
import numpy as np
from torch.utils.data import DataLoader
from src.BreastCancerDiagnosis.Dataloader import TestDataset

model = BreastCancerModel()
model.load_state_dict(load('breast_cancer_diagnosis.pt'))

test_dataset = TestDataset()
test_generator = DataLoader(test_dataset, num_workers=2, shuffle=True, batch_size=32)

f1 = 0
steps = 0
acc = 0
for i, data in enumerate(test_generator):
	test_x, test_y = data
	test_y = test_y.view(-1, 1)

	pred = model.forward(test_x)
	pred = np.round(pred.detach().numpy())
	test_y = np.array(test_y)
	f1 += f1_score(test_y, pred)
	acc += accuracy_score(test_y, pred)
	steps += 1
	if i % 32 == 0:
		print('Step {}'.format(i, f1))

print('Average f1 score across batches={}'.format(f1 / steps))
print('Average accuracy={}'.format(acc / steps))
# F1 score=0.71, accuracy=84.1%