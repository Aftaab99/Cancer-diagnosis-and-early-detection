import numpy as np
from torch import load, Tensor
from CancerDrugTest.Model import MultiClassNet, BinaryNet
import pickle


def predict_single(fingerprint, binary_model_path, multi_model_path, pca_path, scaler_path, use_random):
	# loading the models
	model_b = BinaryNet()
	model_b.load_state_dict(load(binary_model_path))
	model_m = MultiClassNet()
	model_m.load_state_dict(load(multi_model_path))

	# Inference mode
	model_m.eval()
	model_b.eval()

	# Load the PCA and scaler objects and transform the sample
	with open(pca_path, 'rb') as f:
		pca = pickle.load(f)
	with open(scaler_path, 'rb') as f:
		scaler = pickle.load(f)

	if not use_random:
		fingerprint = pca.transform(fingerprint)
	print(fingerprint.shape)
	fingerprint = scaler.transform(fingerprint)
	fingerprint = Tensor(fingerprint)

	# Feed into the loaded model
	binary_confidence = model_b.forward(fingerprint).detach().item()
	multiclass_pred = model_m.forward(fingerprint).detach().numpy()
	multiclass_confidence = np.max(multiclass_pred).item()
	multiclass_classification = np.argmax(multiclass_pred)

	multiclass_classification_protein = ''
	if multiclass_classification == 1:
		multiclass_classification_protein = 'Cyclin-dependent kinase 2'
	elif multiclass_classification == 2:
		multiclass_classification_protein = 'Epidermal growth factor receptor erbB1'
	elif multiclass_classification == 3:
		multiclass_classification_protein = 'Glycogen synthase kinase-3 beta'
	elif multiclass_classification == 4:
		multiclass_classification_protein = 'Hepatocyte growth factor receptor'
	elif multiclass_classification == 5:
		multiclass_classification_protein = 'MAP kinase p38 alpha'
	elif multiclass_classification == 6:
		multiclass_classification_protein = 'Tyrosine-protein kinase LCK'
	elif multiclass_classification == 7:
		multiclass_classification_protein = 'Tyrosine-protein kinase SRC'
	elif multiclass_classification == 8:
		multiclass_classification_protein = 'Vascular endothelial growth factor receptor 2'

	res = {'binary_prediction': np.round(binary_confidence).item(),
		   'binary_probability': '%.2f' %(binary_confidence * 100),
		   'multiclass_prediction': multiclass_classification_protein,
		   'multiclass_probablity': '%.2f' %(multiclass_confidence*100)}
	return res

