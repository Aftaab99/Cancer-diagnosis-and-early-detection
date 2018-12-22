import pandas as pd
import numpy as np
from scipy import sparse
import h5py
from glob import glob


def load_as_dataframe(path):
	if path == 'Dataset/pubchem_neg_sample.h5':
		hf = h5py.File(path, 'r')
		ids = hf["chembl_id"].value  # the name of each molecules
		print(path)
		ap = sparse.csr_matrix((hf["ap"]["data"], hf["ap"]["indices"], hf["ap"]["indptr"]),
							   shape=[len(hf["ap"]["indptr"]) - 1, 2039])
		mg = sparse.csr_matrix((hf["mg"]["data"], hf["mg"]["indices"], hf["mg"]["indptr"]),
							   shape=[len(hf["mg"]["indptr"]) - 1, 2039])
		tt = sparse.csr_matrix((hf["tt"]["data"], hf["tt"]["indices"], hf["tt"]["indptr"]),
							   shape=[len(hf["tt"]["indptr"]) - 1, 2039])
		features = sparse.hstack([ap, mg, tt]).toarray()
		labels = 9 * np.ones(shape=(features.shape[0], 1))
		data = np.concatenate((features, labels), axis=1)[1:1501, :]
		del features, labels
		return pd.DataFrame(data)

	class_label = 9
	if path == 'Dataset/cdk2.h5':
		class_label = 1
	elif path == 'Dataset/egfr_erbB1.h5':
		class_label = 2
	elif path == 'Dataset/gsk3b.h5':
		class_label = 3
	elif path == 'Dataset/hgfr.h5':
		class_label = 4
	elif path == 'Dataset/map_k_p38a.h5':
		class_label = 5
	elif path == 'Dataset/tpk_lck.h5':
		class_label = 6
	elif path == 'Dataset/tpk_src.h5':
		class_label = 7
	elif path == 'Dataset/vegfr2.h5':
		class_label = 8

	hf = h5py.File(path, 'r')
	ids = hf["chembl_id"].value  # the name of each molecules
	print(path)
	ap = sparse.csr_matrix((hf["ap"]["data"], hf["ap"]["indices"], hf["ap"]["indptr"]),
						   shape=[len(hf["ap"]["indptr"]) - 1, 2039])
	mg = sparse.csr_matrix((hf["mg"]["data"], hf["mg"]["indices"], hf["mg"]["indptr"]),
						   shape=[len(hf["mg"]["indptr"]) - 1, 2039])
	tt = sparse.csr_matrix((hf["tt"]["data"], hf["tt"]["indices"], hf["tt"]["indptr"]),
						   shape=[len(hf["tt"]["indptr"]) - 1, 2039])
	features = sparse.hstack([ap, mg, tt]).toarray()
	# the samples' features, each row is a sample, and each sample has 3*2039 features
	labels = (class_label * hf["label"].value).reshape(-1, 1)
	print(labels.shape)
	print(features.shape)
	data = np.concatenate((features, labels), axis=1)[1:1501, :]
	del features, labels

	data = pd.DataFrame(data)
	data = data.loc[data[6117] != 0]
	return pd.DataFrame(data)


all_files = glob('Dataset/*.h5')
combined_dataset = load_as_dataframe(all_files[0])
combined_dataset.to_csv('Dataset/combined_data.csv', index=False)
for file in all_files[1:]:
	data = load_as_dataframe(file)
	data.to_csv('Dataset/combined_data.csv', mode='a', index=False, header=False)

del combined_dataset
