from flask import Flask, render_template, request, jsonify
import pandas as pd
import CancerDrugTest.Predict as Predict
import random

app = Flask(__name__)


@app.route('/diagnosis/breast_cancer', methods=['GET', 'POST'])
def breast_cancer_diagnosis():
	if request.method == 'GET':
		return render_template('breast-cancer-diagnosis.html')
	else:
		pass


@app.route('/prognosis/breast_cancer', methods=['GET', 'POST'])
def breast_cancer_prognosis():
	if request.method == 'GET':
		return render_template('breast-cancer-prognosis.html')
	else:
		pass


@app.route('/prognosis/cervical_cancer', methods=['GET', 'POST'])
def cervical_cancer_prognosis():
	if request.method == 'GET':
		return render_template('cervical-cancer-prognosis.html')
	else:
		pass


@app.route('/diagnosis/colorectal_cancer', methods=['GET', 'POST'])
def colorectal_cancer():
	if request.method == 'GET':
		return render_template('colorectal-cancer-diagnosis.html')
	else:
		pass


@app.route('/diagnosis/skin_cancer', methods=['GET', 'POST'])
def skin_cancer():
	if request.method == 'GET':
		return render_template('skin-cancer.html')
	else:
		pass


@app.route('/drug_discovery/protein_inhibitors', methods=['GET', 'POST'])
def protein_inhibitors():
	if request.method == 'POST':
		if request.form.get('use_random') and int(request.form.get('use_random')) == 1:
			print('Using random file...')
			dataset = pd.read_csv('CancerDrugTest/preprocessed_data.csv')
			nrows = dataset.shape[0]
			x = random.randint(0, nrows)
			print(x)
			random_sample = dataset.iloc[x, 0:80]
			print('Good till here')
			return jsonify(
				Predict.predict_single(random_sample.values.reshape(1, 80), 'CancerDrugTest/drug_test_binary.pt',
									   'CancerDrugTest/drug_test_multi.pt',
									   'CancerDrugTest/pca_compressor.pkl',
									   'CancerDrugTest/scaler.pkl', True))

		else:
			uploaded_fingerprint = request.files.get('protein-fingerprint-csv')
			fingerprint = pd.read_csv(uploaded_fingerprint).head(1)
			if fingerprint.shape[1] != 6117:
				return jsonify({'error': 1})
			return jsonify(Predict.predict_single(fingerprint.values, 'CancerDrugTest/drug_test_binary.pt',
												  'CancerDrugTest/drug_test_multi.pt',
												  'CancerDrugTest/pca_compressor.pkl',
												  'CancerDrugTest/scaler.pkl', False))

	else:
		return render_template('protein-inhibitor-discovery.html')


@app.route('/', methods=['GET'])
def home():
	return render_template('home.html')


if __name__ == '__main__':
	app.run(debug=True)
