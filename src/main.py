from flask import Flask, render_template, request, jsonify, send_file, make_response
import pandas as pd
import CancerDrugTest.Predict as Predict
from PIL import Image, ImageDraw
import random, os
from io import BytesIO
from torch import Tensor, load
import numpy as np
import base64
from BreastCancerDiagnosis.Model import BreastCancerModel

app = Flask(__name__)


def serve_pil_image(pil_img):
	img_io = BytesIO()
	pil_img.save(img_io, format='JPEG', quality=100)
	img_io.seek(0)
	img_i = base64.b64encode(img_io.getvalue()).decode()
	with open('image_enc.txt', 'w') as f:
		f.write(img_i)
	return jsonify({'image': img_i})


@app.route('/diagnosis/breast_cancer', methods=['GET', 'POST'])
def breast_cancer_diagnosis():
	def sliding_window(model, img, step_size):
		img = img.resize([400, 400])
		pred_img = ImageDraw.Draw(img)
		img_array = np.array(img)
		for x in range(0, 400 - step_size, step_size):
			for y in range(0, 400 - step_size, step_size):
				img_crop = img_array[x:x + 32, y:y + 32]
				if img_crop.shape[0] != 32 or img_crop.shape[1] != 32:
					continue
				img_crop_t = Tensor(img_crop).view(1, 3, 32, 32)
				y_pred = model.forward(img_crop_t)
				y_pred = np.round(y_pred.detach().numpy())
				if y_pred == 1:
					pred_img.rectangle(((x, y), (x + 32, y + 32)), outline='red')
		return img

	if request.method == 'GET':
		return render_template('breast-cancer-diagnosis.html')
	else:

		model = BreastCancerModel()
		model.load_state_dict(load('BreastCancerDiagnosis/breast_cancer_diagnosis.pt'))

		if int(request.form.get('use_random')) == 1:
			random_path = os.path.join('BreastCancerDiagnosis/RandomData',
									   random.choice(os.listdir('BreastCancerDiagnosis/RandomData')))

			img = Image.open(random_path).convert('RGB').resize([400, 400])
			img = sliding_window(model, img, 32)
			print('sending file...')
			return serve_pil_image(img)

		elif int(request.form.get('use_random')) == 0:
			print('File: {}'.format(request.files.get('breast-cancer-diagnosis-image')))
			img = Image.open(request.files.get('breast-cancer-diagnosis-image')).convert('RGB')
			img = sliding_window(model, img, 32)
			return serve_pil_image(img)

		return jsonify({'error': 1})


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
