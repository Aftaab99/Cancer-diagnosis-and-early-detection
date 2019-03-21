from flask import Flask, render_template, request, jsonify, send_file, make_response
import pandas as pd
import CancerDrugTest.Predict as Predict
from PIL import Image, ImageDraw
import random
import os
import ntpath
import base64
import pickle
from io import BytesIO
from torch import Tensor, load
import numpy as np
from BreastCancerDiagnosis.Model import BreastCancerModel
from SkinCancer.Model import SkinCancerModel
from BreastCancerPrognosis.Model import BreastCancerPrognosisModel

app = Flask(__name__)


def serve_pil_image(pil_img):
	img_io = BytesIO()
	pil_img.save(img_io, format='JPEG', quality=100)
	img_io.seek(0)
	img_i = base64.b64encode(img_io.getvalue()).decode()
	return {'image': img_i}


@app.route('/diagnosis/breast_cancer', methods=['GET', 'POST'])
def breast_cancer_diagnosis():
	def sliding_window(model, img, step_size):
		size_x, size_y = img.size
		pred_img = ImageDraw.Draw(img)
		img_array = np.array(img)
		for x in range(0, size_x - step_size, step_size):
			for y in range(0, size_y - step_size, step_size):
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
		model.eval()

		if int(request.form.get('use_random')) == 1:
			random_path = os.path.join('BreastCancerDiagnosis/RandomData/images',
									   random.choice(os.listdir('BreastCancerDiagnosis/RandomData/images')))
			img = Image.open(random_path).convert('RGB').resize([400, 400])
			img = sliding_window(model, img, 32)
			return jsonify(serve_pil_image(img))

		elif int(request.form.get('use_random')) == 0:
			print('File: {}'.format(request.files.get('breast-cancer-diagnosis-image')))
			img = Image.open(request.files.get('breast-cancer-diagnosis-image')).convert('RGB')
			img = sliding_window(model, img, 32)
			return jsonify(serve_pil_image(img))

		return jsonify({'error': 1})


@app.route('/prognosis/breast_cancer', methods=['GET', 'POST'])
def breast_cancer_prognosis():
	if request.method == 'GET':
		return render_template('breast-cancer-prognosis.html')
	else:
		form_data = request.form
		clump_thickness = form_data.get('clump_thickness')
		uniformity_cell_size = form_data.get('uniformity_cell_size')
		uniformity_cell_shape = form_data.get('uniformity_cell_shape')
		marginal_adhesion = form_data.get('marginal_adhesion')
		single_epithelial_cell_size = form_data['single_epithelial_cell_size']
		bare_nuclei = form_data['bare_nuclei']
		bland_chromatin = form_data['bland_chromatin']
		normal_nuclei = form_data['normal_nuclei']
		mitosis = form_data['mitosis']
		prediction_tensor = np.array([clump_thickness, uniformity_cell_size, uniformity_cell_shape,
									  marginal_adhesion, single_epithelial_cell_size, bare_nuclei,
									  bland_chromatin, normal_nuclei, mitosis])
		prediction_tensor = prediction_tensor.reshape(1, -1)
		with open('BreastCancerPrognosis/scaler.pkl', 'rb') as f:
			scaler = pickle.load(f)
		prediction_tensor = scaler.transform(prediction_tensor)
		prediction_tensor = Tensor(prediction_tensor)
		model = BreastCancerPrognosisModel()
		model.load_state_dict(load('BreastCancerPrognosis/breast_cancer_prognosis.pt'))
		model.eval()
		pred = model.forward(prediction_tensor).detach().numpy()
		if np.round(pred) == 1:
			return jsonify({'class': 1, 'probability': pred.item()})
		else:
			return jsonify({'class': 0})


@app.route('/prognosis/cervical_cancer', methods=['GET', 'POST'])
def cervical_cancer_prognosis():
	def get_prediction(model, x):
		pass

	if request.method == 'GET':
		return render_template('cervical-cancer-prognosis.html')
	else:
		form_data = request.form
		age = form_data.get('age')
		n_xp = form_data.get('no_sexual_partners')
		fs = form_data.get('first_sex_age')
		n_preg = form_data.get('no_of_pregnancies')
		n_std = form_data.get('no_of_std')
		d_cin = form_data.get('diagnosed_with_cin')
		d_hpv = form_data.get('diagnosed_with_hpv')
		pred_vector = np.array([age, n_xp, fs, n_preg, n_std, d_cin, d_hpv]).reshape(1, -1)
		with open('CervicalCancer/model_bio.pkl', 'rb') as f:
			model_bio = pickle.load(f)
		with open('CervicalCancer/model_cit.pkl', 'rb') as f:
			model_cit = pickle.load(f)
		with open('CervicalCancer/model_hins.pkl', 'rb') as f:
			model_hins = pickle.load(f)
		with open('CervicalCancer/model_sch.pkl', 'rb') as f:
			model_sch = pickle.load(f)
		bio_pred = model_bio.predict(pred_vector).item()
		cit_pred = model_cit.predict(pred_vector).item()
		hins_pred = model_hins.predict(pred_vector).item()
		sch_pred = model_sch.predict(pred_vector).item()

		return jsonify({'bio': bio_pred, 'cit': cit_pred, 'hins': hins_pred, 'sch': sch_pred})


@app.route('/diagnosis/skin_cancer', methods=['GET', 'POST'])
def skin_cancer():
	def get_prediction(pil_img, age, gender):
		pil_img = pil_img.resize([128, 128])
		age_t = Tensor([age]).reshape(1, 1)
		gender_t = Tensor([gender]).reshape(1, 1)
		img_t = Tensor(np.array(pil_img)).reshape(1, 3, 128, 128)
		model = SkinCancerModel()
		model.load_state_dict(load('SkinCancer/model_skin_cancer_epoch40.pt'))
		model.eval()
		pred = np.round(model.forward(img_t, gender_t, age_t).detach().numpy())
		return int(pred)

	if request.method == 'GET':
		return render_template('skin-cancer.html')
	else:
		if 'use_random' in request.form.keys():
			if int(request.form.get('use_random')) == 1:
				random_path = os.path.join('SkinCancer/RandomData/images',
										   random.choice(os.listdir('SkinCancer/RandomData/images')))
				file_name = ntpath.basename(ntpath.splitext(random_path)[0])
				metadata = pd.read_csv('SkinCancer/RandomData/meta_data.csv')
				row = metadata.loc[(metadata['image_id'] == file_name)]
				gender = row['sex'].item()

				if gender == 'male':
					gender = 1
				elif gender == 'female':
					gender = 0
				else:
					gender = -1

				img = Image.open(random_path).convert('RGB')
				res = serve_pil_image(img.resize([400, 400]))
				res['prediction'] = get_prediction(img, age, gender)
				return jsonify(res)
			else:
				print('File: {}'.format(request.files.get('skin-cancer-diagnosis-image')))
				age = int(request.form.get('age'))
				gender = request.form.get('gender')
				img = Image.open(request.files.get('skin-cancer-diagnosis-image')).convert('RGB')
				if gender == 'Female':
					gender = 0
				elif gender == 'Male':
					gender = 1
				else:
					gender = -1

				res = serve_pil_image(img.resize([400, 400]))
				res['prediction'] = get_prediction(img, age, gender)
				return jsonify(res)

		return jsonify({'error': 1})


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
