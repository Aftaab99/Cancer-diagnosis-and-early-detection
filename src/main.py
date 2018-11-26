from PIL import Image, ImageDraw
from src.BreastCancerDiagnosis.Model import Net as BreastCancerModel
import numpy as np
from torch import Tensor
from torch import load
from torchvision import transforms
from flask import Flask, request, render_template, send_from_directory
import pickle
from werkzeug.utils import secure_filename
import pandas as pd
from src.SkinCancer.Model import Net as SkinCancerModel
from src.ColorectalCancer.Model import Net as ColorectalModel
import json
from keras.models import load_model

app = Flask(__name__)


@app.route('/breast_cancer_prognosis')
def breast_cancer_prognosis():
	return render_template('breast_cancer_prognosis.html', pred="<div></div>")


@app.route('/cancer_drug_test')
def cancer_drug_test():
	return render_template('cancer_drug_test.html')


@app.route('/cervical_cancer')
def cervical_cancer():
	return render_template('cervical_cancer.html')


@app.route('/colorectal_cancer')
def colorectal_cancer():
	return render_template('colorectal_cancer.html', pred="Upload a image to get a diagnosis")


@app.route('/skin_cancer')
def skin_cancer():
	return render_template('skin_cancer.html', pred='Upload tumor image to predict')


@app.route('/symptoms')
def symptoms():
	return render_template('symptoms.html')


@app.route('/')
def index():
	return render_template('index.html')


# Gets the image
@app.route('/breast_cancer_diagnosis', methods=['POST', 'GET'])
def breast_cancer_diagnosis():
	if request.method == 'GET':
		return render_template('breast_cancer_diagnosis.html', width=0, height=0)
	imgf = request.files.get('breast_cancer_image')
	if imgf is None:
		return "Error. Upload an image"
	else:
		img = Image.open(imgf.stream)
		x, y = img.size
		x_new = x
		y_new = y
		if x > 400:
			x_new = 400
		if y_new > 400:
			y_new = 400

		img = img.resize([x_new, y_new])

		model = BreastCancerModel()
		model.load_state_dict(load('BreastCancerDiagnosis/breast_cancer_diagnosis.pt'))

		def predict_crop(img_crop):
			transform = transforms.Compose([transforms.ToPILImage(),
											transforms.ToTensor(),
											transforms.Normalize(mean=[0], std=[1])])
			img_tensor = transform(img_crop).view(1, 3, 32, 32)
			y_pred = model.forward(img_tensor)
			y_pred = np.round(y_pred.detach().numpy())
			if y_pred == 0:
				return 'Benign'
			else:
				return 'Malignant'

		draw_img = ImageDraw.Draw(img, 'RGB')
		img_array = np.array(img)
		for i in range(0, x_new - 32, 32):
			for j in range(0, y_new - 32, 32):
				img_crop = img_array[i:i + 32, j:j + 32, :]
				pred = predict_crop(img_crop)
				print('Image size={}, {}'.format(img_crop.shape, pred))
				if pred == 'Malignant':
					draw_img.rectangle(((i, j), (i + 32, j + 32)), outline='red')
		img_n = imgf.filename
		img.save('static/' + img_n, 'JPEG')

		return render_template('breast_cancer_diagnosis.html', img_name=img_n, width=400, height=400)


@app.route('/get_skin_cancer_prediction', methods=['GET', 'POST'])
def get_skin_cancer_prediction():
	img = request.files.get('skin_cancer_image')
	if img == None:
		return 'Error occured. Upload an image'
	sex = request.form.get('Sex')
	age = request.form.get('Age')

	sex_id = '0'
	print("{}, {}".format(sex, age))
	if sex.lower() == 'male':
		print('True...')
		sex_id = '1'

	sex = sex_id
	sex = int(sex)
	print(sex)
	age = int(age)
	img = Image.open(img.stream).convert('L').resize([128, 128])
	img_array = np.array(img)
	model = SkinCancerModel()
	model.load_state_dict(load('SkinCancer/model_skin_cancer_epoch3.pt'))

	img_tensor = Tensor(img_array).view(1, 1, 128, 128)
	y_pred = model.forward(img_tensor, Tensor([sex]).float(), Tensor([age])).detach().numpy()
	y_pred = np.round(y_pred)
	pred = 'Upload to get a prediction'
	if y_pred == 1:
		pred = "You've a malignant melanoma tumor"
	else:
		pred = "Your tumor is benign"
	return render_template('skin_cancer.html', pred=pred)


@app.route('/get_colorectal_cancer_prediction', methods=['POST'])
def get_colorectal_cancer_prediction():
	img = request.files.get('colorectal_cancer_image')
	if img is None:
		return 'Error occured. Upload an image'
	img_array = np.array(img.stream)
	model = ColorectalModel()
	model.load_state_dict(load('ColorectalCancer/ColorectalCancer.pt'))
	transform = transforms.Compose([transforms.ToPILImage(),
									transforms.ToTensor(),
									transforms.Normalize(mean=[0], std=[1])])
	img_tensor = transform(img_array).view(1, 3, 150, 150)
	y_pred = model.forward(img_tensor).detach().numpy()
	y_pred = np.round(y_pred)
	pred = 'Benign'
	if y_pred == 1:
		pred = 'Malignant tumors present'
	render_template(render_template('colorectal_cancer.html'), pred=pred)


@app.route('/get_cervical_cancer_prediction', methods=['POST', 'GET'])
def get_cervical_cancer_prediction():
	data = request.form
	print('OOOOOOOOO')
	age = data['age']
	no_of_partners = data['no_of_partners']
	first_sexual_intercourse = data['first_sexual_intercourse']
	no_of_pregnancies = data['no_of_pregnancies']
	no_of_stds = data['no_of_stds']
	diagnosed_with_cin = data['diagnosed_with_cin']
	diagnosed_with_hpv = data['diagnosed_with_hpv']
	x = [age, no_of_partners, first_sexual_intercourse, no_of_pregnancies, no_of_stds, diagnosed_with_cin,
		 diagnosed_with_hpv]

	for i in range(len(x)):
		if x[i] == None:
			x[i] = 0

	model_h = pickle.load(open('CervicalCancer/model_hins.pkl', 'rb'))
	model_s = pickle.load(open('CervicalCancer/model_sch.pkl', 'rb'))
	model_c = pickle.load(open('CervicalCancer/model_cit.pkl', 'rb'))
	model_b = pickle.load(open('CervicalCancer/model_bio.pkl', 'rb'))
	x = np.array(x).reshape(1, -1)
	scaler = pickle.load(open('BreastCancerPrognosis/scaler.pkl', 'rb'))
	x = scaler.transform(x)
	pred_h = model_h.predict(x)
	pred_s = model_s.predict(x)
	pred_c = model_c.predict(x)
	pred_b = model_b.predict(x)
	label1 = 'Risk of cervical cancer'
	label2 = 'Risk of cervical cancer'
	label3 = 'Risk of cervical cancer'
	label4 = 'Risk of cervical cancer'
	if pred_h == 0:
		label1 = 'No risk'
	if pred_s == 0:
		label2 = 'No risk'
	if pred_c == 0:
		label3 = 'No risk'
	if pred_b == 0:
		label4 = 'No risk'

	pred = '<h3>Hinselmann: %s</h3><h3>Schiller: %s</h3><h3>Citology: %s</h3><h3>Biopsy: %s</h3>' % (
	label1, label2, label3, label4)
	return render_template('colorectal_cancer.html', pred=pred)


@app.route('/get_breast_cancer_prognosis', methods=['POST'])
def get_breast_cancer_prognosis():
	print('XXXXXXXXXXXXX')
	data = request.form
	clump_thickness = data['clump_thickness']
	uniformity_cell_size = data['uniformity_cell_size']
	uniformity_cell_shape = data['uniformity_cell_shape']
	marginal_adhesion = data['marginal_adhesion']
	single_epithelial_cell_size = data['single_epithelial_cell_size']
	bare_nuclei = data['bare_nuclei']
	bland_chromatin = data['bland_chromatin']
	normal_nuclei = data['normal_nuclei']
	mitosis = data['mitosis']
	x = [clump_thickness, uniformity_cell_size, uniformity_cell_shape, marginal_adhesion, single_epithelial_cell_size,
		 bare_nuclei, bland_chromatin, normal_nuclei, mitosis]
	x = np.array(x).reshape(1, -1)
	model = load_model('BreastCancerPrognosis/breast_cancer_prognosis.h5')
	pred = model.predict(x)
	pred1 = np.round(pred)
	pos = '<h3>Prediction: Your tumor is probably on its way to become malignant</h3>'
	neg = '<h3>Prediction: Your tumor will probably stay benign</h3>'
	if pred1 == 1:
		return render_template('breast_cancer_prognosis.html', pred=pos)
	if pred1 == 0:
		return render_template('breast_cancer_prognosis.html', pred=neg)


@app.route('/test_drug', methods=['GET'])
def test_drug():
	return render_template('phd.html', pred='Upload a drug screening file for a prediction')

@app.route('/get_test_drug', methods=['POST'])
def get_test_drug():
	file = request.files.get('drug_file')
	if file is None:
		return 'No file'
	df = pd.read_csv(file)
	x = df.head(1).values
	comp = pickle.load(open('CancerDrugTest/pca_compressor.pkl', 'rb'))
	x = comp.transform(x).reshape(1, -1)
	scaler = pickle.load(open('CancerDrugTest/scaler.pkl', 'rb'))
	x = scaler.transform(x)

	model_binary = load_model('CancerDrugTest/cancer_drug_binary.h5')
	model_multiclass = load_model('CancerDrugTest/cancer_drug_multiclass.h5')

	pred_bin = model_binary.predict(x)
	pred_multiclass = np.argmax(model_multiclass.predict(x))
	label_dict = {0: 'Cyclin-dependent kinase 2', 1: 'Epidermal growth factor receptor erbB1',
				  2: 'Glycogen synthase kinase-3 beta', 3: 'Hepatocyte growth factor receptor',
				  4: 'MAP kinase p38 alpha',
				  5: 'Tyrosine-protein kinase LCK', 6: 'Tyrosine-protein kinase SRC',
				  7: 'Vascular endothelial growth factor receptor 2'}

	pred = 'The given kinase can be used as an inhibitor(Confidence score = {}'.format(float(pred_bin)) + '\n' + \
		   'It is probably a {}'.format(label_dict[pred_multiclass])
	return render_template('phd.html', pred=pred)


@app.route('/login', methods=['GET'])
def login():
	return render_template('login.html', error_res='')


@app.route('/login', methods=['POST'])
def complete_login():
	email = request.files.get('Email')
	password = request.files.get('password')
	if email is None or password is None or len(password) < 8:
		return render_template('login.html', error_res='Invalid or empty username/password')
	return 'Not designed yet'

if __name__ == '__main__':
	app.run()
