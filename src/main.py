from flask import Flask, render_template, request
from flask_scss import Scss
app = Flask(__name__)

# Compiling SCSS to CSS files
Scss(app, asset_dir='static/scss/', static_dir='static/css/')

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
	if request.method == 'GET':
		return render_template('protein-inhibitor-discovery.html')
	else:
		pass

@app.route('/', methods=['GET'])
def home():
	return render_template('home.html')

if __name__ == '__main__':
	app.run()
