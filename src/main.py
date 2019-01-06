from flask import Flask, render_template, request
from flask_assets import Environment, Bundle
app = Flask(__name__)

# Compiling SCSS to CSS files
assets     = Environment(app)
assets.url = app.static_url_path
scss       = Bundle('style.scss', filters='pyscss', output='style.css')

assets.config['SECRET_KEY'] = 'secret!'
assets.config['PYSCSS_LOAD_PATHS'] = assets.load_path
assets.config['PYSCSS_STATIC_URL'] = assets.url
assets.config['PYSCSS_STATIC_ROOT'] = assets.directory
assets.config['PYSCSS_ASSETS_URL'] = assets.url
assets.config['PYSCSS_ASSETS_ROOT'] = assets.directory

assets.register('scss_all', scss)

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
	app.run(debug=True)
