from flask import Flask, render_template, request

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
		return render_template('breast_cancer_prognosis.html')
	else:
		pass


@app.route('/prognosis/cervical_cancer', methods=['GET', 'POST'])
def cervical_cancer_prognosis():
	if request.method == 'GET':
		return render_template('cervical_cancer_prognosis.html')
	else:
		pass


@app.route('/diagnosis/colorectal_cancer', methods=['GET', 'POST'])
def colorectal_cancer():
	if request.method == 'GET':
		return render_template('colorectal_cancer_diagnosis.html')
	else:
		pass


@app.route('/diagnosis/skin_cancer', methods=['GET', 'POST'])
def skin_cancer():
	if request.method == 'GET':
		return render_template('skin_cancer.html')
	else:
		pass


@app.route('/drug_discovery/protein_inhibitors', methods=['GET', 'POST'])
def protein_inhibitors():
	if request.method == 'GET':
		return render_template('protein_inhibitor_discovery.html')
	else:
		pass


if __name__ == '__main__':
	app.run()
