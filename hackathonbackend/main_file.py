from PIL import Image, ImageDraw
from hackathonbackend.BreastCancerDiagnosis.Model import Net as BreastCancerModel
import numpy as np
from torch import load
from torchvision import transforms
from flask import Flask, request, render_template, url_for, redirect, flash
import pickle
import pandas as pd
import json
from keras.models import load_model

app=Flask(__name__)

@app.route('/breast_cancer_diagnosis', methods=['GET', 'POST'])
def breast_cancer_diagnosis():
	if request.method == 'GET':
		return render_template('breast_cancer_diagnosis.html', img_name='images/upload_image.png', width=100, height=100,
							   upload_message='Upload an image to analyse')

	else:
		img=request.files.get('breast_cancer_image')
		if img is None:
			flash('Error. Please upload an image')
			return redirect(url_for('breast_cancer_diagnosis'))

		img = Image.open(img.stream)
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
		img_n = 'images/temp.jpg'
		img.save('static/' + img_n, 'JPEG')

		return render_template('breast_cancer_diagnosis.html', img_name=img_n, width=400, height=400)

@app.route('/')
def index():
	return render_template('index.html')

if __name__=='__main__':
	app.run()