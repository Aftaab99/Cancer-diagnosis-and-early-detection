import requests
import os
import zipfile


def download_file_from_google_drive(id, destination):
	URL = "https://drive.google.com/uc?export=download"

	session = requests.Session()

	response = session.get(URL, params={'id': id}, stream=True)
	token = get_confirm_token(response)

	if token:
		params = {'id': id, 'confirm': token}
		response = session.get(URL, params=params, stream=True)

	save_response_content(response, destination)


def get_confirm_token(response):
	for key, value in response.cookies.items():
		if key.startswith('download_warning'):
			return value

	return None


def save_response_content(response, destination):
	CHUNK_SIZE = 32768

	with open(destination, "wb") as f:
		for chunk in response.iter_content(CHUNK_SIZE):
			if chunk:  # filter out keep-alive new chunks
				f.write(chunk)


def extract(file_path):
	zip_ref = zipfile.ZipFile(file_path, 'r')
	zip_ref.extractall(os.path.dirname(file_path))
	zip_ref.close()


if __name__ == "__main__":
	breast_cancer_files_id = '1IsleC3NZzj45h_Sy-caRgfvkABIdb-xZ'
	breast_cancer_files_destination = 'BreastCancerDiagnosis/RandomData.zip'
	skin_cancer_files_id = '1YA8oswI9bQsiXcmgkTg3J467mNv7DuYy'
	skin_cancer_destination = 'SkinCancer/RandomData.zip'

	if not os.path.exists('./BreastCancerDiagnosis/RandomData/'):
		os.mkdir('./BreastCancerDiagnosis/RandomData')
		print('Downloading breast cancer sample data...')
		download_file_from_google_drive(breast_cancer_files_id, breast_cancer_files_destination)
		print('Extracting data...')
		extract('./BreastCancerDiagnosis/RandomData.zip')
		os.remove('./BreastCancerDiagnosis/RandomData.zip')

	if not os.path.exists('./SkinCancer/RandomData/'):
		os.mkdir('./SkinCancer/RandomData')
		print('Downloading skin cancer sample data...')
		download_file_from_google_drive(skin_cancer_files_id, skin_cancer_destination)
		print('Extracting data...')
		extract('./SkinCancer/RandomData.zip')
		os.remove('./SkinCancer/RandomData.zip')


