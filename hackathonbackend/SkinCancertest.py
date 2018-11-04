from PIL import Image, ImageDraw
from BreastCancerDiagnosis.Model import Net as BreastCancerModel
import numpy as np
from torch import load
from torchvision import transforms
from SkinCancer.Model import Net as SkinCancerModel

x=Image.open('samples/skin_cancer_benign.jpg').convert('RGB').resize([300,300])
x.show()
img = np.array(x)
model = SkinCancerModel()
model.load_state_dict(load('SkinCancer/model_skin_cancer.pt'))
transform = transforms.Compose([transforms.ToPILImage(),
									transforms.ToTensor(),
									transforms.Normalize(mean=[0], std=[1])])
img_tensor = transform(img).view(1, 3, 300, 300)
y_pred = model.forward(img_tensor).detach().numpy()
y_pred = np.round(y_pred)
print('Predicted_class:', end=' ')
if y_pred == 1:
	print('Malignant tumor')
else:
	print('Benign tumor')
