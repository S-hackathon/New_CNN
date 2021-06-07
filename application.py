# filter warnings
from types import LambdaType
import warnings
import keras
from sklearn import *
from keras.layers.core import Dense
warnings.simplefilter(action="ignore", category=FutureWarning)

# keras imports
from keras.applications.inception_v3 import InceptionV3, preprocess_input
from keras.preprocessing import image
from keras.models import Model
from keras.models import model_from_json
from keras.layers import Input
# other imports
from sklearn.preprocessing import LabelEncoder
import numpy as np
import glob
import cv2
import h5py
import os
import json
import pickle as cPickle
import datetime
import time
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# load the user configs
with open('conf/conf.json') as f:    
	config = json.load(f)

# config variables
model_name = config["model"]
weights = config["weights"]
include_top = config["include_top"]
train_path = config["train_path"]
features_path = config["features_path"]
labels_path = config["labels_path"]
#test_size = config["test_size"]
results = config["results"]
model_path = config["model_path"]
classifier_path = config["classifier_path"]
# create the pretrained models
# check for pretrained weight usage or not
# check for top layers to be included or not
base_model = InceptionV3(include_top=False, weights=weights, input_tensor=Input(shape=(299,299,3)))

model = Model(base_model.input, base_model.output)
image_size = (299, 299)

print ("[INFO] successfully loaded base model and model...")

loaded_model = cPickle.load(open(classifier_path, 'rb'))

print ("[INFO] successfully Loaded Trained Model...")

cur_path = "test"
for test_path in glob.glob(cur_path + "/*.png"):
	#load = i + ".png"
	print ("[INFO] loading", test_path,"image ")
	img = image.load_img(test_path, target_size=image_size)
	x = image.img_to_array(img)
	#x=x.reshape(-2)
	x = np.expand_dims(x, axis=0)
	#print(x.shape)
	
	x = preprocess_input(x)
	feature = model.predict(x)
	#flat = LambdaType(lambda x: keras.backend.batch_flatten(x))
	flat=feature.flatten()
	flat=np.expand_dims(flat,axis=1)
	flat=np.transpose(flat)
	preds = loaded_model.predict(flat)
	#print (preds)
	if(preds==0):
		print('bird')
	else:
		print('Not bird')
	show_image = cv2.imread(test_path)
	show_image = cv2.resize(show_image, (500, 500)) 
	#disease = preds
	#print (show_image)
	#cv2.putText(show_image, preds, (40,50), cv2.FONT_HERSHEY_SIMPLEX, .75, (0, 255, 0), 2)
	cv2.imshow("result",show_image)
	cv2.waitKey(0)