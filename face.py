from keras.engine import Model
from keras import models
from keras import layers
from keras.layers import Input
from keras.preprocessing import image
from keras_vggface.vggface import VGGFace
import numpy as np
from keras_vggface import utils
import scipy.spatial
import cv2
import os
import glob
import pickle

def load_stuff(filename):
	saved_stuff = open(filename, "rb")
	stuff = pickle.load(saved_stuff)
	saved_stuff.close()
	return stuff
	
face_size = 224
precompute_features_file="data/precompute_features.pickle"
precompute_features_map = load_stuff(precompute_features_file)
face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt.xml")
model = VGGFace(model='resnet50',include_top=False,input_shape=(224, 224, 3),pooling='avg')  # pooling: None, avg or max
print("Loading VGG Face model done")


    
def draw_label(image, point, label, font=cv2.FONT_HERSHEY_SIMPLEX,font_scale=1, thickness=2):
	size = cv2.getTextSize(label, font, font_scale, thickness)[0]
	x, y = point
	cv2.rectangle(image, (x, y - size[1]), (x + size[0], y), (255, 0, 0), cv2.FILLED)
	cv2.putText(image, label, point, font, font_scale, (255, 255, 255), thickness)
        
def identify_face(features, threshold=100):
	distances = []
	for person in precompute_features_map:
		person_features = person.get("features")
		distance = scipy.spatial.distance.euclidean(person_features, features)
		distances.append(distance)
	min_distance_value = min(distances)
	min_distance_index = distances.index(min_distance_value)
	if min_distance_value < threshold:
		return precompute_features_map[min_distance_index].get("name")
	else:
		return "?"

def crop_face(imgarray, section, margin=20, size=224):
	img_h, img_w, _ = imgarray.shape
	if section is None:
		section = [0, 0, img_w, img_h]
	(x, y, w, h) = section
	margin = int(min(w,h) * margin / 100)
	x_a = x - margin
	y_a = y - margin
	x_b = x + w + margin
	y_b = y + h + margin
	if x_a < 0:
		x_b = min(x_b - x_a, img_w-1)
		x_a = 0
	if y_a < 0:
		y_b = min(y_b - y_a, img_h-1)
		y_a = 0
	if x_b > img_w:
		x_a = max(x_a - (x_b - img_w), 0)
		x_b = img_w
	if y_b > img_h:
		y_a = max(y_a - (y_b - img_h), 0)
		y_b = img_h
	cropped = imgarray[y_a: y_b, x_a: x_b]
	resized_img = cv2.resize(cropped, (size, size), interpolation=cv2.INTER_AREA)
	resized_img = np.array(resized_img)
	return resized_img, (x_a, y_a, x_b - x_a, y_b - y_a)
	
	
print("In main")
video_capture = cv2.VideoCapture(0)
while True:
	if not video_capture.isOpened():
		sleep(5)
	ret, frame = video_capture.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=10,minSize=(64, 64))
	face_imgs = np.empty((len(faces),face_size, face_size, 3))
	for i, face in enumerate(faces):
		face_img, cropped = crop_face(frame, face, margin=10, size=face_size)
		(x, y, w, h) = cropped
		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
		face_imgs[i, :, :, :] = face_img
	if len(face_imgs) > 0:
		features_faces = model.predict(face_imgs)
		predicted_names = [identify_face(features_face) for features_face in features_faces]
	for i, face in enumerate(faces):
		label = "{}".format(predicted_names[i])
		draw_label(frame, (face[0], face[1]), label)
	cv2.imshow('Keras Faces', frame)
	if cv2.waitKey(5) == 27:
		break
video_capture.release()
cv2.destroyAllWindows()

