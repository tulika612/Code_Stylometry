#creating database
import cv2, sys, numpy, os
import numpy as np
#import dlib
from imutils import face_utils
from imutils.face_utils import FaceAligner


#name = input("Enter name of person:")



def getname(name):
	path = 'datasets'
	directory = os.path.join(path, name)
	print(directory)
	if not os.path.exists(directory):
		os.makedirs(directory, exist_ok = 'True')

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
        
def execute(name):
	face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_alt.xml')
	path = 'datasets'
	face_size=224
	directory = os.path.join(path, name)
	number_of_images = 0
	MAX_NUMBER_OF_IMAGES = 300
	count = 0
	video_capture = cv2.VideoCapture(0)
	
	while number_of_images < MAX_NUMBER_OF_IMAGES:
		ret, frame = video_capture.read()

		if ret:
                	#frame_counter = frame_counter + 1
                	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                	faces = face_cascade.detectMultiScale(gray,scaleFactor=1.2,minNeighbors=10,minSize=(64, 64))
                	# only keep the biggest face as the main subject
                	face = None
                	if len(faces) > 1:  # Get the largest face as main face
                    		face = max(faces, key=lambda rectangle: (rectangle[2] * rectangle[3]))  # area = w * h
                	elif len(faces) == 1:
                    		face = faces[0]
                	if face is not None:
                    		face_img, cropped = crop_face(frame, face, margin=40, size=face_size)
                    		(x, y, w, h) = cropped
                    		cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 200, 0), 2)
                    		cv2.imwrite(os.path.join(directory, str(name+str(number_of_images)+'.jpg')), face_img)
                    		number_of_images += 1
		cv2.imshow('Video', frame)
		if cv2.waitKey(5) == 27:  # ESC key press
			break
	video_capture.release()
	cv2.destroyAllWindows()
