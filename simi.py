import os
from itertools import zip_longest
import PIL
import PIL.Image
import PIL.ImageFont
import PIL.ImageOps
import PIL.ImageDraw
import shutil, random
import numpy as np
import csv
import ipywidgets
import traitlets

import tensorflow as tf
sess = tf.Session()

from keras import backend as K
K.set_session(sess)

init_op = tf.global_variables_initializer()
sess.run(init_op)

from keras.models import Model
from keras.models import load_model
	
PIXEL_ON = 0  # PIL color to use for "on"
PIXEL_OFF = 255  # PIL color to use for "off"


	
def grouper(n, iterable, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper(3, 'ABCDEFG', 'x') --> ABC DEF Gxx
    args = [iter(iterable)] * n
    print(args)
    return zip_longest(fillvalue=fillvalue, *args)
    

def chunks(name):
	path = 'data/simi/files/'+name
	fpath= 'data/simi/chunks/'+name
	if not os.path.isdir(fpath):
		os.makedirs(fpath)
	n=20
	i=0
	print('In file')
	with open(path,'r', encoding="utf-8", errors='ignore') as f:
		for k,g in enumerate(grouper(n, f, fillvalue=''), 1):
			i+=2
			with open(os.path.join(fpath, '{0}.txt').format(i * n), 'w') as fout:
				fout.writelines(g)

def text_image(text_path, font_path='data/cour.ttf'):
    grayscale = 'L'
    with open(text_path) as text_file:  # can throw FileNotFoundError
        lines = tuple(text_file.readlines())

    large_font = 20  # get better resolution with larger size
    font_path = font_path or 'cour.ttf'  # Courier New. works in windows. linux may need more explicit path
    try:
        font = PIL.ImageFont.truetype(font_path, size=large_font)
    except IOError:
        font = PIL.ImageFont.load_default()
        print('Could not use chosen font. Using default.')

    # make the background image based on the combination of font and lines
    pt2px = lambda pt: int(round(pt * 96.0 / 72))  # convert points to pixels
    max_width_line = max(lines, key=lambda s: font.getsize(s)[0])
    #max height is adjusted down because it's too large visually for spacing
    test_string = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    max_height = pt2px(font.getsize(test_string)[1])
    max_width = pt2px(font.getsize(max_width_line)[0])
    height = max_height * len(lines)  # perfect or a little oversized
    width = int(round(max_width + 40))  # a little oversized
    image = PIL.Image.new(grayscale, (width, height), color=PIXEL_OFF)
    draw = PIL.ImageDraw.Draw(image)

    # draw each line of text
    vertical_position = 5
    horizontal_position = 5
    line_spacing = int(round(max_height * 0.8))  # reduced spacing seems better
    for line in lines:
        draw.text((horizontal_position, vertical_position),
                  line, fill=PIXEL_ON, font=font)
        vertical_position += line_spacing
   
    return image
				
def image(name):
	fpath= 'data/simi/chunks/'+name
	mypath= 'data/simi/images/'+name
	if not os.path.isdir(mypath):
		os.makedirs(mypath)
	dir_path=os.listdir(fpath)
	for eachFile in dir_path:
		if eachFile.endswith(".txt"):
			f_path=fpath+'/'+eachFile 
			print(f_path)
			image = text_image(f_path)
			image.save('{0}.png'.format(os.path.join(mypath,eachFile)))

def extract_feature_vector(input_image,vgg_model):
	layer_name = 'dense_8'
	intermediate_layer_model = Model(inputs=vgg_model.input,outputs=vgg_model.get_layer(layer_name).output)
	with sess.as_default():
		with sess.graph.as_default():
			intermediate_output = intermediate_layer_model.predict(input_image)
	return intermediate_output
				
def cnn_vgg(name):
	from keras.preprocessing import image
	from keras.applications.vgg16 import VGG16
	from keras.applications.vgg16 import preprocess_input
	from keras import models
	from keras import layers
	from keras import optimizers
	
	vgg_model = load_model('data/last4unfreezed_E20_L20.h5')
	vgg_model.summary()
	fpath = 'data/simi/images/'+name
	dir_path=os.listdir(fpath)
	feature_list = []
	for eachFile in dir_path:
		f_path=fpath+'/'+eachFile
		img = image.load_img(f_path, target_size=(256, 256))
		img_data = np.expand_dims(img, axis=0)
		feature = extract_feature_vector(img_data,vgg_model)
		feature_list.append(feature)
	
	print(feature_list[0].shape)
	print(len(feature_list))	
	return feature_list


def calculate(f1,f2):
	chunks(f1)
	chunks(f2)
	image(f1)
	image(f2)
	vgg1 = cnn_vgg(f1)
	vgg2 = cnn_vgg(f2)
	feature_vector1 = np.asarray(vgg1)
	feature_vector_new1 = np.reshape(feature_vector1, (feature_vector1.shape[0]*feature_vector1.shape[1], feature_vector1.shape[2]))
	feature_vector_new11 = np.expand_dims(feature_vector_new1, axis=0)
	feature_vector2 = np.asarray(vgg2)
	feature_vector_new2 = np.reshape(feature_vector2, (feature_vector2.shape[0]*feature_vector2.shape[1],feature_vector2.shape[2]))
	feature_vector_new12 = np.expand_dims(feature_vector_new2, axis=0)
	lstm_model = load_model('data/cnn_lstm_L256_Final.h5')
	layer_name = 'lstm_6'
	intermediate_layer_model = Model(inputs=lstm_model.input,outputs=lstm_model.get_layer(layer_name).output)
	with sess.as_default():
		with sess.graph.as_default():
			k1 = intermediate_layer_model.predict(feature_vector_new11)
			print(k1)
	with sess.as_default():
		with sess.graph.as_default():
			k2 = intermediate_layer_model.predict(feature_vector_new12)
			print(k2)
	cal = np.linalg.norm(k1-k2)
	return cal
