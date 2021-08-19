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
	path = 'data/files/'+name
	fpath= 'data/chunks/'+name
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
    """Convert text file to a grayscale image with black characters on a white background.
    arguments:
    text_path - the content of this file will be converted to an image
    font_path - path to a font file (for example impact.ttf)
    """
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
    # crop the text
    #c_box = PIL.ImageOps.invert(image).getbbox()
    #image = image.crop(c_box)
    #size = 128, 128
    #im = Image.open(image)
    #im_resized = image.resize(size, Image.ANTIALIAS)
    return image
				
def image(name):
	fpath= 'data/chunks/'+name
	mypath= 'data/images/'+name
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
	fpath = 'data/images/'+name
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


def getfile(name):
	chunks(name)
	image(name)
	vgg = cnn_vgg(name)
	path = 'data/cnn-lstm-1024_256_try_E10_L20_BS10.h5'
	vgg = cnn_vgg(name)
	feature_vector = np.asarray(vgg)
	feature_vector_new = np.reshape(feature_vector, (feature_vector.shape[0]*feature_vector.shape[1], feature_vector.shape[2]))
	feature_vector_new1 = np.expand_dims(feature_vector_new, axis=0)
	mode = load_model(path)
	with sess.as_default():
		with sess.graph.as_default():
			k = mode.predict(feature_vector_new1)
			print(k[0])
	(m,ind) = max((v,ind) for ind,v in enumerate(k[0]))
	print(ind)
	i=0
	authors = ['0opslab','brianway','johnno1962','newweb','seadroid','andengineexamples',
           'chao420456','lemire','pacman','Viscent','applewjg','chweixin','mark-watson',
           'quyi','waimai','BrandConstantin','dlna_framework','mthli','sample-server','weather']
	auth_map={}
	for a in authors:
		auth_map[a] = i
		i+=1
	for key, value in auth_map.items():
		if ind == value:
			final = key
	print(final) 
	with open('data/cache.csv', 'a') as csvfile:
		csvwriter = csv.writer(csvfile) 
		csvwriter.writerow([name,final])
	return final
