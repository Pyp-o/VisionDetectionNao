#OS and rand
import os
import random as rand


# TensorFlow and tf.keras
import tensorflow as tf
import numpy as np
import cv2

#from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

#Jeux de test :
training_x = []
training_y = []
testing_x = []
testing_y = []

NO_BALL      = 149
NUMBER_RED   = 161
NUMBER_GREEN = 251
NUMBER_BLUE  = 243

#NO BALL
for i in range(0,NO_BALL) :
	name = './Images/No_ball/img101.png'
	if(i>=0 and i<100) :
		final_img = cv2.normalize(cv2.imread(name), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		training_x.append(final_img)
		training_y.append(0)
	else :
		final_img = cv2.normalize(cv2.imread(name), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		testing_x.append(final_img)
		testing_y.append(0)

#RED BALL
for i in range(0,NUMBER_RED) :
	name = './Images/Balle_rouge/img'+str(i)+'.png'
	if(i>=0 and i<120) :
		final_img = cv2.normalize(cv2.imread(name), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		training_x.append(final_img)
		training_y.append(1)
	else :
		final_img = cv2.normalize(cv2.imread(name), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		testing_x.append(final_img)
		testing_y.append(1)

#GREEN BALL
for i in range(0,NUMBER_GREEN) :
	name = './Images/Balle_verte/img'+str(i)+'.png'
	if(i>=0 and i<120) :
		final_img = cv2.normalize(cv2.imread(name), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		training_x.append(final_img)
		training_y.append(2)
	else :
		final_img = cv2.normalize(cv2.imread(name), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		testing_x.append(final_img)
		testing_y.append(2)

#BLUE BALL
for i in range(0,NUMBER_BLUE) :
	name = './Images/Balle_bleue/img'+str(i)+'.png'
	if(i>=0 and i<120) :
		final_img = cv2.normalize(cv2.imread(name), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		training_x.append(final_img)
		training_y.append(3)
	else :
		final_img = cv2.normalize(cv2.imread(name), None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		testing_x.append(final_img)
		testing_y.append(3)

#Test :
'''
print(training_x[45])
cv2.imshow('test',training_x[45])
cv2.waitKey(0)
'''

#Array convertion
training_x = np.array(training_x)
#print(training_x)
training_y = np.array(training_y)
#print(training_y)

testing_x = np.array(testing_x)
#print(testing_x)
testing_y = np.array(testing_y)
#print(testing_y)

#Suffle le testing set :
shuffle_index=[]
testing_x_shuffled = []
testing_y_shuffled = []

for index in range(len(testing_x)) :
    shuffle_index.append(index)
rand.shuffle(shuffle_index)
#print(shuffle_index)

for i in range(len(testing_x)) :
	testing_x_shuffled.append(testing_x[shuffle_index[i]])
	testing_y_shuffled.append(testing_y[shuffle_index[i]])

testing_x_shuffled = np.array(testing_x_shuffled)
#print(testing_x)
testing_y_shuffled = np.array(testing_y_shuffled)
#print(testing_y)

#new model set identical as the old one
model = tf.keras.models.load_model('./Saved_Model/myModel')

#Testing
predictions = model.predict(testing_x)
predictions = np.array(predictions)

predictions_sh = model.predict(testing_x_shuffled)
predictions_sh = np.array(predictions_sh)

class_names = ['Nothing','Red Ball','Green Ball','Blue Ball']

nb_v = 0
nb_f = 0

for i in range(len(predictions)) :
	
	pre  = class_names[np.argmax(predictions[i])]
	real = class_names[testing_y[i]]

	pre_2 = class_names[np.argmax(predictions_sh[i])]
	real_2 = class_names[testing_y_shuffled[i]]

	if(pre_2==real_2) : 
		nb_v = nb_v +1
	else :
		nb_f = nb_f +1

	print(pre_2,' -- ',real_2,'\n')

print('Nombre de predictions vraies : ',nb_v,'\nNombre de predictions fausses : ',nb_f)	

