#OS
import os
import random as rand

# TensorFlow and tf.keras
import tensorflow as tf
import numpy as np
import cv2

#from tensorflow.keras import layers
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

#Modele 1 :

model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(32, 3, activation='relu',input_shape=(240, 320, 3)),
  tf.keras.layers.MaxPool2D((2,2)),
  tf.keras.layers.Conv2D(32, 3,activation='relu'),
  tf.keras.layers.MaxPool2D((2,2)),
  tf.keras.layers.Conv2D(32, 3,activation='relu'),
  tf.keras.layers.MaxPool2D((2,2)),
  tf.keras.layers.Conv2D(32, 3,activation='relu'),
  tf.keras.layers.MaxPool2D((2,2)),
  tf.keras.layers.Conv2D(32, 3,activation='relu'),
  tf.keras.layers.MaxPool2D((2,2)),                     
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(4, activation='softmax')
])

'''
model = tf.keras.models.Sequential ([
    tf.keras.layers.Conv2D(32,3,3, activation = 'relu', input_shape = (240, 320, 3)),
    tf.keras.layers.Conv2D(32,3,3, activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size = (2,2)),
    tf.keras.layers.Dropout(0.5),
    
    tf.keras.layers.Conv2D(64,3,3, activation = 'relu'),
    tf.keras.layers.MaxPool2D(pool_size = (2,2)),
    tf.keras.layers.Dropout(0.5), 
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation = 'relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(4, activation = 'softmax')
])
'''

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

#Jeux de test :
training_x = []
training_y = []
testing_x = []
testing_y = []

NO_BALL      = 250
NUMBER_RED   = 250
NUMBER_GREEN = 250
NUMBER_BLUE  = 250

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

#Training
model.fit(training_x, training_y, epochs=6, verbose=2)

#Evaluate
results = model.evaluate(testing_x, testing_y, verbose=2)
print("test loss, test acc:", results)

#model save in external file
print("Voulez-vous sauvegarder votre modele ? (y/n) : ")
rep = input()
if(rep == 'y') :
	model.save('./Saved_Model/myModel')




