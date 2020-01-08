from A1.A1 import A1
from A2.A2 import A2
from Datasets.DataPreprocessing import DataPreprocessing, Rgb2Grayscale, HogTransform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time

import numpy as np



# # ======================================================================================================================
# # Please change this path to the relevant path you're running this code from 

path_to_dir = '/home/fsaxena/amls/AMLSassignment19_20/AMLS_19-20_SN14002056'

# # ======================================================================================================================
# # Constants

celeb = 'celeba'
cartoon = 'cartoon_set'
taskList = [
            'gender',
            'smiling',
            'face_shape',
            'eye_color'
            ]

gender = taskList[0]
emotion = taskList[1]
face = taskList[2]
eye = taskList[3]

# # ======================================================================================================================
# # Loading all the datasets

celeb_data = DataPreprocessing(path_to_dir, celeb)
cartoon_data = DataPreprocessing(path_to_dir, cartoon)


# # ======================================================================================================================
# # Splitting the datasets into training and test sets
# # Validation set is produced when K-fold cross validation takes place during model testing phase
# # Please see the respective classes A1/A2/B1/B2 in the train() function for when validation sets are produced

# gender_data_train, gender_data_test = celeb_data.split_train_test(gender)
emotion_data_train, emotion_data_test = celeb_data.split_train_test(emotion)

# cartoon_eye_train, cartoon_eye_test =  cartoon_data.split_train_test(eye)
# cartoon_face_train, cartoon_face_test =  cartoon_data.split_train_test(face)


# # ======================================================================================================================
# # Data preprocessing

grayTransform = Rgb2Grayscale()
hog = HogTransform()
scaler = StandardScaler()
pca = PCA(.95)

# # ======================================================================================================================
# Task A1

# # Time marker to start timing the model computation
# A1_start_time = time.time()

# # Training data
# x_train = gender_data_train[0]
# y_train = gender_data_train[1]


# # Testing data
# x_test = gender_data_test[0]
# y_test = gender_data_test[1]


# model_A1 = A1(x_train, y_train, x_test, y_test)

# acc_A1_test = model_A1.prediction()

# time_taken = time.time() - A1_start_time
# time_taken = round(time_taken, 2)

# print("A1 took " + str(time_taken) + " seconds to complete ")

# # ======================================================================================================================
# # Task A2

A2_start_time = time.time()

# Training data
x_train = emotion_data_train[0]
y_train = emotion_data_train[1]


# Testing data
x_test = emotion_data_test[0]
y_test = emotion_data_test[1]

model_A2 = A2(x_train, y_train, x_test, y_test)

acc_A2_train = model_A2.train()
acc_A2_test = model_A2.prediction()

time_taken = time.time() - A2_start_time
time_taken = round(time_taken, 2)

print("A2 took " + str(time_taken) + " seconds to complete ")


# # ======================================================================================================================
# # Task B1
# model_B1 = B1(args...)
# acc_B1_train = model_B1.train(args...)
# acc_B1_test = model_B1.test(args...)
# Clean up memory/GPU etc...


# # ======================================================================================================================
# # Task B2
# model_B2 = B2(args...)
# acc_B2_train = model_B2.train(args...)
# acc_B2_test = model_B2.test(args...)
# Clean up memory/GPU etc...


# # ======================================================================================================================
# ## Print out your results with following format:
# print('TA1:{},{};TA2:{},{};TB1:{},{};TB2:{},{};'.format(acc_A1_train, acc_A1_test,
#                                                         acc_A2_train, acc_A2_test,
#                                                         acc_B1_train, acc_B1_test,
#                                                         acc_B2_train, acc_B2_test))
