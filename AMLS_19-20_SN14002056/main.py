from A1.A1 import A1
from Datasets.DataPreprocessing import DataPreprocessing, Rgb2Grayscale, HogTransform, PCATransform
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
# # Validation set is produced when K-fold cross validation takes place during model training phase

gender_data_train, gender_data_test = celeb_data.split_train_test(gender)
# emotion_data_train, emotion_data_test = celeb_data.split_train_test(emotion)

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


## Training data

A1_start_time = time.time()


gender_data_input_grayed = grayTransform.transform(gender_data_train[0])
gender_data_input_HOGged = hog.transform(gender_data_input_grayed)
x_train_gender_scaled = scaler.fit_transform(gender_data_input_HOGged)

# pca.fit(x_train_gender_scaled)

# x_train_gender_scaled = pca.transform(x_train_gender_scaled)
# x_train_gender_prepared_PCA = x_train_gender_scaled
y_train_gender = gender_data_train[1]


## Testing data


gender_data_test_grayed = grayTransform.transform(gender_data_test[0])
gender_data_test_HOGged = hog.transform(gender_data_test_grayed)
x_test_gender_scaled = scaler.fit_transform(gender_data_test_HOGged)


# x_test_gender_prepared_PCA = pca.transform(x_test_gender_scaled)
# print(x_test_gender_prepared_PCA)
# print(len(x_test_gender_prepared_PCA))
y_test_gender = gender_data_test[1]
x_train = x_train_gender_scaled
x_test = x_test_gender_scaled

# model_A1 = A1(x_train_gender_scaled, y_train_gender, x_test_gender_scaled, y_test_gender, 'linear')
model_A1 = A1(x_train, y_train_gender, x_test, y_test_gender, 'linear')


acc_A1_train = model_A1.pipeline()

# acc_A1_train = model_A1.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)

acc_A1_test = model_A1.prediction()   # Test model based on the test set.

time_taken = time.time() - A1_start_time
time_taken = round(time_taken, 2)

print("A1 took " + str(time_taken) + " seconds to complete ")

# # ======================================================================================================================
# # Task A2

# A2_start_time = time.time()


# model_A2 = A2(args...)
# acc_A2_train = model_A2.train(args...)
# acc_A2_test = model_A2.test(args...)
# Clean up memory/GPU etc...

# print("A2 took %s seconds to complete " % (round(time.time() - A2_start_time), 2))


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
