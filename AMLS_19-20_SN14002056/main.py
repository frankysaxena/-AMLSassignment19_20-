from A1.A1 import A1
from A2.A2 import A2
from B1.B1 import B1
from B2.B2 import B2


from Datasets.DataPreprocessing import DataPreprocessing, Rgb2Grayscale, HogTransform
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import time
import gc
import cv2

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

print("------------------TASK A1 DATASET-------------------")
gender_data_train, gender_data_test = celeb_data.split_train_test(gender)
gender_test = celeb_data.unseen_testset(gender)

print("------------------TASK A2 DATASET-------------------")
emotion_data_train, emotion_data_test = celeb_data.split_train_test(emotion)
emotion_test = celeb_data.unseen_testset(emotion)

print("------------------TASK B1 DATASET-------------------")
cartoon_face_train, cartoon_face_test =  cartoon_data.split_train_test(face)
face_test = cartoon_data.unseen_testset(face)

print("------------------TASK B2 DATASET-------------------")
cartoon_eye_train, cartoon_eye_test =  cartoon_data.split_train_test(eye)
eye_test = cartoon_data.unseen_testset(eye)


# # ======================================================================================================================
# # Data preprocessing

grayTransform = Rgb2Grayscale()
hog = HogTransform()
scaler = StandardScaler()
pca = PCA(.95)

# # ======================================================================================================================
# Task A1

# Time marker to start timing the model computation
A1_start_time = time.time()

# Training data
x_train = gender_data_train[0]
y_train = gender_data_train[1]


# Testing data
x_test = gender_data_test[0]
y_test = gender_data_test[1]

# Unseen testing data
x_unseen_test = gender_test[0]
y_unseen_test = gender_test[1]

model_A1 = A1(x_train, y_train, x_test, y_test)

acc_A1_train = model_A1.train()
acc_A1_test = model_A1.prediction(x_unseen_test, y_unseen_test) # Predict on separate unseen test data

time_taken = time.time() - A1_start_time
time_taken = round(time_taken, 2)

print("A1 took " + str(time_taken) + " seconds to complete ")

print("-----------------------------------------------")
print("Cleaning memory before next task... ")

gc.collect() # Clean up memory

print("Cleared!")


# # ======================================================================================================================
# # Task A2

A2_start_time = time.time()

# Training data
x_train = emotion_data_train[0]
y_train = emotion_data_train[1]


# Testing data
x_test = emotion_data_test[0]
y_test = emotion_data_test[1]

# Unseen testing data
x_unseen_test = emotion_test[0]
y_unseen_test = emotion_test[1]

model_A2 = A2(x_train, y_train, x_test, y_test)

acc_A2_train = model_A2.train()
acc_A2_test = model_A2.prediction(x_unseen_test, y_unseen_test) # Predict on separate unseen test data

time_taken = time.time() - A2_start_time
time_taken = round(time_taken, 2)

print("A2 took " + str(time_taken) + " seconds to complete ")

print("-----------------------------------------------")
print("Cleaning memory before next task... ")

gc.collect() # Clean up memory

print("Cleared!")
# # ======================================================================================================================
# # Task B1

B1_start_time = time.time()


# Training data
x_train = cartoon_face_train[0]
y_train = cartoon_face_train[1]


# Testing data
x_test = cartoon_face_test[0]
y_test = cartoon_face_test[1]

# Unseen testing data
x_unseen_test = face_test[0]
y_unseen_test = face_test[1]

model_B1 = B1(x_train, y_train, x_test, y_test)

acc_B1_train = model_B1.train()
acc_B1_test = model_B1.prediction(x_unseen_test, y_unseen_test) # Predict on separate unseen test data

time_taken = time.time() - B1_start_time
time_taken = round(time_taken, 2)

print("B1 took " + str(time_taken) + " seconds to complete ")


print("-----------------------------------------------")
print("Cleaning memory before next task... ")

gc.collect() # Clean up memory

print("Cleared!")
# # ======================================================================================================================
# # Task B2


B2_start_time = time.time()


# Training data
x_train = cartoon_eye_train[0]
y_train = cartoon_eye_train[1]


# Testing data
x_test = cartoon_eye_test[0]
y_test = cartoon_eye_test[1]

# Unseen testing data
x_unseen_test = eye_test[0]
y_unseen_test = eye_test[1]

model_B2 = B2(x_train, y_train, x_test, y_test)

acc_B2_train = model_B2.train()
acc_B2_test = model_B2.prediction(x_unseen_test, y_unseen_test) # Predict on separate unseen test data

time_taken = time.time() - B2_start_time
time_taken = round(time_taken, 2)

print("B2 took " + str(time_taken) + " seconds to complete ")

# # ======================================================================================================================
# ## Print out your results with following format:
print("-----------------------------RESULTS------------------------------------")

data = [['Task', 'Train Accuracy (%)', 'Test Accuracy (%)'], 
        ['A1', str(acc_A1_train), str(acc_A1_test)], 
        ['A2', str(acc_A2_train), str(acc_A2_test)], 
        ['B1', str(acc_B1_train), str(acc_B1_test)], 
        ['B2', str(acc_B2_train), str(acc_B2_test)]]

col_width = max(len(word) for row in data for word in row) + 2  # padding
for row in data:
    print("".join(word.ljust(col_width) for word in row))