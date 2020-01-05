from A1.TaskA import TaskA
from DataPreprocessing import DataPreprocessing


path_to_dir = '/home/fsaxena/amls/AMLSassignment19_20/AMLS_19-20_SN14002056'
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


celeb_data = DataPreprocessing(path_to_dir, celeb)
cartoon_data = DataPreprocessing(path_to_dir, cartoon)

# # ======================================================================================================================
# # Data preprocessing

gender_data_train, gender_data_val, gender_data_test = celeb_data.split_train_val_test(gender)
emotion_data_train, emotion_data_val, emotion_data_test = celeb_data.split_train_val_test(emotion)

cartoon_eye_train, cartoon_eye_val, cartoon_eye_test =  cartoon_data.split_train_val_test(eye)
cartoon_face_train, cartoon_face_val, cartoon_face_test =  cartoon_data.split_train_val_test(face)


print(gender_data_train[1])
print(emotion_data_train[1])

print(cartoon_eye_train[1])
print(cartoon_face_train[1])



# # ======================================================================================================================
# # Task A1
# model_A1 = A1(args...)                 # Build model object.
# acc_A1_train = model_A1.train(args...) # Train model based on the training set (you should fine-tune your model based on validation set.)
# acc_A1_test = model_A1.test(args...)   # Test model based on the test set.
# Clean up memory/GPU etc...             # Some code to free memory if necessary.


# # ======================================================================================================================
# # Task A2
# model_A2 = A2(args...)
# acc_A2_train = model_A2.train(args...)
# acc_A2_test = model_A2.test(args...)
# Clean up memory/GPU etc...


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

# If you are not able to finish a task, fill the corresponding variable with 'TBD'. For example:
# acc_A1_train = 'TBD'
# acc_A1_test = 'TBD'