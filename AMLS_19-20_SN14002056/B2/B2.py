import pandas as pd
from scipy import misc
import cv2
import numpy as np
import pickle


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import plot_confusion_matrix
from matplotlib import pyplot as plt
import parfit.parfit as pf

from Datasets.DataPreprocessing import DataPreprocessing, HogTransform, Rgb2Grayscale

class B2:
    
    
    def __init__(self, X_train, Y_train, X_test, Y_test):
        
        """ Initialise the class with the input training data set and the testing data set for the class instance """
        
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
    
    def prediction(self, x_unseen, y_unseen):
        
        """ Prediction function to use the saved pickle model from the train function. Tests on the test dataset """

        with open('B2/best_model_B2.sav', 'rb') as f:
            loaded_model = pickle.load(f)

        print("-----------------------------------------------------------------------------------------------------")
        print(" Predicting on test dataset... ")
        print("-----------------------------------------------------------------------------------------------------")

        X_test = np.reshape(x_unseen, (2500, 122500))
        y_test_pred = loaded_model.predict(X_test)
        test_accuracy = 100*np.sum(y_test_pred == y_unseen)/len(y_unseen)
        test_accuracy = round(test_accuracy, 1)
        print('Test accuracy: ' + str(test_accuracy) + '%')
        return test_accuracy
        
    
    def train(self):
        
        """ Training pipeline starting at data preprocessing stage to take the raw data and transform it.  """
        """ After transformation, we move to the classify stage where the respective classifiers are used to train the data sequentially.  """

        
        X_train = np.array(self.X_train)
        
        """ reshaping it to a much smaller image size for faster computation """
        X_train = np.reshape(X_train, (8000, 122500))
        X_test = np.reshape(self.X_test, (2000, 122500))
        

        """ Grid for hyperparameter tuning of respective classifiers """

        grid = {
            'n_estimators': [5, 6, 7, 8, 9, 10], # number of trees in the random forest
            'criterion': ['entropy']
            }
#         grid = {
#             'depth': [5, 6, 7, 8, 9, 10], # depth of trees
#             }
        grid_search = GridSearchCV(RandomForestClassifier(), grid, cv=2, n_jobs=-1, scoring='accuracy', verbose=50, return_train_score=True)    
#         grid_search = GridSearchCV(DecisionTreeClassifier(), grid, cv=2, n_jobs=-1, scoring='accuracy', verbose=50, return_train_score=True)    

        """ Again, save the best model and train this last time on the training set for accuracy score """

        grid_res = grid_search.fit(X_train, self.Y_train)
        train_accuracy = float(grid_res.best_score_) * 100
        train_accuracy = round(train_accuracy, 1)
        print("Train accuracy: " + str(train_accuracy) + "%")

        print("-----------------------------------------------------------------------------------------------------")
        print("Completed training pipeline with cross-fold validation")
        print("-----------------------------------------------------------------------------------------------------")
        print("Best estimator: ")
        print(grid_res.best_estimator_)
        print("-----------------------------------------------------------------------------------------------------")
        print("Best parameters: ")
        print(grid_res.best_params_)
        print("-----------------------------------------------------------------------------------------------------")

        
        """ Save the best model into a file to load into prediction function """        
        with open('B2/best_model_B2.sav', 'wb') as f:
            pickle.dump(grid_res, f)
        
        return train_accuracy