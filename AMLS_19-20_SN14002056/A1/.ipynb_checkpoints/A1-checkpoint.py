import pandas as pd
import sklearn
from scipy import misc
import cv2
import numpy as np
from sklearn.linear_model import SGDClassifier

from Datasets.DataPreprocessing import DataPreprocessing, HogTransform, Rgb2Grayscale

class A1:
    
    def __init__(self, X_train, Y_train, X_test, Y_test, model):
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test
        
        self.model = model

        
    def train(self):
        
        print("-----------------------------------------------------------------------------------------------------")
        print("Training the dataset on " + self.model + " SVM using the Stochastic Gradient Descent optimiser")
        print("-----------------------------------------------------------------------------------------------------")

        if self.model == 'linear':
            sgd_clf = SGDClassifier(random_state=42, loss = 'hinge', max_iter=1000, n_jobs = -1)
        
        if self.model == 'logistic':
            sgd_clf = SGDClassifier(random_state=42, loss = 'log', max_iter=1000, n_jobs = -1)

        return sgd_clf.fit(self.X_train, self.Y_train)
    
    
    
#     def cross_validate()
    
    
    
    def prediction(self):

        
        prediction_model = self.train()
        
        print("-----------------------------------------------------------------------------------------------------")
        print("Completed training. Predicting on test dataset... ")
        print("-----------------------------------------------------------------------------------------------------")

        y_test_pred = prediction_model.predict(self.X_test)
        accuracy_score = 100*np.sum(y_test_pred == self.Y_test)/len(self.Y_test)
        print('Accuracy: ' + str(accuracy_score) + '%')
        

        
        
    
""" save best model as pickle file """