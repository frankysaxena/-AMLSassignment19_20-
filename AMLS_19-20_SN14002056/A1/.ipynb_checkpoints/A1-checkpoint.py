import pandas as pd
from scipy import misc
import cv2
import numpy as np

from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV


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
    
    
    
    def prediction(self):
        
        prediction_model = self.pipeline()
        
        print("-----------------------------------------------------------------------------------------------------")
        print("Completed training. Predicting on test dataset... ")
        print("-----------------------------------------------------------------------------------------------------")

        y_test_pred = prediction_model.predict(self.X_test)
        accuracy_score = 100*np.sum(y_test_pred == self.Y_test)/len(self.Y_test)
        print('Accuracy: ' + str(accuracy_score) + '%')
        
    
    def pipeline(self):
                
        A1Pipeline = Pipeline([
#             ('toGray', Rgb2Grayscale()),
#             ('toHog', HogTransform()),
#             ('toScale', StandardScaler()),
            ('toPCA', PCA(0.95)),
            ('classify', SGDClassifier(random_state=42, loss = 'hinge', max_iter=1000, n_jobs = -1, tol=1e-3))
        ])
                
        param_grid = [
            {
                'classify': [
                    SGDClassifier(random_state=42, max_iter=1000, tol=1e-1),
                    SGDClassifier(random_state=42, max_iter=1000, tol=1e-2),
                    SGDClassifier(random_state=42, max_iter=1000, tol=1e-3),
                    SGDClassifier(random_state=42, max_iter=1000, tol=1e-4),
                    SGDClassifier(random_state=42, max_iter=1000, tol=1e-5),
                    svm.SVC(kernel='rbf', C=1),
                    svm.SVC(kernel='rbf', C=10),
                    svm.SVC(kernel='rbf', C=100)
                 ]
            }
        ]
        
        
        grid_search = GridSearchCV(A1Pipeline,
                           param_grid,
                           cv=5,
                           n_jobs=-1,
                           scoring='accuracy',
                           verbose=10,
                           return_train_score=True)        
        
        grid_res = grid_search.fit(self.X_train, self.Y_train)
        
        best_score = round(grid_res.best_score_, 2)
        best_score = grid_res.best_score_ 
        
        print("-----------------------------------------------------------------------------------------------------")
        print("Completed training pipeline with cross-fold validation")
        print("-----------------------------------------------------------------------------------------------------")
        print("Best estimator: ")
        print(grid_res.best_estimator_)
        print("-----------------------------------------------------------------------------------------------------")
        print("Best parameters: ")
        print(grid_res.best_params_)
        print("-----------------------------------------------------------------------------------------------------")
        print("Best score achieved: ")
        print(best_score)
                         
        return grid_res

