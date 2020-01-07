import pandas as pd
from scipy import misc
import cv2
import numpy as np
import pickle


from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot as plt
import parfit.parfit as pf

from Datasets.DataPreprocessing import DataPreprocessing, HogTransform, Rgb2Grayscale

class A2:
    
    
    def __init__(self, X_train, Y_train, X_test, Y_test):
        
        """ Initialise the class with the input training data set and the testing data set for the class instance """
        
        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

    
    def train_specific(self, model):
        
        grayTransform = Rgb2Grayscale()
        hog = HogTransform()
        scaler = StandardScaler()
        pca = PCA(.95)
        
        """ The train_specific function was written to gather results/plots for specific models to easily compare against each other """
        
        gender_data_input_grayed = grayTransform.transform(self.X_train)
        gender_data_input_HOGged = hog.transform(gender_data_input_grayed)
        x_train_gender_scaled = scaler.fit_transform(gender_data_input_HOGged)

        pca.fit(x_train_gender_scaled)

        x_train_gender_scaled = pca.transform(x_train_gender_scaled)
        x_train_gender_prepared_PCA = x_train_gender_scaled
        
        gender_data_test_grayed = grayTransform.transform(self.X_test)
        gender_data_test_HOGged = hog.transform(gender_data_test_grayed)
        x_test_gender_scaled = scaler.fit_transform(gender_data_test_HOGged)


        x_test_gender_prepared_PCA = pca.transform(x_test_gender_scaled)
        
        print("-----------------------------------------------------------------------------------------------------")
        print("Training the dataset on " + model + " SVM using the Stochastic Gradient Descent optimiser")
        print("-----------------------------------------------------------------------------------------------------")

        if model == 'linear':
            sgd_clf = SGDClassifier(random_state=42, loss = 'hinge', max_iter=1000, n_jobs = -1)
            sgd_clf.fit(x_train_gender_prepared_PCA, self.Y_train)

            return sgd_clf
        
        if model == 'logistic':
            sgd_clf = SGDClassifier(random_state=42, tol=1e-1, loss = 'log', max_iter=1000, n_jobs = -1)
        
            sgd_clf.fit(x_train_gender_prepared_PCA, self.Y_train)
            lr_probs = sgd_clf.predict_proba(self.X_test)
            lr_probs = lr_probs[:, 1]
            lr_auc = roc_auc_score(x_test_gender_prepared_PCA, lr_probs)
            print(model + ': ROC AUC=%.3f' % (lr_auc))
            lr_fpr, lr_tpr, _ = roc_curve(self.Y_test, lr_probs)
            # plot the roc curve for the model
            plt.plot(lr_fpr, lr_tpr, marker='.', label=model)
            # axis labels
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            # show the legend
            plt.legend()
            plt.title('A1: ROC Curve of ' + model + ' Regression using SGD')
            # show the plot
            with open('A1/' + model + '_test.png', 'wb') as f:
                plt.savefig(f)
    
    
    
    def prediction(self):
        
        """ Prediction function to use the saved pickle model from the train function. Tests on the test dataset """

        with open('A1/best_model_A1.sav', 'rb') as f:
            loaded_model = pickle.load(f)

        print("-----------------------------------------------------------------------------------------------------")
        print("Completed training. Predicting on test dataset... ")
        print("-----------------------------------------------------------------------------------------------------")

        y_test_pred = loaded_model.predict(self.X_test)
        accuracy_score = 100*np.sum(y_test_pred == self.Y_test)/len(self.Y_test)
        print('Accuracy: ' + str(accuracy_score) + '%')
        
    
    def train(self):
        
        """ Training pipeline starting at data preprocessing stage to take the raw data and transform it.  """
        """ After transformation, we move to the classify stage where the respective classifiers are used to train the data sequentially.  """

                
        A1Pipeline = Pipeline([
            ('toGray', Rgb2Grayscale()),
            ('toHog', HogTransform()),
            ('toScale', StandardScaler()),
            ('toPCA', PCA(0.95)),
            ('classify', SGDClassifier(random_state=42, loss = 'log', max_iter=1000, n_jobs = -1, tol=1e-3))
        ])
        
        
        
        """ After transformation, we move to the classify stage where the respective classifiers are used to train the data sequentially.  """

        param_grid = [
            {
                'classify': [
                    SGDClassifier(random_state=42, max_iter=1000, tol=1e-1),
                    SGDClassifier(random_state=42, max_iter=1000, tol=1e-2),
                    SGDClassifier(random_state=42, max_iter=1000, tol=1e-3),
                    svm.SVC(kernel='rbf', C=1, probability=True),
                    svm.SVC(kernel='rbf', C=10),
                    svm.SVC(kernel='rbf', C=100)
                 ]
            }
        ]
        
        
        grid_search = GridSearchCV(A1Pipeline,
                           param_grid,
                           cv=2,
                           n_jobs=-1,
                           scoring='accuracy',
                           verbose=50,
                           return_train_score=True)        
        
        grid_res = grid_search.fit(self.X_train, self.Y_train)
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
        
        """ Save the best model into a file to load into prediction function """
        
        with open('A1/best_model_A1.sav', 'wb') as f:
            pickle.dump(grid_res, f)

        return grid_res
