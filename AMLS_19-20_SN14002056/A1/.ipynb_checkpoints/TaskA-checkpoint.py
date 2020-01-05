import pandas as pd
import sklearn
from scipy import misc
import cv2
import numpy as np 


class TaskA:
    
    path = ''
    assignment = ''
    
    def __init__(self):
        pass
#         self.path = path
#         self.assignment = assignment
#         self.dataset = ''
    
    
    def get_data(self, path, assignment):
        return pd.read_csv(self.path + '/Datasets/original_dataset_AMLS_19-20/' + self.assignment + '/labels.csv', sep='\t')
    
#     def data_prepare(self, dataset):
        
#     def model():
        