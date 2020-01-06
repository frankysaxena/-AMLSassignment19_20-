from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin

from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA

from sklearn.model_selection import cross_val_predict
from sklearn.preprocessing import StandardScaler
from skimage import color

from skimage.feature import hog
from skimage.transform import rescale


class DataPreprocessing:
    
    def __init__(self, path, dataset):
        self.path = path
        self.dataset = dataset        
    
    def get_raw_dataframe(self, path, dataset):
        return pd.read_csv(self.path + '/Datasets/original_dataset_AMLS_19-20/' + self.dataset + '/labels.csv', sep='\t')

    def convert_img_to_vec(self, img_file):
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        return img
    
    def df_to_vec(self, path, dataset):
        vec_Array = []
        path_to_img_dir = self.path+'/Datasets/original_dataset_AMLS_19-20/'+self.dataset+'/img/'
        
        df = self.get_raw_dataframe(path, dataset)

        if self.dataset == 'celeba':
            print("----------------------------------------------------")
            print("Converting raw images to pixel info in Celebrity dataset...")

            for img_name in df['img_name']:
                img_vec = np.array(self.convert_img_to_vec(path_to_img_dir + img_name))
                vec_Array.append(img_vec)
                
            return vec_Array
        
        if self.dataset == 'cartoon_set':
            print("----------------------------------------------------")
            print("Converting raw images to pixel info in Cartoon dataset...")
            
            for img_name in df['file_name']:
                img_vec = np.array(self.convert_img_to_vec(path_to_img_dir + img_name))
                vec_Array.append(img_vec)
        
            return vec_Array

        
    def split_train_test(self, task):
        
        vector_array = self.df_to_vec(self.path, self.dataset)
        df = self.get_raw_dataframe(self.path, self.dataset)

        x_dataset = vector_array
        y_dataset = np.array(df[str(task)])

        print("Fetching data for the " + task + " task")
        print("Length of input pixel list: " + str(len(x_dataset)))
        print("Length of output labels list: " + str(len(y_dataset)))
        print("-----------------------DONE-------------------------")

        
        x, x_test, y, y_test = train_test_split(
            x_dataset,
            y_dataset,
            test_size=0.2,
            shuffle=True,
            random_state=42,
        )
        
        train_data = (x, y)
        test_data = (x_test, y_test)
        
        return train_data, test_data

    
class Rgb2Grayscale(BaseEstimator, TransformerMixin):
    
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        """returns itself"""
        return self
    
    def transform(self, rgb_array, y=None):
        
        """convert the RGB channel pixel features into single grayscale channel"""
        
        return np.array([color.rgb2gray(img) for img in rgb_array])


class HogTransform(BaseEstimator, TransformerMixin):
    
    def __init__(self, y=None, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(3, 3), block_norm='L2-Hys'):
        
        """ HOG transform as one of the choices for feature extraction. Takes a grayscale image vector as the input """
        self.y = y
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.block_norm = block_norm
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, gray_vector, y=None):
        
        print("-----------------------------------------------")
        print("Transforming each image to extract HOG features")
        print("-----------------------------------------------")

        def local_hog(gray_vector):
            return hog(gray_vector,
                       orientations=self.orientations,
                       pixels_per_cell=self.pixels_per_cell,
                       cells_per_block=self.cells_per_block,
                       block_norm=self.block_norm)

        try: # parallel
            return np.array([local_hog(img) for img in gray_vector])
        except:
            return np.array([local_hog(img) for img in gray_vector])
        
    