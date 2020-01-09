import pandas as pd
import cv2
import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from skimage import color
from skimage.feature import hog
from skimage.transform import rescale

""" Importing all necessary libraries for data processing """


class DataPreprocessing:
    
    def __init__(self, path, dataset):
        
        """ This class object requires the instantiation of the path to the data and the specific dataset. We recreate the object for different assignment tasks. """
        """ Ensures reusability of the code since same class can be used for different tasks """

        self.path = path
        self.dataset = dataset        
    
    def get_raw_dataframe(self, path, dataset):
        
        """ Using a simple pandas read_csv import to extract labels from tab-separated-value dataset """
        
        return pd.read_csv(self.path + '/Datasets/original_dataset_AMLS_19-20/' + self.dataset + '/labels.csv', sep='\t')

    
    def convert_img_to_vec(self, img_file):
        
        """ Utilised OpenCV2 library to read the image file and convert it into RGBA channels """
        
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        return img
    
    
    def convert_image_to_vector_and_resize(self, img_file):
        
        """ Resizing function specifically used for B task since images were 500px by 500px """
        """ And so needed to be resized to much smaller to be able to compute models much more quickly """
        """ Comparison of image quality is written up in the report. """
        """ 35% seemed to be large enough to distinguish key features, but also small enough to not overflow RAM when matrices are computed """
        
        img = cv2.imread(img_file, cv2.IMREAD_UNCHANGED)
        scale_percent = 35 # percent of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        dim = (width, height)
        # resize image
        resized_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
        return resized_img
    
    def df_to_vec(self, path, dataset):
        
        """ This method takes the raw dataframe of the dataset and uses the specific image file to feed into the image converter methods above. """
        
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
            
            """Calling a different image to vector converter function here to make sure we resize the large PNG image"""
            
            print("----------------------------------------------------")
            print("Converting raw images to pixel info in Cartoon dataset...")
            
            for img_name in df['file_name']:
                img_vec = np.array(self.convert_image_to_vector_and_resize(path_to_img_dir + img_name))
                vec_Array.append(img_vec)
        
            return vec_Array

        
    def split_train_test(self, task):
        
        """Taking the datasets that have been computed and then convert them into the respective numpy arrays"""

        vector_array = self.df_to_vec(self.path, self.dataset)
        df = self.get_raw_dataframe(self.path, self.dataset)

        x_dataset = vector_array
        y_dataset = np.array(df[str(task)])

        print("Fetching data for the " + task + " task")
        print("Length of input pixel list: " + str(len(x_dataset)))
        print("Length of output labels list: " + str(len(y_dataset)))
        print("-----------------------DONE-------------------------")
        
        """Split the data into testing and training sets"""
        """As mentioned, validation set is computed during cross-validation, GridSearch tasks"""

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

    
    def get_raw_test_dataframe(self, path, dataset):
        
        """ Using a simple pandas read_csv import to extract labels from tab-separated-value dataset """
        
        return pd.read_csv(self.path + '/Datasets/test_dataset_AMLS_19-20/' + self.dataset + '/labels.csv', sep='\t')
    
    
    
    def test_df_to_vec(self, path, dataset):
        
        """ This method takes the raw dataframe of the dataset and uses the specific image file to feed into the image converter methods above. """

        
        vec_Array = []
        
        path_to_img_dir = self.path+'/Datasets/test_dataset_AMLS_19-20/'+self.dataset+'/img/'
        
        dftest = self.get_raw_test_dataframe(path, dataset)

        if self.dataset == 'celeba':
            print("----------------------------------------------------")
            print("Converting raw images to pixel info in Celebrity dataset...")

            for img_name in dftest['img_name']:
                img_vec = np.array(self.convert_img_to_vec(path_to_img_dir + img_name))
                vec_Array.append(img_vec)
                
            return vec_Array
        
        if self.dataset == 'cartoon_set':
            
            """Calling a different image to vector converter function here to make sure we resize the large PNG image"""
            
            print("----------------------------------------------------")
            print("Converting raw images to pixel info in Cartoon dataset...")
            
            for img_name in dftest['file_name']:
                img_vec = np.array(self.convert_image_to_vector_and_resize(path_to_img_dir + img_name))
                vec_Array.append(img_vec)
        
            return vec_Array
    
    def unseen_testset(self, task):
        
        """ Creating a separate function to handle unseen test data """
        
        test_vector_array = self.test_df_to_vec(self.path, self.dataset)
        test_df = self.get_raw_test_dataframe(self.path, self.dataset)
        
        print("----------------------------------------------------")
        print("Fetching unseen test data for the " + task + " task")
        print("----------------------------------------------------")

        x_unseen = test_vector_array
        y_unseen = np.array(test_df[str(task)])
        
        unseen_data = (x_unseen, y_unseen)
        print("----------------------------------------------------")
        print("Completed ")
        print("----------------------------------------------------")
        return unseen_data
    
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