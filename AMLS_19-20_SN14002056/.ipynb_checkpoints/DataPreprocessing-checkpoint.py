from sklearn.model_selection import train_test_split
import pandas as pd
import cv2
import numpy as np


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
        
    def split_train_val_test(self, task):
        
        df = self.get_raw_dataframe(self.path, self.dataset)
        vector_array = self.df_to_vec(self.path, self.dataset)

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
            
        x_train, x_val, y_train, y_val = train_test_split(
            x,
            y,
            test_size=0.25,
            shuffle=True,
            random_state=42,
        )
        
        train_data = (x_train, y_train)
        val_data = (x_val, y_val)
        test_data = (x_test, y_test)
        
        return train_data, val_data, test_data
    
    def get_file_with_label(self, )