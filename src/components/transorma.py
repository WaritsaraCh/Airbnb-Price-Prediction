import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from src.Airbnb.logger import logging
from src.Airbnb.exception import customexception
from src.Airbnb.utils.utils import save_object

from sklearn.preprocessing import LabelEncoder, StandardScaler


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('Artifacts', 'preprocessor.pkl')
    

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
        
    def initate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            
            logging.info("Read the train and test data completed")
            
            
            cat_column = train_df.columns[train_df.dtypes == 'object']
            num_column = train_df.columns[train_df.dtypes != 'object']
            
            logging.info(f'Data Pre-Processing initiated')
            
            train_df.last_review.fillna(methods='ffill', inplace=True)
            test_df.last_review.fillna(methods='ffill', inplace=True)
            
            train_df.first_review.fillna(methods='ffill', inplace=True)
            test_df.first_review.fillna(methods='ffill', inplace=True)
            
            train_df.host_since.fillna(methods='ffill', inplace=True)
            test_df.host_since.fillna(methods='ffill', inplace=True)
            
            null_column = ['bathsrooms', 'beds', 'bedrooms']
            for col in null_column:
                train_df[col].fillna(train_df[col].median())
                test_df[col].fillna(test_df[col].median())
            logging.info(f'Handling missing values completed')
            
            # Handing Amenities column [train data]
            amenities_count_train = []
            for amenities in train_df['amenities']:
                amenities_count_train.append(len(amenities))
            train_df['amenities'] = amenities_count_train
            
            # Handing Amenities column [test data]
            amenities_count_test = []
            for amenities in test_df['amenities']:
                amenities_count_test.append(len(amenities))
            test_df['amenities'] = amenities_count_test
            
            logging.info("Amenities column handled successfully")
            
            train_df = train_df.dropna()
            test_df = test_df.dropna()
            
            target_column = 'log_price'
            drop_column = [
                target_column,
                "id",
                "name",
                "log_price",
                "description",
                "first_review",
                "host_since",
                "last_review",
                "neighbourhood",
                "thumbnail_url",
                "zipcode"
            ]
            
            input_feature_train_df = train_df.drop(columns=drop_column, axis=1)
            target_feature_train_df = train_df[target_column]
            logging.info('created input and target feature for train data')  
            
            input_feature_test_df = test_df.drop(columns=drop_column, axis=1)
            target_feature_test_df = test_df[target_column]
            logging.info('created input and target feature for test data')
            
            
            # Tranforming Input and Target Feature of Training and Testing Data
            combine_df = pd.concat([
                input_feature_train_df,
                input_feature_test_df
            ], axis=0)
            preprocessing_obj = LabelEncoder()
            req_col = ['property_type', 'room_type', 'bed_type', 'cancellation_policy', 'city', 'host_has_profile_pic','host_identity_verified','host_response_rate','instant_bookable']
            for col in req_col:
                combine_df[col] = preprocessing_obj.fit_transform(combine_df[col])
            
            input_feature_train_df[req_col] = combine_df[req_col][:len(input_feature_train_df)]
            input_feature_test_df[req_col] = combine_df[req_col][len(input_feature_test_df):]
            
            logging.info('Apply preprocessing object on training and testing data completed')
            
            train_array = np.c_[input_feature_train_df.values, np.array(target_feature_train_df)]
            test_array = np.c_[input_feature_test_df.values, np.array(target_feature_test_df)]
            
            logging.info('Training and Testing data transformed into array format')
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logging.info('Preprocessing pickle file saved')
            
            return (
                train_array,
                test_array,
            )
        
        
        except Exception as e:
            logging.info("Exception occured in reading train and test data")
            raise customexception(e, sys)