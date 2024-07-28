import sys
import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from source.exception import CustomException
from source.logger import logging
from source.utils import save_object

@dataclass
class Data_Transformation_Config:
    preprocessor_of_file_path=os.path.join('artifact', "preprocessor.pkl")

class data_transformation:
    def __init__(self):
        self.data_transformation_config = Data_Transformation_Config()

    def get_data_transformer_object(self):
        '''
        This function is for data transformation '''
        try:
            numerical_columns =["writing_score","reading_score"]
            categorical_columns =[
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course",
            ]
            numerical_pipeline = Pipeline(
                steps=[
                    ("Imputer",SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("Imputer", SimpleImputer(strategy="most_frequent")),
                    ("One_Hot_Encoder", OneHotEncoder()),
                   
                ]

            )
            logging.info("numerical columns scaling completed")
            logging.info("categorical columns encoding completed")

            preprocessor = ColumnTransformer(
                [("num_pipeline", numerical_pipeline, numerical_columns),
                 ("categorical_pipeline", categorical_pipeline, categorical_columns)]
                
            )
            
            return preprocessor
        
        except Exception as e:
            raise CustomException(e,sys)
            
    def initiate_data_transformation(self, train_path,test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("obtaining preprocessing object")

            preprocessing_object = self.get_data_transformer_object()
            target_column_name = "math_score"
            numerical_columns = ["reading_score","writing_score"]

            input_fearture_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_fearture_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info(f" Applying preprocessing object on training dataframe and testing dataframe")

            input_feature_train_arr = preprocessing_object.fit_transform(input_fearture_train_df)
            input_feature_test_arr = preprocessing_object.transform(input_fearture_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)

            ]

            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)

            ]

            logging.info("saved preprocessing_object")

            save_object(

                file_path=self.data_transformation_config.preprocessor_of_file_path,
                obj=preprocessing_object

            )
            return (train_arr,
                    test_arr,
                    self.data_transformation_config.preprocessor_of_file_path
            )


        except Exception as e:
            raise CustomException(e,sys)


