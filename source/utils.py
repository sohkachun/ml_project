import os
import sys
import dill

import numpy as np
import pandas as pd

from source.exception import CustomException, logging
from sklearn.metrics import r2_score

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path,"wb") as file_obj:
            dill.dump(obj,file_obj)
    
    except Exception as e:
        raise CustomException(e, sys) 

def evaluate_models(x_train,y_train, x_val,y_val, models ):
    try:
        report = {}

        for model_name, model in models.items():
            logging.info(f"Evaluating model: {model_name}")

            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)
            y_val_pred = model.predict(x_val)
            
            # Calculate scores
            train_score = r2_score(y_train, y_train_pred)
            val_score = r2_score(y_val, y_val_pred)
            
            # Store scores in report
            report[model_name] = {
                'train_score': train_score,
                'val_score': val_score
            }
            
            logging.info(f"{model_name} - Train Score: {train_score:.4f}, Validation Score: {val_score:.4f}") 
            return report

    except Exception as e:
        raise CustomException(e,sys)
    

def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)
    except Exception as e:
        raise CustomException(e, sys)