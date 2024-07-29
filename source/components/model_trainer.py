import os 
import sys
from dataclasses import dataclass

from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from source.exception import CustomException
from source.logger import logging
from source.utils import save_object, evaluate_models

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join("artifact", "model.pkl")

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_array, test_array, preprocessor_path=None):
        try:
            logging.info("Splitting Training and Test input data")
            X_train,y_train, X_val, y_val = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models ={
                "Random Forest":RandomForestRegressor(),
                "Decision Tree":DecisionTreeRegressor(),
                "Gradient Boost": GradientBoostingRegressor(),
                "K-neighbors Regression": KNeighborsRegressor(),
                "Linear Regression": LinearRegression(),
                "XGB Regressor":XGBRegressor(),
                "AdaBoost Regressor": AdaBoostRegressor()

            }

            # Usage
            model_report = evaluate_models(x_train=X_train, y_train=y_train, x_val=X_val, y_val=y_val, models=models)

            # To get the best model score from dict
            best_model_score = max(model_report.values())

            best_model_name = max(model_report, key=model_report.get)

            best_model = models[best_model_name]



            print(f"Best Model: {best_model_name}")
            print(f"Best Score: {best_model_score}")

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted= best_model.predict(X_val)
            r2_square = r2_score(y_val,predicted)
            return r2_square


        except Exception as e:
            raise CustomException(e,sys)
