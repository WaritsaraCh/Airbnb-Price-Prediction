import os
import sys
import numpy as np


from dataclasses import dataclass
from src.Airbnb.logger import logging
from src.Airbnb.exception import customexception
from src.Airbnb.utils.utils import evaluate_model
from src.Airbnb.utils.utils import save_object

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from catboost import CatBoostRegressor



@dataclass
class ModelTrainerConfig:
    train_model_file_path = os.path.join('Artifacts', 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model Trainer method started")
            
            x_train, y_train, x_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:,-1]
            )
            
            models = {
                'LinearRegression':LinearRegression(),
                'RandomForestRegressor':RandomForestRegressor(),
                'GradientBoostingRegressor':GradientBoostingRegressor(),
                'CatBoostRegressor':CatBoostRegressor()
            }
            
            models_report:dict = evaluate_model(x_train, y_train, x_test, y_test, models)
            print(models_report)
            print('\n=======================================================================\n')
            logging.info(f'Models Report : {models_report}')
            
            best_model_score = max(sorted(models_report.values()))
            
            best_model_name = list(models_report.keys())[
                list(models_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]
            
            
            print('Best Model Found , Model Name : ', best_model_name, 'R2 Score : ', best_model_score)
            print('\n=======================================================================\n')
            logging.info(f'Best Model Found , Model Name : {best_model_name}, R2 Score : {best_model_score}')
            
            save_object(
                file_path=self.model_trainer_config.train_model_file_path,
                obj=best_model
            )
        except Exception as e:
            logging.info("Exception occured at Model Trainer")
            raise customexception(e, sys)
        
        
       