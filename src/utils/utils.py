import os
import sys
import pickle
import numpy as np
import pandas as pd

from src.Airbnb.logger import logging
from src.Airbnb.exception import customexception

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
    except Exception as e:
        raise customexception(e, sys)
    
def evaluate_model(X_train, y_train, x_test, y_test, models):
    try:
        report = {}
        
        # ใช้ .items() จะช่วยให้ดึงทั้งชื่อ (key) และตัวโมเดล (model) ออกมาได้พร้อมกัน
        for model_name, model in models.items():
            
            # 1. Train model
            model.fit(X_train, y_train)
            
            # 2. Predict data
            y_test_pred = model.predict(x_test)
            
            # 3. Calculate R2 Score
            test_model_score = r2_score(y_test, y_test_pred)
            
            # 4. เก็บชื่อโมเดลเป็น Key และคะแนนเป็น Value
            report[model_name] = test_model_score
            
        return report

    except Exception as e:
        raise customexception(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        logging.info("Error in loading object")
        raise customexception(e, sys)