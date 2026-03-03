import os
import numpy as np
import sys
import pandas as pd
from src.Airbnb.logger import logging
from src.Airbnb.utils.utils import load_object
from src.Airbnb.exception import customexception


class PredictPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            preprocessor_path = os.path.join("Artifacts", "preprocessor.pkl")
            model_path = os.path.join("Artifacts", "model.pkl")
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)
            logging.info('Preprocessor and Model Pickle files loaded')
            scaled_data = preprocessor.transform(features)
            logging.info('Data Scaled')
            pred = model.predict(scaled_data)
            return pred
        except Exception as e:
            raise customexception(e, sys)

class CustomData:
    def __init__(self,
                 property_type: str,
                 room_type: str,
                 amenities: int,
                 accommodates: int,
                 bathrooms: int,
                 bed_type: str,
                 cancellation_policy: str,
                 cleaning_fee: float,
                 city: str,
                 host_has_profile_pic: str,
                 host_identity_verified: str,
                 host_response_rate: str,
                 instant_bookable: str,
                 latitude: float,
                 longitude: float,
                 number_of_reviews: int,
                 review_scores_rating: int,
                 bedrooms: int,
                 beds: int):
        
        self.property_type = property_type
        self.room_type = room_type
        self.amenities = amenities
        self.accommodates = accommodates
        self.bathrooms = bathrooms
        self.bed_type = bed_type
        self.cancellation_policy = cancellation_policy
        self.cleaning_fee = cleaning_fee
        self.city = city
        self.host_has_profile_pic = host_has_profile_pic
        self.host_identity_verified = host_identity_verified
        self.host_response_rate = host_response_rate
        self.instant_bookable = instant_bookable
        self.latitude = latitude
        self.longitude = longitude
        self.number_of_reviews = number_of_reviews
        self.review_scores_rating = review_scores_rating
        self.bedrooms = bedrooms
        self.beds = beds

    def get_data_as_dataframe(self):
        try:
            # ฟังก์ชันช่วยแปลง String เป็น Boolean
            def to_bool(val):
                if isinstance(val, str):
                    return val.lower() == 'true'
                return bool(val)

            # แปลงค่า cleaning_fee ให้เป็น Boolean ก่อนส่งเข้า DataFrame
            # (ตรวจสอบด้วยว่าตอน Train คุณใช้ Boolean หรือไม่ ถ้าใช่ต้องแปลงแบบนี้)
            cleaning_fee_val = to_bool(self.cleaning_fee)
            
            # กรณีของ host_has_profile_pic, host_identity_verified, instant_bookable 
            # ถ้าตอน Train ใช้ 't'/'f' ให้คงไว้เป็น String แต่ถ้าใช้ True/False ต้องแปลงด้วย
            
            custom_data_input_dict = {
                # --- Numerical (10) ---
                'amenities': [float(self.amenities or 0)],
                'accommodates': [float(self.accommodates or 0)],
                'bathrooms': [float(self.bathrooms or 0)],
                'latitude': [float(self.latitude or 0)],
                'longitude': [float(self.longitude or 0)],
                'host_response_rate': [float(self.host_response_rate or 0)],
                'number_of_reviews': [float(self.number_of_reviews or 0)],
                'review_scores_rating': [float(self.review_scores_rating or 0)],
                'bedrooms': [float(self.bedrooms or 0)],
                'beds': [float(self.beds or 0)],

                # --- Categorical (9) ---
                'property_type': [self.property_type],
                'room_type': [self.room_type],
                'bed_type': [self.bed_type],
                'cancellation_policy': [str(self.cancellation_policy).lower()],
                'cleaning_fee': [cleaning_fee_val], # ส่งค่า Boolean (True/False) แทน String
                'city': [self.city],
                'host_identity_verified': [self.host_identity_verified],
                'instant_bookable': [self.instant_bookable],
                'host_has_profile_pic': [self.host_has_profile_pic]
            }
            
            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise customexception(e, sys)