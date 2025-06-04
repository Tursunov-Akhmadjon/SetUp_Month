import os
import sys
import pandas as pd
import numpy
from src.exception import CustomException
from src.logging import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        model_path = 'artifacts\model.pkl'
        preprocessor_path = 'artifacts\preprocessing.pkl'
        model=load_object(file_path=model_path)

class CustomData:
    def __init__(self, 
                 gender:str, 
                 ):
          self.gender = gender,
    
    def get_data_as_data_frame
