import os
import sys
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

from src.exception import CustomException
from src.logging import logging

def save_object(file_path, obj):
    try:
        dir_path=os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            dill.dump(obj, file_obj)
            
    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try: 
        with open(file_path, 'rb') as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise CustomException(e, sys)


def evaluate_model(X_train, y_train, X_test, y_test, model, param):
    try:
        report = {}
        best_models = {}

        for model_name, model in model.items():
            logging.info('Tunning is started by GridSearchCV')
            
            model_param = param.get(model_name, None)

            if model_param:
                logging.info(f"Tuning hyperparameters for {model_name}")
                gs = GridSearchCV(model, model_param, cv=3, n_jobs=-1, verbose=0)
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_
            else:
                model.fit(X_train, y_train)
                best_model = model

            y_test_pred=model.predict(X_test)
            test_model_score=r2_score(y_test, y_test_pred)

            report[model_name] = test_model_score
            best_models[model_name] = best_model
            logging.info(f'Best model found! {model_name}, R2 score: {test_model_score}')

            return report, best_models
        
    except Exception as e:
        raise CustomException(e, sys)
