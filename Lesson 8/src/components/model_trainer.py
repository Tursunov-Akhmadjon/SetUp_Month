import os
import sys

from dataclasses import dataclass
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

from src.exception import CustomException
from src.logging import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainingConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainingConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Data Split")

            X_train, y_train, X_test, y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )

            models = {
                "RandomForestRegressor": RandomForestRegressor(),
                "GradientBoostingRegressor": GradientBoostingRegressor(),
                "AdaboostRegressor": AdaBoostRegressor(),
                "DecisionTreeRegressor": DecisionTreeRegressor(),
                "LinearRegressor": LinearRegression(),
                "XGBoostRegressor": XGBRegressor()
            }

            params = {
                "Decision Tree": {
                    'max_depth': [1,2,10,None],
                    'min_sample_split': [1,3,5],
                    'criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
                },
                "Random Forest":{
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Gradient Boosting":{
                    'learning_rate': [0.1,0.01,0.05,0.001],
                    'subsample': [0.6,0.7,0.75,0.8,0.85,0.9],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "Linear Regression": {},
                "XGBoost Regression": {
                    'learning_rate': [0.1,0.01,0.05,0.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
                "AdaBoost Regression": {
                    'learning_rate': [0.1,0.01,0.05,0.001],
                    'n_estimators': [8,16,32,64,128,256]
                },
            }


            model_report, best_model = evaluate_model(
                X_train=X_train, y_train=y_train, 
                X_test=X_test, y_test=y_test, 
                model=models, param=params)

            best_model_name = max(model_report, key=model_report.get)
            best_model_score=model_report[best_model_name]
            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException('Yomon Model: Tuning yoki boshqa model rivojlantirish methodi kerak!')

            logging.info(f'Best Model Found: {best_model_name} R2 score {best_model_score:.2f}')

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )

            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            return r2

        except Exception as e:
            raise CustomException(e, sys)