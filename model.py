import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, make_scorer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
import json
import codecs

class Model:

    def __init__(self,esitimator):
        self.estimator = esitimator

    def fit(self,data,evaluation = True, cv=5):
        x , y =  data[data.columns[:-1]], data["Market Share_total"]
        self.x_train, self.x_dev, self.y_train, self.y_dev = train_test_split(x, y, test_size=0.3, random_state=42)
        self.estimator.fit(self.x_train ,self.y_train)
        if evaluation:            
            y_pred = self.predict(self.x_dev)
            self._r2_score , self.mae = self.evaluate(self.y_dev, y_pred)
            self.r2_score_cv, self.mae_cv = self.perform_cv(cv)

    def predict(self,data):
        return self.estimator.predict(data)

    def save_results(self, path):
        result = { "Cross-validation": { "R2 Squared": str(self.r2_score_cv), "MAE":str(self.mae_cv) },
                    "Dev-Set" : { "R2 Squared":str(self._r2_score),    "MAE":str(self.mae) }
                }
        with codecs.open( path , "w", encoding= "utf-8") as J:
            json.dump(result,J, indent=4)
        print("results saved to {}".format(path))

    def evaluate(self,y_truth,y_pred):
        _r2_score = self.calculate_r2_score(y_truth, y_pred)
        mae = self.calculate_mae(y_truth, y_pred)
        return _r2_score , mae

    def calculate_mae(self,y_truth, y_pred):
        return mean_absolute_error(y_truth, y_pred)

    def calculate_r2_score(self,y_truth, y_pred):
        return r2_score(y_truth, y_pred)

    def perform_cv(self, cv = 5):
        r2_score_cv = self.perform_cv_r2_score(self.x_train , self.y_train, cv)
        mae_cv = self.perform_cv_mae(self.x_train , self.y_train, cv)
        return r2_score_cv, mae_cv

    def perform_cv_r2_score(self,x,y,cv = 5):
        cv_r2_scores = cross_val_score( self.estimator, x , y , cv = cv , scoring='r2')
        mean, std = self.get_mean_std(cv_r2_scores)
        return { str(cv) + "-Fold":cv_r2_scores , "Mean":mean, "Std":std }

    def perform_cv_mae(self,x ,y ,cv = 5):
        mse_scorer = make_scorer(mean_absolute_error)
        cv_maes = cross_val_score( self.estimator, x , y , cv = cv , scoring=mse_scorer)
        mean, std = self.get_mean_std(cv_maes)
        return { str(cv) + "-Fold":cv_maes , "Mean":mean, "Std":std }

    def get_mean_std(self, lst):
        return np.mean(lst), np.std(lst)
