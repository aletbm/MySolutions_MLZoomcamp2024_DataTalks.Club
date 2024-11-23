import pandas as pd
import random
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from catboost import CatBoostClassifier
import cloudpickle

class MyStandardScaler(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.ss = StandardScaler().set_output(transform="pandas")
        self.numeric_cols = ['age', 'height', 'eat_vegetables', 'main_meals', 'drink_water', 'physical_activity', 'use_of_technology']
        return

    def fit(self, X):
        self.ss.fit(X[self.numeric_cols])
        return self
    
    def transform(self, X):
        X[self.numeric_cols] = self.ss.transform(X[self.numeric_cols])
        return X.to_dict("records")

seed_value = 42
os.environ['PYTHONHASHSEED'] = str(seed_value)
random.seed(seed_value)
np.random.seed(seed_value)

pathroot = "../dataset/"
df = pd.read_csv(pathroot+"ObesityDataSet_raw_and_data_sinthetic.csv")

df = df.rename(columns={"family_history_with_overweight": "overweight_familiar",
                       "FAVC":"eat_HC_food",
                       "FCVC":"eat_vegetables",
                       "NCP":"main_meals",
                       "CAEC":"snack",
                       "CH2O":"drink_water",
                       "SCC":"monitoring_calories",
                       "FAF":"physical_activity",
                       "TUE":"use_of_technology",
                       "CALC":"drink_alcohol",
                       "MTRANS":"transportation_type",
                       "NObeyesdad":"obesity_level"
                       }).rename(columns=str.lower)

df = df.drop_duplicates()

df_full_train, df_test = train_test_split(df, test_size=0.15, random_state=seed_value, stratify=df["obesity_level"])

cols_drop = ["obesity_level", "weight"]
X_full_train, y_full_train = df_full_train.drop(cols_drop, axis=1), df_full_train["obesity_level"]
X_test, y_test =  df_test.drop(cols_drop, axis=1), df_test["obesity_level"]

pipe = Pipeline([('ss', MyStandardScaler()), ('dv', DictVectorizer(sparse=False))])

X_full_train = pipe.fit_transform(X_full_train)
X_test = pipe.transform(X_test)

le = LabelEncoder()
le.fit(y_full_train)
y_full_train = le.transform(y_full_train)
y_test = le.transform(y_test)

class_full_weight = compute_class_weight(class_weight="balanced", classes=np.unique(y_full_train), y=y_full_train)
class_full_weight = dict(zip(np.unique(y_full_train), class_full_weight))

sample_full_weights = compute_sample_weight(
    class_weight=class_full_weight,
    y=y_full_train
)
cbc = CatBoostClassifier(loss_function='MultiClass',
                        eval_metric='AUC',
                        iterations=10000,
                        depth=6,
                        classes_count=7,
                        class_weights=class_full_weight,
                        learning_rate=0.1,
                        od_type='Iter',
                        early_stopping_rounds=1000,
                        bootstrap_type='MVS',
                        sampling_frequency='PerTree',
                        random_seed=seed_value,
                        verbose=True)

if __name__ == "__main__":
    cbc.fit(X_full_train, y_full_train, sample_weight=sample_full_weights, eval_set=(X_test, y_test))

    with open('../model/obesity-levels-model.bin', 'wb') as f_out:
        cloudpickle.dump((pipe, le, cbc), f_out)
