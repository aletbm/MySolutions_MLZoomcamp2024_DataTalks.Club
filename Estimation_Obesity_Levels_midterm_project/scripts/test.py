import requests
url = "http://EstimationObesityLevels.us-east-1.elasticbeanstalk.com/predict"

patient = {'gender': 'Female',
           'age': 21.0,
           'height': 1.62,
           'overweight_familiar': 'yes',
           'eat_hc_food': 'no',
           'eat_vegetables': 2.0,
           'main_meals': 3.0,
           'snack': 'Sometimes',
           'smoke': 'no',
           'drink_water': 2.0,
           'monitoring_calories': 'no',
           'physical_activity': 0.0,
           'use_of_technology': 1.0,
           'drink_alcohol': 'no',
           'transportation_type': 'Public_Transportation'
           #'obesity_level': 'Normal_Weight'
           }

print(requests.post(url, json=patient).json())

#pipenv install numpy pandas scikit-learn flask catboost waitress requests cloudpickle
#pipenv install awsebcli --dev