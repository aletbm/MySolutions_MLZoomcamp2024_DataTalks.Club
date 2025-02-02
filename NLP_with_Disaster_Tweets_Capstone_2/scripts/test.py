import requests

req = {"keyword":"ablaze",
       "location":"London",
       "text": "Birmingham Wholesale Market is ablaze BBC News - Fire breaks out at Birmingham's Wholesale Market http://t.co/irWqCEZWEU"}

url = 'http://localhost:80/predict'
response = requests.post(url, json=req)
print(response.json())