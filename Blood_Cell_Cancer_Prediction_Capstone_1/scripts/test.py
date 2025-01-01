import requests

req = {"url": "https://i.postimg.cc/c1Q236XF/0-6114.jpg"}
url = 'http://localhost:80/predict'
response = requests.post(url, json=req)
print(response.json())