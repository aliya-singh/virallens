import requests

url = "http://127.0.0.1:8000/predict"
data = {"features": [1, 2, 15.6, 3, 5, 8, 2, 4, 1, 2.3, 0]}  
response = requests.post(url, json=data)

print("Status Code:", response.status_code)
print("Response Text:", response.text)

if response.status_code == 200:
    print(response.json())
