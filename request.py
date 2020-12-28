import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'surface':2,'nbr_pieces':2, 'langitude':2, 'latitude':2})

print(r.json())