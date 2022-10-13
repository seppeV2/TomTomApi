import requests
import json

API_Key = "m48ctXO8JwRAMDkRxbq2yqtngG3dsO7F"
url = "https://api.tomtom.com/origindestination/1/analysis/selected-link?key="+API_Key

bd = open("body.JSON")
body = json.load(bd)

r = requests.post(url, json = body)

r = requests.get(url)

print(r.status_code)