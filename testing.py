import requests
import json

API_Key = "kOZG1tpItKNqMTAw9R8B5O752IjbxBfV"
url = "https://api.tomtom.com/routing/1/calculateRoute/41.160492,-8.663769:41.16159,-8.65161/json?maxAlternatives=0&departAt=2024-05-3T17:40:20Z&traffic=true&key="+API_Key

print(url)

bd = open("body.JSON")
body = json.load(bd)

r = requests.post(url, json = body)

print(r.status_code)