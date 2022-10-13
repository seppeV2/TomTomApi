import requests
import json
import urllib.parse as urlparse


API_Key = "m48ctXO8JwRAMDkRxbq2yqtngG3dsO7F"
#url = "https://api.tomtom.com/origindestination/1/analysis/selected-link?key="+API_Key


url = "https://api.tomtom.com/origindestination/1/analysis/flowmatrix?"+API_Key



bd = open("body.JSON")
body = json.load(bd)


r = requests.post(url, json = body)

print(r.status_code)
print(r.headers)