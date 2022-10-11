import http.client
import json 

API_Key = "m48ctXO8JwRAMDkRxbq2yqtngG3dsO7F"
url = "https://api.tomtom.com/origindestination/1/analysis/flowmatrix?key={"+API_Key+"}"


body = open('body.JSON')
myobj = json.load(body)

h = http.client.HTTPSConnection(url)
h.request("GET", "/get")
r = h.getresponse()
print(r.status)