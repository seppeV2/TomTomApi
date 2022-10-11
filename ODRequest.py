import http.client
import json 

API_Key1 = "m48ctXO8JwRAMDkRxbq2yqtngG3dsO7F"
API_Key2 = "Gcw5ye6iGdXCPqtOllT2idD7OUmOUIG7"
API_Key3 = "ealGbxH7zbd5RKJg94Yz4yvnrB2d0cu1"
url = "https://api.tomtom.com/origindestination/1/analysis/flowmatrix?key={"+API_Key1+"}"


body = open('body.JSON')
myobj = json.load(body)

h = http.client.HTTPSConnection(url)
h.request("GET", "/get")
r = h.getresponse()
print(r.status)
