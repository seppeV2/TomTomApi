import requests
import json

apiKey = "m48ctXO8JwRAMDkRxbq2yqtngG3dsO7F"

baseUrl = "https://api.tomtom.com/origindestination/1/analysis/selected-link"
extraUrl = "?key=%s" % (apiKey)
url = baseUrl + extraUrl

bd = open('ODAnalysis/body.JSON')
body = json.load(bd)

w = open('ODAnalysis/test.json', 'w')
json.dump(body, w)
w.close()

r = requests.post(url, json = body )

print(url)
print(r.status_code)
print(r.headers)
