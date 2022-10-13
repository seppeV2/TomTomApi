import requests
import json

#here the od api is used
#the body is loaded into body.json file and processed in this python script 
#the url and body are both build according the tomtom documentation 
#when I run this I get the https 400 code (error)

#the json file says: 
#{"fault":{"faultstring":"Received 405 Response without Allow Header","detail":{"errorcode":"protocol.http.Response405WithoutAllowHeader"}}}


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
