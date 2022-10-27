import json 
import requests 
import xmltodict
import sys
 
 #this is a test where we receive a png plot of the flow rates instead of just data.
 #click on the url to see the result

# setting path
sys.path.append('../TomTomApi')

from mapBuilding.MapBuilding import initialise_map, add_marker, add_polyLine

#base url
baseUrl = "https://api.tomtom.com/traffic/map/4/tile/flow/"

style = 'absolute'
zoom = 12
x = 2044
y = 1360
format = 'png'
apiKey = "m48ctXO8JwRAMDkRxbq2yqtngG3dsO7F"

addUrl = "%s/%s/%s/%s.%s?key=%s" % (style,str(zoom), str(x), str(y), format, apiKey)

url = baseUrl + addUrl
print(url)

r = requests.get(url)
print(r.status_code)