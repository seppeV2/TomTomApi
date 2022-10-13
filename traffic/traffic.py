import json 
import requests 
import xmltodict
import sys
 
# setting path
sys.path.append('../TomTomApi')

from mapBuilding.MapBuilding import initialise_map, add_marker, add_polyLine

#here we send an request to the tomtom traffic api, with most important parameter point 
#The road closest the this point is taken, from that road some info is given like the speed in free flow, current speed
#travel time (free flow/current) ... also a list is given with coordinates that visualize the section monitored. 



#Buildig the right url to send the https request (according the tomtom traffic api documentation

#base url
baseUrl = "https://api.tomtom.com/traffic/services/4/flowSegmentData/"

#url with options
style = "reduced-sensitivity"
zoom = str(10)
formatt = "xml"
apiKey = "m48ctXO8JwRAMDkRxbq2yqtngG3dsO7F"
lonpoint = 50.88354237733045
latpoint = 4.709234551137543
point = "%s,%s" % (str(lonpoint), str(latpoint)) #jean baptiste van monsstraat in leuven
#build the url with extra
extraUrl = "%s/%s/%s?key=%s&point=%s" % (style, zoom, formatt, apiKey, point)
#complete url
url = baseUrl + extraUrl

#https request (GET)
r = requests.get(url)

#print some info
print(r.status_code)
print(url)

#store the respons in a json file (respons.json)
#so we can do something with the respons
jsonTrans = xmltodict.parse(r.text)
respons = open('traffic/respons.json', 'w')
json.dump(jsonTrans, respons)
respons.close()

##Visualisation##

#open the respons of the https request
f = open("traffic/respons.json")
x = json.load(f)

#build an array of [lon, lat] from the section that is monitorred
polyArray = []
for i in x["flowSegmentData"]["coordinates"]["coordinate"]:
    polyArray.append([float(i['latitude']),float(i['longitude'])])

#building a map with a polyline whit the section that is used. 
map = initialise_map(api_key = apiKey,location = [lonpoint,latpoint], zoom = 14, style = 'main')
add_marker(TomTom_map= map,location = [lonpoint,latpoint] , popup = "<b>jean baptiste van monsstraat</b>", tipmsg = 'street name')
add_polyLine(TomTom_map = map, coorList = polyArray,)

map.save('traffic/result.html')