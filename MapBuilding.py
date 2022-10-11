# First we import a bunch of libraries
import pandas as pd
import numpy as np
import folium # map visualisation package
import requests # this we use for API calls
import json
import matplotlib.pyplot as plt
import branca.colormap as cm
from dateutil import tz
import datetime
import time
from tqdm import tqdm

api_key1 = 'm48ctXO8JwRAMDkRxbq2yqtngG3dsO7F'
api_key2 = 'Gcw5ye6iGdXCPqtOllT2idD7OUmOUIG7'
api_key3 = 'ealGbxH7zbd5RKJg94Yz4yvnrB2d0cu1'

#locations =  [longitude, latitude] 
location_amsterdam = [52.377956, 4.897070]
location_brussel = [50.8503396, 4.3517103]

# Initiate the map with the TomTom maps API
def initialise_map(api_key=api_key1, location=location_brussel, zoom=14, style = "main"):
    """
    The initialise_map function initialises a clean TomTom map
    """
    maps_url = "http://{s}.api.tomtom.com/map/1/tile/basic/"+style+"/{z}/{x}/{y}.png?tileSize=512&key="
    TomTom_map = folium.Map(
    location = location, # on what coordinates [lat, lon] we want to initialise our map
    zoom_start = zoom, # with what zoom level we want to initialise our map, from 0 to 22
    tiles = str(maps_url + api_key),
    attr = 'TomTom')
    return TomTom_map

def add_marker(TomTom_map, location, popup, tipmsg):
    folium.Marker(location =location, popup = popup, tooltip = tipmsg).add_to(TomTom_map)


# Save map as TomTom_map
TomTom_map = initialise_map()
add_marker(TomTom_map, location_brussel, "<b>Brussel Centrum</b>", 'Click here')
TomTom_map.save('map.html')

print('https://api.tomtom.com/origindestination/1/analysis/selected-link?key={'+api_key1+'}')
