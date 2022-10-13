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



# Initiate the map with the TomTom maps API
def initialise_map(api_key, location, zoom, style):
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

#add a marker with this function 
def add_marker(TomTom_map, location, popup, tipmsg):
    folium.Marker(location =location, popup = popup, tooltip = tipmsg).add_to(TomTom_map)

#add a polyline with this function
def add_polyLine(TomTom_map, coorList):
    folium.PolyLine(coorList).add_to(TomTom_map)
