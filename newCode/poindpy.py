# Importing required packages
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn import metrics
import folium
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_notebook, show
from bokeh.tile_providers import CARTODBPOSITRON, get_provider, Vendors
from bokeh.models import GeoJSONDataSource
from bokeh.palettes import Spectral5, Category20
import pathlib

import sys
sys.path.insert(0, '..')
import os

import poidpy as poid

path = str(pathlib.Path(__file__).parents[1]) + '/AllZonings/'
zones_city = gpd.read_file(path + 'ZoningSmallLeuven.zip')

zones_studyarea = zones_city
# Plot study area
zones_city = zones_city.to_crs(epsg=4326)
poly = zones_city.unary_union

m = folium.Map(location=(poly.centroid.y, poly.centroid.x), zoom_start=11)

zones_gjson = folium.features.GeoJson(zones_city, name=f"Zones Ghent", style_function= lambda feature:{
        "fillOpacity":0.75 if feature['properties']['ZONENUMMER'] in zones_studyarea.ZONENUMMER.to_list() else 0.4,
        'weight': 0.8,
        'color': 'black',
        "fillColor": 'blue' if feature['properties']['ZONENUMMER'] in zones_studyarea.ZONENUMMER.to_list() else 'grey'
    })

zones_gjson.add_to(m)

folium.GeoJsonPopup(fields=['ZONENUMMER'], aliases=['ZONENUMMER']).add_to(zones_gjson)



