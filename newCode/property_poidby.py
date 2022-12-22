# Importing required packages
import pandas as pd
import geopandas as gpd
import numpy as np
from sklearn import metrics
import folium
import pathlib
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_notebook, show
from bokeh.tile_providers import CARTODBPOSITRON, get_provider, Vendors
from bokeh.models import GeoJSONDataSource
from bokeh.palettes import Spectral5, Category20
from shapely.geometry import Polygon, Point

#output_notebook()

import sys
sys.path.insert(0, '..')
import os

import poidpy as poid

def main():
    zonings = ['ZoningSmallLeuven', 'BruggeWithoutZeeBrugge', 'Hasselt']
    #zonings = ['ZoningSmallLeuven']
    for zone in zonings:
        make_shape_file(zone)
   
def make_shape_file(zone):
    zone_path = str(pathlib.Path(__file__).parents[1]) + '/AllZonings/{}.zip'.format(zone)
    result_path = str(pathlib.Path(__file__).parent)  + '/poidby_results/'
    zones_city = gpd.read_file(zone_path)
    zones_city = zones_city.to_crs(epsg=4326)

    try:
        demand_obj = poid.read_pickle(f'{zone}.class', path=result_path)
        demand_obj.set_initial_path = result_path
    except:
        demand_obj = poid.create_demand_class(zone, zones_city, result_path, timeout=1000)

    dataframe = gpd.GeoDataFrame(demand_obj.poi.loc[:,['osmid','geometry', 'landuse', 'building']])
    dataframe = dataframe[dataframe["landuse"].notnull()]
    dataframe.to_file('landuse_{}.zip'.format(zone))