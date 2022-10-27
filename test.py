import geopandas as gpd

zoning = gpd.read_file('AllZonings/ZoningLeuven.zip')

print(zoning)