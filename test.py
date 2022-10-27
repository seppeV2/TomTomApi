import geopandas as gpd

zoning = gpd.read_file('ZoningLeuven.zip')

print(zoning)