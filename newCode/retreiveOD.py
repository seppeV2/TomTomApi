import dyntapy.demand_data
import pandas as pd
import geopandas as gpd
import pathlib

od_table = pd.read_csv(str(pathlib.Path(__file__).parent)+"/data/Verplaatsingen_bestuurder_uur_spmVlaanderenVersie4.2.2_2017.csv")
zoning = gpd.read_file(str(pathlib.Path(__file__).parent)+'/data/ZoningLeuven.zip')
origin_column = 'H'
destination_column = 'B'
zone_column = 'ZONENUMMER'
flow_column = 'Tab:7'

odMatrix = dyntapy.od_matrix_from_dataframes(od_table,zoning, origin_column, destination_column,zone_column, flow_column)
print(odMatrix)
