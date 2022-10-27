import dyntapy
import pandas as pd
import geopandas as gpd

#od_table = pd.read_csv("data/Verplaatsingen_bestuurder_uur_spmVlaanderenVersie4.2.2_2017.CVS")
zoning = gpd.read_file("data/ZoningLeuven/ZoningLeuven.zip")
origin_column = ""
destination_column = ""
zone_column = ""
flow_column = ""
return_relabelling = True


print(zoning)
#dyntapy.demand_data.od_matrix_from_dataframes(od_table,zoning, origin_column, destination_column, flow_column, return_relabelling)