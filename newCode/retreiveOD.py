from dyntapy.demand_data import od_matrix_from_dataframes
import pandas as pd
import geopandas as gpd
import pathlib
import numpy as np




def get_od_matrix_from_database():

    #get the right parameters
    od_table = pd.read_csv(str(pathlib.Path(__file__).parent)+"/data/Verplaatsingen_bestuurder_uur_spmVlaanderenVersie4.2.2_2017.csv")
    zoning = gpd.read_file(str(pathlib.Path(__file__).parents[1])+'/AllZonings/ZoningSmallLeuven.zip')
    origin_column = 'H'
    destination_column = 'B'
    zone_column = 'ZONENUMMER'
    flow_column = 'Tab:7'

    odMatrix, X, Y = od_matrix_from_dataframes(od_table,zoning, origin_column, destination_column,zone_column, flow_column)
    odMatrix = np.round(odMatrix)

    #also save the matrix to csv file
    np.savetxt(str(pathlib.Path(__file__).parent)+"/data/results/odMatrixResult.csv", odMatrix, delimiter=",", fmt='%d')

    return odMatrix

get_od_matrix_from_database()