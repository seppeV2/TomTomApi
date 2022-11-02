from operational_functions import od_matrix_from_tomtom
import pathlib
import pandas as pd 
import geopandas as gpd
from dyntapy.demand_data import od_matrix_from_dataframes
from math import sqrt
import numpy as np
np.set_printoptions(suppress=True)


def main():
    original_od, tomtom_od = setup_test_case()
    




#This function sets up the test case we work with (to start)
#   The case is situated in Leuven and contains 44 zones
#   An average work day flow is taken from the morning peak hour (7 am to 9 am)
def setup_test_case():
    #first retrieve the original od matrix
    #get the right parameters
    od_table = pd.read_csv(str(pathlib.Path(__file__).parent)+"/data/Verplaatsingen_bestuurder_uur_spmVlaanderenVersie4.2.2_2017.csv")
    zoning = gpd.read_file(str(pathlib.Path(__file__).parents[1])+'/AllZonings/ZoningSmallLeuven.zip')
    origin_column = 'H'
    destination_column = 'B'
    zone_column = 'ZONENUMMER'
    flow_column = 'Tab:8'
    #result from 7 to 8
    result1  = od_matrix_from_dataframes(od_table,zoning, origin_column, destination_column,zone_column, flow_column)
    #only get the first result form the previous function
    od8 = result1[0]
    flow2_column = 'Tab:9'
    #result form 8 to 9
    result2 = od_matrix_from_dataframes(od_table, zoning, origin_column, destination_column, zone_column, flow2_column)
    od9 = result2[0]
    #result form 7 to 9
    original_od = np.add(od8,od9)

    #now retrieve the data from tomtom (saved as cvs file in data/move_results)
    path = str(pathlib.Path(__file__).parent)+'/data/move_results/testCaseLeuven.csv'
        #this needs to change for every new tomtom move analysis 
    flow_column = 'Date range: 2021-01-25 - 2021-01-29 Time range: 07:00 - 09:00'
    tomtom_od = od_matrix_from_tomtom(path, flow_column)
    #we want the average flow in the peak hours so we divide the flow from one work week by 5
    tomtom_od = np.round((tomtom_od / 5),3)
    return original_od, tomtom_od

main()
