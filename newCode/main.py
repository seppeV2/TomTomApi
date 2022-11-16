from operational_functions import od_matrix_from_tomtom, calculate_gap,matrix_to_list, normalize, get_split_matrices, list_to_matrix
import pathlib
import pandas as pd 
import geopandas as gpd
from dyntapy.demand_data import od_matrix_from_dataframes
import numpy as np
import scipy
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from matplotlib.container import BarContainer
np.set_printoptions(suppress=True)


def main():
    zonings = ['ZoningSmallLeuven', 'BruggeWithoutZeeBrugge', 'Hasselt']
    moves = ['testCaseLeuven','BruggeWithoutZeeBrugge', 'Hasselt']

    origGap = []
    approxGap = []
    slopes = []
    intercepts = []

    plot = True

    for zone, move in zip(zonings, moves):
        print("START SETUP FOR {}\n".format(move))
        original_od, tomtom_od = setup_test_case(zone, move)
            #normalize the matrices
        original_od = normalize(original_od)
        tomtom_od = normalize(tomtom_od)    

        print("START LINEAR REGRESSION\n")
        slopes_one_zone = []
        intercepts_one_zone = []
        loop = 0
        shapes = get_split_matrices(tomtom_od)
        approx_matrix = np.zeros(tomtom_od.shape)
        for i  in range(len(shapes)):
            #resize matrix to list to apply the linear regression
            tomtom_list = matrix_to_list(tomtom_od, shapes[i])
            original_list = matrix_to_list(original_od, shapes[i])

            #actual linear regression
            res = scipy.stats.linregress(tomtom_list, original_list)
            slopes_one_zone.append(res.slope)
            intercepts_one_zone.append(res.intercept)
            print("The regression for split {}: Y = {} * X + {} ".format(loop, res.slope, res.intercept))

            #reconstruct the matrix after applying the linear regression
                #apply linear regression 
            intermediate_list = (tomtom_list * res.slope) + res.intercept
            intermediate_array = list_to_matrix(intermediate_list, shapes[i])
            approx_matrix += intermediate_array
            loop += 1 

        original_gap = calculate_gap(original_od, tomtom_od)
        origGap.append(original_gap)
        print('\noriginal gap = {}'.format(original_gap))
    
        approx_gap = calculate_gap(original_od, approx_matrix)
        approxGap.append(approx_gap)
        print('second gap = {}'.format(approx_gap))

        slopes.append(slopes_one_zone)
        intercepts.append(intercepts_one_zone)

    if plot:
        #make some plots 
        fig,ax = plt.subplots()

        width = 0.3
        x_original = [x-width/2 for x in range(len(origGap))]
        x_approx = [x+width/2 for x in range(len(approxGap))]

        ax.bar(x_original, origGap, width, label = 'original Gap',color = 'darkslategray')
        ax.bar(x_approx, approxGap, width, label = 'approx Gap',color = 'crimson')
        
        ax.set_xticks(range(len(moves)), moves)
        ax.set_title('Gaps for different zones before and after linear regression')
        ax.set_ylabel('normalized gap')
        ax.legend()

        fig,ax2 = plt.subplots()
        x = np.linspace(0,10,300)

        for s, i , zone in zip(slopes, intercepts, move):
            for k in range(len(s)):
                y1 = s[k]*x + i[k]
                ax2.plot(x,y1, label = 'Eq for {}th piece of {} od_reg'.format(k,zone))

        ax2.legend()
        ax2.set_ylabel("Y")
        ax2.set_xlabel("X")
        ax2.set_title('Linear equations from the different test zones + split matrices')

        plt.show()




#This function sets up the test cases we work with (to start)
#   An average work day flow is taken from the morning peak hour (7 am to 9 am)
def setup_test_case(nameZoning: str, nameTomTomCsv: str):
    #first retrieve the original od matrix
    #get the right parameters
    od_table = pd.read_csv(str(pathlib.Path(__file__).parent)+"/data/Verplaatsingen_bestuurder_uur_spmVlaanderenVersie4.2.2_2017.csv")
    zoning = gpd.read_file(str(pathlib.Path(__file__).parents[1])+'/AllZonings/{}.zip'.format(nameZoning))
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
    path = str(pathlib.Path(__file__).parent)+'/data/move_results/{}.csv'.format(nameTomTomCsv)
        #this needs to change for every new tomtom move analysis 
    flow_column = 'Date range: 2021-01-25 - 2021-01-29 Time range: 07:00 - 09:00'
    tomtom_od = od_matrix_from_tomtom(path, flow_column)
    #we want the average flow in the peak hours so we divide the flow from one work week by 5
    tomtom_od = np.round((tomtom_od / 5),3)
    return original_od, tomtom_od

main()
