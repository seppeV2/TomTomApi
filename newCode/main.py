from operational_functions import heatmaps, od_matrix_from_tomtom, calculate_gap,matrix_to_list, normalize, get_split_matrices, list_to_matrix
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
        #amount of km road in the zone
    road_coverage = [997.67,1069.56,1487.82]
        #squared km 
    area = [57.52,95.36,102.53]

    origGap = []
    approxGap = []
    slopes = []
    intercepts = []

    plot = True
    heatmap = False

    tomtomData = {}
    originalData = {}
    shapes_dic = {}
    approxOD = {}

    for zone, move in zip(zonings, moves):
        print("START SETUP FOR {}\n".format(move))
        original_od, tomtom_od = setup_test_case(zone, move)
            #normalize the matrices
        original_od = normalize(original_od)
        tomtom_od = normalize(tomtom_od) 

        tomtomData[move] = tomtom_od
        originalData[move] = original_od
        
        

        print("START LINEAR REGRESSION\n")
        slopes_one_zone = []
        intercepts_one_zone = []
        loop = 0
        shapes = get_split_matrices(tomtom_od)
        shapes_dic[move] = shapes

        if heatmap:
            heatmaps(original_od, tomtom_od, zone, 'original', 'tomtom')  
            heatmaps(shapes[0], shapes[1], 'split form zone {}'.format(zone), 'split 1', 'split 2')

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
    
        approxOD[move] = approx_matrix
        approx_gap = calculate_gap(original_od, approx_matrix)
        approxGap.append(approx_gap)
        print('second gap = {}'.format(approx_gap))

        slopes.append(slopes_one_zone)
        intercepts.append(intercepts_one_zone)

    slopes_first_shape = [slope[0] for slope in slopes]
    slopes_second_shape = [slope[1] for slope in slopes]


    if plot:
       
       #plot the bar plots
        bars(origGap,approxGap, intercepts, moves)

        #plot the equations
        equations(slopes, intercepts, move)    

        #plot the scatters (with their linear fit)
        modelSlope, modelIntercept = scatters(slopes_first_shape ,area, road_coverage)
        averageModelIntercept = np.average([i[0] for i in intercepts])
        print("Model to build MOW: [({} * road_coverage + {}) * tomtom_od + {}] * number_of_trips".format(modelSlope, modelIntercept, averageModelIntercept))
        string = "Model used: [({} * road_coverage + {}) * tomtom_od + {}] * number_of_trips".format(modelSlope, modelIntercept, averageModelIntercept)
        approx_gap2 = []
        for idx, move in enumerate(moves): 
            newMatrix = (((modelSlope * road_coverage[idx]) + modelIntercept) * tomtomData[move])*shapes_dic[move][0] + approxOD[move]*shapes_dic[move][1]
            approx_gap2.append(calculate_gap(originalData[move], newMatrix))

        bars(origGap, approxGap, intercepts, moves, approx_gap2, string)
        plt.show()


#make scatter and plot the best linear fit
def scatters(slopes,area, road_coverage):
    x2 = np.linspace(min(area)-20,max(area)+20,1000)

    _, ax3 = plt.subplots()
    plt.scatter(area, slopes)
    res = scipy.stats.linregress(area, slopes)
    y3 = res.slope * x2 + res.intercept
    plt.plot(x2,y3, label = "y = {} x + {}".format(res.slope, res.intercept))
    ax3.legend()
    ax3.set_ylabel("area")
    ax3.set_xlabel("slope linear equation")
    ax3.set_title('scatter plot between area and slope of the linear equation')
    plt.savefig(str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/scatter_area.png')

    x3 = np.linspace(min(road_coverage)-20,max(road_coverage)+20,1000)
    _, ax4 = plt.subplots()
    plt.scatter(road_coverage, slopes)
    res2 = scipy.stats.linregress(road_coverage, slopes)
    y4 = res2.slope * x3 + res2.intercept
    plt.plot(x3,y4, label = "y = {} x + {}".format(res2.slope, res2.intercept))
    ax4.legend()
    ax4.set_ylabel("road coverage")
    ax4.set_xlabel("slope linear equation")
    ax4.set_title('scatter plot between road coverage and slope of the linear equation')
    plt.savefig(str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/scatter_road_coverage.png')

    return res2.slope, res2.intercept

#make the bars
def bars(origGap,approxGap,intercepts, moves, approxGapModel = [], string = ""):
    if approxGapModel == []:
        #make some plots 
        _,ax = plt.subplots()

        width = 0.3
        x_original = [x-width/2 for x in range(len(origGap))]
        x_approx = [x+width/2 for x in range(len(approxGap))]

        ax.bar(x_original, origGap, width, label = 'original Gap',color = 'darkslategray')
        ax.bar(x_approx, approxGap, width, label = 'approx Gap',color = 'crimson')
        
        ax.set_xticks(range(len(moves)), moves)
        ax.set_title('Gaps for different zones before and after linear regression')
        ax.set_ylabel('normalized gap')
        ax.legend()
        plt.savefig(str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/gap_bar_chart.png')

        #plot a bar with the intercepts of the linear equations
        intercept_needed = [i[0] for i in intercepts]
        _, ax2 = plt.subplots()
        ax2.bar(moves, intercept_needed, label = "intercepts of the linear equations")
        ax2.set_title('intercepts of the linear equations (first shape)')
        plt.savefig(str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/intercept_bar.png')
    else:
        _,ax3 = plt.subplots()
        width = 0.3
        x_original = [x-width for x in range(len(origGap))]
        x_approx = [x for x in range(len(approxGap))]
        x_approx2 = [x+width for x in range(len(approxGapModel))]


        ax3.bar(x_original, origGap, width, label = 'original Gap',color = 'darkslategray')
        ax3.bar(x_approx, approxGap, width, label = 'approx Gap linear regression',color = 'crimson')
        ax3.bar(x_approx2, approxGapModel, width, label = 'approx Gap own model', color = 'green')
        
        ax3.set_xticks(range(len(moves)), moves)
        ax3.set_title('Original gap vs linear reg Gap vs Model gap')
        ax3.set_ylabel('normalized gap')
        ax3.set_xlabel(string)
        ax3.legend()
        plt.savefig(str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/modelGap.png')


#make the linear eq plots
def equations(slopes, intercepts, move):
    _,ax2 = plt.subplots()
    x = np.linspace(0,10,300)

    for s, i , zone in zip(slopes, intercepts, move):
        for k in range(len(s)):
            y1 = s[k]*x + i[k]
            ax2.plot(x,y1, label = 'Eq for {}th piece of {} od_reg'.format(k,zone))

    ax2.legend()
    ax2.set_ylabel("Y")
    ax2.set_xlabel("X")
    ax2.set_title('Linear equations from the different test zones + split matrices')
    plt.savefig(str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/linear_equations.png')


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
