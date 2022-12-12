from operational_functions import heatmaps, outliers, od_matrix_from_tomtom, calculate_gap,matrix_to_list, normalize, get_split_matrices, list_to_matrix, visualize_splits, setup_test_case
from output_functions import bars, equations, scatters, calculate_model
import pathlib
import pandas as pd 
import numpy as np
import scipy
from data_processing_properties import create_OD_from_info
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
from matplotlib.container import BarContainer
np.set_printoptions(suppress=True)


def main():
    zonings = ['ZoningSmallLeuven', 'BruggeWithoutZeeBrugge', 'Hasselt']
    moves = ['LeuvenExternal', 'BruggeExternal', 'HasseltExternal']
    cluster_ways = ['population_per_stasec_TOTAL', 'households_cars_statsec_2021_total_huisH', 'households_cars_statsec_2021_total_wagens']
    methods = ['sum']#, 'average']

        #amount of km road in the zone
    road_coverage = [997.67,1069.56,1487.82]
        #squared km 
    area = [57.52,95.36,102.53]
    #cutoffValues = {cluster_ways[0]:[5000,10000]}
    

    for cluster_way in cluster_ways:
        for method in methods:
            origGap = []
            approxGap = []
            slopes = []
            intercepts = []

            plot = True
            heatmap = True

            tomtomData = {}
            originalData = {}
            shapes_dic = {}
            approxOD = {}
            for zone, move in zip(zonings, moves):

                path = str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/{}/{}'.format(cluster_way, method)
                path1 = str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/{}/{}/{}'.format(cluster_way, method, zone)

                os.makedirs(path1, exist_ok=True)
                print("START SETUP FOR {}\n".format(move))
                original_od, tomtom_od = setup_test_case(zone, move)
                difference = original_od - tomtom_od
                #normalize the matrices
                original_od, ori_norm = normalize(original_od)
                tomtom_od,tom_norm= normalize(tomtom_od) 
                norm,_ = normalize(difference)

                #norm_fact = ori_norm/tom_norm

                tomtomData[move] = tomtom_od
                originalData[move] = original_od
                
                cluster_matrix = create_OD_from_info(cluster_way+'_'+zone+'_dictionary', method)
                shapes = get_split_matrices(cluster_matrix,3)
                shapes_dic[move] = shapes

                print("START LINEAR REGRESSION\n")
                slopes_one_zone = []
                intercepts_one_zone = []
                loop = 0
                

                if heatmap:
                    heatmaps(original_od, tomtom_od, zone, 'original', 'tomtom', path1)  
                    visualize_splits(shapes, zone ,path1)
                    #outliers(norm, zone)

                approx_matrix = np.zeros(tomtom_od.shape)
                for i  in range(len(shapes)):
                    if np.sum(shapes[i]) == 0:
                        slopes_one_zone.append(0)
                        intercepts_one_zone.append(0)
                    else:
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



            if plot:
            
            #plot the bar plots
                bars(origGap,approxGap, intercepts, moves, path)

                #plot the equations
                equations(slopes, intercepts, move, path)    
                

                #plot the scatters (with their linear fit)
                #modelSlope, modelIntercept = scatters(slopes_first_shape ,area, road_coverage)
                #approx_gap2, string = calculate_model(intercepts, modelSlope, modelIntercept, moves, road_coverage, tomtomData, shapes_dic, approxOD, originalData)

main()
