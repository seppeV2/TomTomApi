from operational_functions import calculate_gap_RMSE, setup_test_case,calculate_perc_matrix, heatmap,find_optimal_coef,calculate_gap_RMSE
from output_functions import bars, equations, scatters, calculate_model, correlation_analyses, gap_bars, gap_bars_sum, compare_with_splits, residua_in_bars, residua_3D_plot, heatmaps
from regression import simple_linear_reg, explanatory_linear_regression, split_linear_regression, linear_residua_split
import pathlib
import pandas as pd 
import numpy as np
from scipy.stats import pearsonr
from data_processing_properties import create_OD_from_info
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os
from matplotlib.container import BarContainer
np.set_printoptions(suppress=True)


def main():
    zonings = ['ZoningSmallLeuven', 'BruggeWithoutZeeBrugge', 'Hasselt', 'Roeselare']
    moves = ['LeuvenExternal', 'BruggeExternal', 'HasseltExternal','roeselare']
    zonings = ['Roeselare']
    moves = ['roeselare']
    network_properties = ['population_statbel', 'households_statbel', 'cars_statbel']
    methods = ['sum', 'destination']#, 'origin'] # --> origin and sum based are very similar

    cutoff_values = {'trips': [2],
                    network_properties[0]: [2000], 
                    network_properties[1]: [1000], 
                    network_properties[2]: [1000]}

    #decide which analysis you want to run
    run_correlation = False
    explanatory_analyses = False
    gaps_analysis =  True
    gaps_analysis_sum = False
    split_analysis = False
    residua_analysis = True
    simple_linear_residua = False
    gap_percentage = False
    different_approach = True
    total_gap = {}
    summary = ''

    if explanatory_analyses:
        for property in network_properties:
            total_gap[property] = {}
    else:
        total_gap['none'] = {}

    for zone, move in zip(zonings,moves):
        print('Start analysis for zone: %s' % zone)
        # Set up the matrices for this zone and move
        original_od, tomtom_od = setup_test_case(zone, move)

        if gap_percentage:
            gap_percentage_matrix = calculate_perc_matrix(original_od, tomtom_od, zone)
            print(np.sum(gap_percentage_matrix))
        if residua_analysis:
            residua = original_od - tomtom_od
            residua_in_bars(residua, 'bar_residua_{}'.format(zone), zone)
            residua_3D_plot(residua, zone)
        
        
        # Calculate the original Gap (via RMSE)
        original_gap = calculate_gap_RMSE(original_od, tomtom_od)

        # Simple linear regression 
        simple_approx_gap, slope, intercept, approx_od = simple_linear_reg(original_od, tomtom_od, zone)

        summary += 'SIMPLE LINEAR REGRESSION {}\n\nY = {} * X + {}, with gap = {}\n\nSIMPLE LINEAR REGRESSION RESIDUA\n\n'.format(zone, np.round(slope[0],3), np.round(intercept,3), simple_approx_gap)


        if different_approach:
            multiply_fac = find_optimal_coef(original_od, tomtom_od)
            print(multiply_fac)
            approx_2 = tomtom_od * multiply_fac
            
            approx_residua = original_od - approx_od
            heatmap(approx_residua, zone,  'approx_residual',  fileName='approx_residual_{}'.format(zone))



            approx_residua_2 = original_od - approx_2
            approx_residua_2 = np.array(approx_residua_2).astype(float)
            approx_coef = calculate_gap_RMSE(original_od, approx_residua_2)
            heatmap(approx_residua_2, zone,  'approx_residual',addiTitle = ' average values = {}'.format(np.average(approx_residua_2)), fileName='approx_2_residual_{}'.format(zone))

            residua_3D_plot(approx_residua_2, zone, extraName = '_OwnCoef')




        if not explanatory_analyses:
            total_gap['none'][move]  = [(original_gap, 'original')]
            total_gap['none'][move].append((simple_approx_gap, 'simple reg.'))
            total_gap['none'][move].append((approx_coef, 'approx coef'))
            # total_gap['none'][move].append((simple_approx_gap_norm, 'simple reg. (norm)'))
        else:
            summary += 'EXPLANATORY LINEAR REGRESSION\n\n'

        for network_property in network_properties:
            if explanatory_analyses:
                total_gap[network_property][move] = [(original_gap, 'original')]
                total_gap[network_property][move].append((simple_approx_gap, 'Reg. simple'))
                # total_gap[network_property][move].append((simple_approx_gap_norm, 'Reg. simple (norm)'))
            for method in methods:
                if explanatory_analyses:
                    # Do linear regression with explanatory variables
                    explanatory_approx_gap, slope, intercept = explanatory_linear_regression(original_od, tomtom_od, zone, network_property ,method)
                    summary += '{} for {} with method {}\nY = {} * X1 + {} * X2 + {}\n\n'.format(zone,network_property, method, np.round(slope[0],3), np.round(slope[1],3),np.round(intercept,3))
                    total_gap[network_property][move].append((explanatory_approx_gap, 'Reg. {} {}'.format(network_property.split('_')[0], method)))
                
                # correlation analysis
                if run_correlation:
                    correlation_analyses(original_od, tomtom_od, network_property, zone, method)

                if  split_analysis and method == 'sum' and network_property == network_properties[0]:
                    approx_gap, string, summary_split, shapes, _ = split_linear_regression(original_od, tomtom_od, zone, network_property, method, fixed = True, fixed_jump = cutoff_values[network_property][0])
                    #approx_gap, string, summary_split = split_linear_regression(original_od, tomtom_od, zone, fixed = True, fixed_jump = 2)
                    compare_with_splits(shapes, original_od, tomtom_od, network_property, zone)
                    
                    if  explanatory_analyses:
                        total_gap[network_property][move].append((approx_gap, string))
                    else:
                        total_gap['none'][move].append((approx_gap, string))
                    summary += summary_split+'Approx gap after split = {}\n\n'.format(approx_gap)

                if simple_linear_residua and method == 'sum':
                    approx_gap, string, summary_split, shapes = linear_residua_split(original_od, tomtom_od, network_property, zone, move)

                    if not explanatory_analyses:
                        total_gap['none'][move].append((approx_gap, 'residua reg. {} splitted'.format(network_property)))
                    else:
                        total_gap[network_property][move].append((approx_gap, 'residua reg. {} splitted'.format(network_property)))
                    summary += summary_split
                    summary += 'Gap = {}'.format(approx_gap)

    # Perform the gap function
    if gaps_analysis:
        if not explanatory_analyses:
            gap_bars(total_gap, moves, ['none'])
        else:
            gap_bars(total_gap, moves, network_properties)

    if gaps_analysis_sum:
        gap_bars_sum(total_gap, moves, network_properties)      
    
    print(summary)

main()