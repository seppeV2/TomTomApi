from operational_functions import heatmaps, outliers, od_matrix_from_tomtom, calculate_gap,matrix_to_list, normalize, get_split_matrices, list_to_matrix, visualize_splits
from data_processing_properties import create_OD_from_info
import numpy as np
import scipy
import pandas as pd 
import pathlib
import matplotlib.pyplot as plt
import os
from scipy.stats import pearsonr

# function to make heatmaps between the diff of the original matrix and the tomtom matrix
# and the property matrix with it's method of merging the property (OD) eg 'sum'
# the results are stored in a folder, the pearson correlation factor is displayed on the graphs as well.
def correlation_analyses(original_od, tomtom_od, network_property, zone, method ):
    residua = np.subtract(original_od, tomtom_od)
    path0 = str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/Correlation_analyses/absolute/{}/{}'.format(method,zone)
    path1 = str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/Correlation_analyses/normalized/{}/{}'.format(method,zone)
    os.makedirs(path0, exist_ok=True)
    os.makedirs(path1, exist_ok=True)
    property_matrix = create_OD_from_info(network_property+'_'+zone+'_dictionary', method)
    correlation, _ = pearsonr(matrix_to_list(residua), matrix_to_list(property_matrix))
    heatmaps(residua, property_matrix, zone, 'Residu Matrix', network_property, 'pearson correlation factor = {}'.format(round(np.average(correlation),3)), network_property, path0)
    heatmaps(residua/np.max(residua), property_matrix/(np.max(property_matrix)), zone, 'Residu Matrix', network_property, 'pearson correlation factor = {}, (normalized)'.format(round(np.average(correlation),3)), network_property, path1)
    fig,_ = plt.subplots(1,1)
    fig.suptitle('Scatter residu vs {}'.format(network_property))
    plt.scatter(matrix_to_list(residua),matrix_to_list(property_matrix))
    plt.ylabel('{}'.format(network_property))
    plt.xlabel('Residu')
    plt.savefig(path0+'/scatter_residu_vs_{}'.format(network_property))
    plt.close()

    fig,_ = plt.subplots(1,1)
    fig.suptitle('Scatter residu vs {} (normalized)'.format(network_property))
    plt.scatter(matrix_to_list(residua)/np.max(matrix_to_list(residua)),matrix_to_list(property_matrix)/np.max(matrix_to_list(property_matrix)))
    plt.ylabel('Normalized {}'.format(network_property))
    plt.xlabel('Normalized Residu')
    plt.savefig(path1+'/scatter_residu_vs_{}'.format(network_property))
    plt.close()
   
#functiont to calculate_model
def calculate_model(intercepts, modelSlope, modelIntercept, moves, road_coverage, tomtomData, shapes_dic, approxOD, originalData):
    averageModelIntercept = np.average([i[0] for i in intercepts])
    print("Model to build MOW: [({} * road_coverage + {}) * tomtom_od + {}] * number_of_trips".format(modelSlope, modelIntercept, averageModelIntercept))
    string = "Model used: [({} * road_coverage + {}) * tomtom_od + {}] * number_of_trips".format(modelSlope, modelIntercept, averageModelIntercept)
    approx_gap2 = []
    for idx, move in enumerate(moves): 
        newMatrix = (((modelSlope * road_coverage[idx]) + modelIntercept) * tomtomData[move])*shapes_dic[move][0] + approxOD[move]*shapes_dic[move][1]
        approx_gap2.append(calculate_gap(originalData[move], newMatrix))
    return approx_gap2, string

#make the linear eq plots
def equations(slopes, intercepts, move, path):
    x = np.linspace(0,10,300)
    for i in range(len(slopes[0])):
        _,ax = plt.subplots()
        for j in range(len(slopes)):
            y1 = slopes[j][i]*x + intercepts[j][i]
            ax.plot(x,y1, label = 'Eq for {}th split of OD {}'.format(i+1,move[j]))
        ax.legend()
        ax.set_ylabel("Y")
        ax.set_xlabel("X")
        ax.set_title('Linear equations from the {}th split of the matrix'.format(i+1))
        plt.savefig(path+'/linear_equations_split_{}.png'.format(i+1))

def gap_bars_sum(gaps, moves, properties):

    path = str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/general_info/sum_specific_gaps/'
    os.makedirs(path, exist_ok=True)

    amount_of_bars = len(gaps) + 2

    # color of the bars
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    orig_gap = np.array([gaps[properties[0]][move][0][0] for move in moves])
    # width of the bar
    width = 0.15
    
    if amount_of_bars%2 == 1:
        # uneven
        start = -(amount_of_bars//2) * width
    else:
        # even
        start = -(amount_of_bars//2 + 0.5) * width

    offset = [start + x * width for x in range(amount_of_bars)]
    x_original = [x+offset[0] for x in range(len(moves))]

    simp_gap = np.array([gaps[properties[0]][move][1][0] for move in moves])
    x_simp = [x+offset[1] for x in range(len(moves))]

    _,ax1 = plt.subplots()
    ax1.bar(x_original, orig_gap, width, label = 'Original', color = colors[0])
    ax1.bar(x_simp, simp_gap, width, label = 'simple Reg', color = colors[1])

    # Only save the third element (as that is the sum)
    for idx, property in enumerate(properties): 
        gaps_list = []
        for move in moves:
            (gap, legend) = gaps[property][move][2]
            gaps_list.append(gap)
        x_gap = [x+offset[idx+2] for x in range(len(moves))]
        ax1.bar(x_gap, gaps_list ,width, label = legend, color = colors[idx+2])

    ax1.set_xticks(range(len(moves)), moves)
    ax1.set_title('Bars absolute gap per zone.')
    ax1.set_ylabel('Absolute gap')
    ax1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',prop={'size': 6})
    plt.tight_layout()
    plt.savefig(path+'gap_bar_absolute_sum.png')
    plt.close()

    orig_norm_gap = np.full(orig_gap.shape, 1)
    simp_gap_norm = np.array([gaps[properties[0]][move][1][0]/orig_gap[idx] for idx, move in enumerate(moves)])

    _,ax1 = plt.subplots()
    ax1.bar(x_original, orig_norm_gap, width, label = 'Original', color = colors[0])
    ax1.bar(x_simp, simp_gap_norm, width, label = 'simple Reg', color = colors[1])

    # Only save the third element (as that is the sum)
    for idx, property in enumerate(properties): 
        gaps_list = []
        for i, move in enumerate(moves):
            (gap, legend) = gaps[property][move][2]
            gaps_list.append(gap/orig_gap[i])
        x_gap = [x+offset[idx+2] for x in range(len(moves))]
        ax1.bar(x_gap, gaps_list, width, label = legend, color = colors[idx+2])

    ax1.set_xticks(range(len(moves)), moves)
    ax1.set_title('Bars normalized gap per zone.')
    ax1.set_ylabel('Normalized gap')
    ax1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',prop={'size': 6})
    plt.tight_layout()
    plt.savefig(path+'gap_bar_normalized_sum.png')
    plt.close()



# this function makes a plot of the gaps between 
def gap_bars(gaps, moves, properties):
    for property in properties:
        path = str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/general_info/{}/'.format(property)
        os.makedirs(path, exist_ok=True)

        # color of the bars
        colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
        orig_gap = np.array([gaps[property][move][0][0] for move in moves])
        # width of the bar
        width = 0.15
        
        if len(gaps[property][moves[0]])%2 == 1:
            # uneven
            start = -(len(gaps[property][moves[0]])//2) * width
        else:
            # even
            start = -(len(gaps[property][moves[0]])//2 + 0.5) * width

        offset = [start + x * width for x in range(len(gaps[property][moves[0]]))]
        x_original = [x+offset[0] for x in range(len(moves))]

        _,ax1 = plt.subplots()

        ax1.bar(x_original, orig_gap, width, label = 'Original', color = colors[0])

        for i  in range(1,len(gaps[property][moves[0]])):
            gap_list = []
            for idx, move in enumerate(moves):
                (gap, legend) = gaps[property][move][i]
                legend_string = legend
                gap_list.append(gap)    
            x_gap = [x+offset[i] for x in range(len(gap_list))]
            ax1.bar(x_gap, gap_list, width, label = legend_string,color = colors[i])
        
        ax1.set_xticks(range(len(moves)), moves)
        ax1.set_title('Bar charts of the absolute gaps from the different zones.')
        ax1.set_ylabel('Absolute gap')
        ax1.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',prop={'size': 6})
        plt.tight_layout()
        plt.savefig(path+'gap_bar_absolute.png')
        plt.close()

        #do the same for the normalized gaps
        orig_norm_gap = np.full(orig_gap.shape, 1)
        x_original = [x+offset[0] for x in range(len(orig_gap))]

        _,ax2 = plt.subplots()
        ax2.bar(x_original, orig_norm_gap, width, label = 'Original', color = colors[0])

        for i in range(1,len(gaps[property][moves[0]])):
            norm_gap_list = []
            legend_string = ""
            for idx, move in enumerate(moves):
                (gap, legend) = gaps[property][move][i]
                legend_string = legend
                norm_gap_list.append(gap/orig_gap[idx])   
            x_gap = [x+offset[i] for x in range(len(norm_gap_list))]
            ax2.bar(x_gap, norm_gap_list, width, label = legend,color = colors[i])

        ax2.set_xticks(range(len(moves)), moves)
        ax2.set_title('Bar charts of the normalized gaps from the different zones.')
        ax2.set_ylabel('Normalized gap')
        ax2.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left',prop={'size': 6})
        plt.tight_layout()        
        plt.savefig(path+'gap_bar_normalized.png')
        plt.close()
        
        

#make the bars
def bars(origGap,approxGap,intercepts, moves, path, approxGapModel = [], string = ""):
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
        plt.savefig(path+'/gap_bar_chart.png')
        for i in range(len(intercepts[0])):
            #plot a bar with the intercepts of the linear equations
            intercept_needed = [j[i] for j in intercepts]
            _, ax2 = plt.subplots()
            ax2.bar(moves, intercept_needed, label = "intercepts of the linear equations")
            ax2.set_title('intercepts of the linear equations ({}th shape)'.format(i+1))
            plt.savefig(path+'/intercept_bar_{}th.png'.format(i+1))
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
        plt.savefig(path+'/modelGap.png')

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


def compare_with_splits(shapes, origin_od, tomtom_od, network_property, zone):
    residua = origin_od - tomtom_od
    explanatory_od = create_OD_from_info(network_property+'_'+zone+'_dictionary')
    path1 = str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/split_analysis/{}'.format(zone)

    for idx, shape in enumerate(shapes): 
        heatmaps(explanatory_od*shape, residua*shape, zone, '{}_split{}'.format(network_property, idx), 'Residua_split{}'.format(idx),fileName='{}_split_{}'.format(network_property, idx),path=path1)
