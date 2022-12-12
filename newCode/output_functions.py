from operational_functions import heatmaps, outliers, od_matrix_from_tomtom, calculate_gap,matrix_to_list, normalize, get_split_matrices, list_to_matrix, visualize_splits
import numpy as np
import scipy
import pandas as pd 
import pathlib
import matplotlib.pyplot as plt

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

