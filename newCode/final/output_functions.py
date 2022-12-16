from math import sqrt
import pathlib 
import numpy as np
import pandas as pd
from sympy import *
from scipy.stats import linregress 
import matplotlib.pyplot as plt
import seaborn as sns
from dyntapy.demand_data import od_matrix_from_dataframes
import geopandas as gpd
import os
from sklearn.linear_model import LinearRegression 

def bar_outputs(gaps, gaps_norm, names, names_norm, moves):
    path = str(pathlib.Path(__file__).parent) + '/plots/bar_gaps/'
    os.makedirs(path, exist_ok=True)

    # color of the bars
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']
    orig_gap = np.array([gaps[move][0] for move in moves])
    # width of the bar
    width = 0.15
    
    if len(gaps[moves[0]])%2 == 1:
        # uneven
        start = -(len(gaps[moves[0]])//2) * width
    else:
        # even
        start = -(len(gaps[moves[0]])//2 + 0.5) * width

    offset = [start + x * width for x in range(len(gaps[moves[0]]))]
    x_original = [x+offset[0] for x in range(len(moves))]
    _,ax1 = plt.subplots()

    ax1.bar(x_original, orig_gap, width, label = names[moves[0]][0], color = colors[0])
    j=0
    for i  in range(1,len(gaps[moves[0]])):
        gap_list = []
        for idx, move in enumerate(moves):
            gap = gaps[move][i]
            gap_list.append(gap)    
        x_gap = [x+offset[i] for x in range(len(gap_list))]
        j += -len(colors)+1 if i==len(colors) else 1
        ax1.bar(x_gap, gap_list, width, label = names[moves[0]][i],color = colors[j])
    
    ax1.set_xticks(range(len(moves)), moves)
    ax1.set_title('Bar charts of the absolute gaps from the different zones.')
    ax1.set_ylabel('Absolute gap')
    ax1.legend(prop={'size': 8})
    plt.tight_layout()
    plt.savefig(path+'gap_bar_absolute.png')
    plt.close()

    orig_norm_gap = np.array([gaps_norm[move][0] for move in moves])
    _,ax2 = plt.subplots()

    ax2.bar(x_original, orig_norm_gap, width, label = names_norm[moves[0]][0], color = colors[0])
    j=0
    for i  in range(1,len(gaps_norm[moves[0]])):
        gap_list_norm = []
        for idx, move in enumerate(moves):
            gap = gaps_norm[move][i]
            gap_list_norm.append(gap)    
        x_gap = [x+offset[i] for x in range(len(gap_list))]
        j += -len(colors)+1 if i==len(colors) else 1
        ax2.bar(x_gap, gap_list_norm, width, label = names_norm[moves[0]][i],color = colors[j])
    
    ax2.set_xticks(range(len(moves)), moves)
    ax2.set_title('Bar charts of the normalized gaps from the different zones.')
    ax2.set_ylabel('normalized gap')
    ax2.legend(prop={'size': 8})
    plt.tight_layout()
    plt.savefig(path+'gap_bar_normalized.png')
    plt.close()

def heatmaps(matrix1, matrix2, zone, name1, name2, addiTitle='' , fileName='', path = ''):
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('{} {}'.format(zone, addiTitle))
    sns.heatmap(matrix2, ax=ax[0]).set(title=name2)
    sns.heatmap(matrix1, ax=ax[1]).set(title=name1)
    if path == '':
        path = str(pathlib.Path(__file__).parents[1])+'/graphsFromResults/general_info/heatmaps'
        os.makedirs(path, exist_ok=True)
    if fileName != '':
        plt.savefig(path+'/heatmap_{}.png'.format(fileName))
    else: 
        plt.savefig(path+'/heatmap_{}.png'.format(zone))
    plt.close()
   
def visualize_splits(shapes, zone, path):
    amount_shapes = len(shapes)
    extra = 0 if len(shapes)%2 == 0 else 1
    for i in range(int(amount_shapes//2)+extra):
        fig, ax = plt.subplots(1,2)
        fig.suptitle(zone)
        sns.heatmap(shapes[i*2],ax=ax[0]).set(title='shape_{}'.format(i*2+1))
        if (i*2)+1 < len(shapes):
            sns.heatmap(shapes[i*2+1],ax=ax[1]).set(title='shape_{}'.format(i*2+2))
        plt.savefig(path+'/visual_shapes_{}_{}.png'.format(zone, i+1))  
        plt.close()
    