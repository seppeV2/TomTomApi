from dyntapy.demand_data import od_matrix_from_dataframes
import pandas as pd
import geopandas as gpd
import pathlib
from math import isnan
import numpy as np


""" def get_od_matrix_from_database():

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
 """

def map_zones_onto_extra_info_statbel(infoFile):
    print('Start with {}'.format(infoFile))
    information = pd.read_csv(str(pathlib.Path(__file__).parent)+'/data/statbel/rawData/{}.csv'.format(infoFile))
    print(information.head())
    info_stasec = 'CD_SECTOR'
    mapping = pd.read_csv(str(pathlib.Path(__file__).parent)+'/data/statbel/rawData/KULZone_StatZone.csv')
    print(mapping.head())
    mappingZone = 'ZONENUMMER'
    mappingStasec = 'CS01012021'
    mappingColumn = []
    print('information length = {}'.format(len(information)))
    print('mapp length = {}'.format(len(mapping)))
    for i in range(len(information)):
        print('{} of {} done\t\t {}%\t\t {} left\t\t'.format(i+1, len(information), round((i+1)/len(information)*100,3),len(mapping)))
        info = information.iloc[i]
        if len(mapping) == 0:
            mappingColumn.append(None)
        else:
            for j in range(len(mapping)):
                statZone = mapping.iloc[j]
                if info[info_stasec] == statZone[mappingStasec]:
                    mappingColumn.append(statZone[mappingZone])
                    mapping = mapping.drop(j)
                    mapping.dropna(inplace=True)
                    mapping.reset_index(drop=True, inplace=True)    
                    break
                if j == len(mapping)-1:
                    mappingColumn.append(None)
        print('last item = {}'.format(mappingColumn[-1]))   
    information[mappingZone] = mappingColumn
    information.to_csv(str(pathlib.Path(__file__).parent)+'/data/statbel/{}.csv'.format(infoFile))
    print(information.head())
    print('Done\n')

def build_dic_zones_extra_info(extraInfoName, zoneName, infoName):
    zones = gpd.read_file(str(pathlib.Path(__file__).parents[1])+'/AllZonings/{}.zip'.format(zoneName))
    extraInfo = pd.read_csv(str(pathlib.Path(__file__).parent)+'/data/statbel/{}.csv'.format(extraInfoName))
    name = "ZONENUMMER"
    store = dict()
    print('START\n')
    for idx in range(len(zones)):
        zone = zones.iloc[idx][name]
        store[zone] = 0
    print(store)
    print('setup Done\n')
    for i in range(len(extraInfo)):
        print('{}% done'.format(round((i+1)/len(extraInfo)*100,3)))
        info = extraInfo.iloc[i]
        zoneNumber = info[name]
        if not isnan(info[name]) and str(int(zoneNumber)) in store.keys() :
            store[str(int(zoneNumber))] += info[infoName]

    result = pd.DataFrame.from_dict(store,orient='index')
    result.to_csv(str(pathlib.Path(__file__).parent)+'/data/statbel/{}_{}_{}_dictionary.csv'.format(extraInfoName, infoName, zoneName))

# From the properties csv file (zone number and property information)
# An OD file is made according the different method (that is a variable of the function)
# Sum = (O + D), Average = (O + D)/2, Destination (based) = (D) (with O, D = property value of origin and destination)
def create_OD_from_info(fileName, mergeWay= 'sum'):
    # Head of the data pandas frame
    dataHead = ['zoneName', 'amount']
    data = pd.read_csv(str(pathlib.Path(__file__).parent)+'/data/statbel/{}.csv'.format(fileName), names = dataHead)
    OD = np.zeros((len(data),len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            if i != j:
                if mergeWay == 'sum':
                    OD[i,j] += data['amount'][i] + data['amount'][j]
                elif mergeWay == 'average':
                    OD[i,j] += (data['amount'][i] + data['amount'][j])/2
                elif mergeWay == 'destination':
                    OD[i,j] += data['amount'][j]
                elif mergeWay == 'origin':
                    OD[i,j] += data['amount'][j]
    return OD


""" zonings = ['ZoningSmallLeuven', 'BruggeWithoutZeeBrugge', 'Hasselt']
info = ['households_cars_statsec_2021','households_cars_statsec_2021', 'population_per_stasec']
infoName = ['total_huisH', 'total_wagens','TOTAL']
for i in range(len(zonings)):
    for j in range(len(zonings)):
        build_dic_zones_extra_info(info[i], zonings[j], infoName[i])
 """
    
def create_landuse_list(fileName):
    df = pd.read_csv(str(pathlib.Path(__file__).parent) + '/data/poidby_landuse/rawdata/landuse_{}.csv'.format(fileName))
    landUse = {}
    color_code = {  'industrial':1,
                    'commercial':2,
                    'residential':3,
                    'retail':4,
                    }
    for i in range(len(df)):
        line = df.iloc[i]
        try:
            landuse_list = landUse[line['ZONENUMMER']]
            added = False
            for idx, (l,a) in enumerate(landuse_list):
                if l == line['landuse']:
                    landUse[line['ZONENUMMER']].pop(idx)
                    landUse[line['ZONENUMMER']].append((line['landuse'], line['area_calc']+a))
                    added = True
            if not added:
                    landUse[line['ZONENUMMER']].append((line['landuse'], line['area_calc']))
        except KeyError:
            landUse[line['ZONENUMMER']] = [(line['landuse'], line['area_calc'])]            
    
    keys = landUse.keys()
    final_dic = {}
    for key in keys:
        first = True
        for use in landUse[key]:
            (l,a) = use
            if first:
                (largest_l, largest_a) = (l,a)
                first = False
            else:
                if a > largest_a:
                    (largest_l, largest_a) = (l,a)
        try:
            final_dic[key] = color_code[largest_l]
        except KeyError:
            final_dic[key] = 0
    result = pd.DataFrame.from_dict(final_dic,orient='index')
    result.to_csv(str(pathlib.Path(__file__).parent)+'/data/poidby_landuse/landuse_{}_dictionary.csv'.format(fileName))

""" 
zonings = ['ZoningSmallLeuven', 'BruggeWithoutZeeBrugge', 'Hasselt']

for zone in zonings:
    create_landuse_list(zone)

 """
