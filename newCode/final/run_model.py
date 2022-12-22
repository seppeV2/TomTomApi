from data_processing import create_OD_from_info
from operational_functions import get_split_matrices, import_test_case, matrix_to_list, list_to_matrix, calculate_gap_RMSE
import numpy as np

zonings = ['ZoningSmallLeuven', 'BruggeWithoutZeeBrugge', 'Hasselt']
moves = ['LeuvenExternal', 'BruggeExternal', 'HasseltExternal']
original_od, tomtom_od = import_test_case(zonings[0])

def make_model(tomtom_od, zone, network_property='households_statbel'):
    intercepts = [0	,12,24,32,44,50,58]

    property_OD = create_OD_from_info(network_property+'_'+zone+'_dictionary')
    slices, _ = get_split_matrices(property_OD, 7)
    
    approx_matrix = np.zeros(tomtom_od.shape)
    for idx,slice in enumerate(slices):
        tomtom_slice_list = matrix_to_list(tomtom_od,slice)
        approx_list = tomtom_slice_list + intercepts[idx]
        tomtom_slice_approx = list_to_matrix(approx_list, slice)

        approx_matrix = np.add(approx_matrix, tomtom_slice_approx)
        

    return approx_matrix

approx_matrix = make_model(tomtom_od, zonings[0])

print('gap form model = {}'.format(calculate_gap_RMSE(original_od, approx_matrix)/calculate_gap_RMSE(original_od, tomtom_od)))

