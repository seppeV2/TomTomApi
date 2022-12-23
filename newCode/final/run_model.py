from data_processing import create_OD_from_info
from operational_functions import get_split_matrices, import_test_case, matrix_to_list, list_to_matrix, calculate_gap_RMSE
import numpy as np

zonings = ['ZoningSmallLeuven', 'BruggeWithoutZeeBrugge', 'Hasselt']
moves = ['LeuvenExternal', 'BruggeExternal', 'HasseltExternal']
original_od, tomtom_od = import_test_case(zonings[0])

def make_model(tomtom_od, original_od, zone, network_property='households_statbel'):
    slopes = [0,2.4,0.8,0.27,0.33,0.13,-0.22,-0.52]
    intercepts = [0	,2.6,13.4,23.7,33.9,44.6,53,65]

    property_OD = create_OD_from_info(network_property+'_'+zone+'_dictionary')
    slices, _ = get_split_matrices(property_OD, 8)
    approx_matrix = np.zeros(tomtom_od.shape)

    for idx,slice in enumerate(slices):
        if idx != 7:
            tomtom_slice_list = matrix_to_list(tomtom_od,slice)
            original_list = matrix_to_list(original_od,slice)

            approx_list = slopes[idx] * tomtom_slice_list + intercepts[idx]
            print('slice {}'.format(idx))
            print('max value tom/original = {}'.format(np.max(np.subtract(original_list,tomtom_slice_list))))
            print('max value tom/approx = {}'.format(np.max(np.subtract(original_list,approx_list))))


            tomtom_slice_approx = list_to_matrix(approx_list, slice)
            approx_matrix = np.add(approx_matrix, tomtom_slice_approx)
        else:
            approx_matrix = np.add(approx_matrix, slice * tomtom_od)
    return approx_matrix

approx_matrix = make_model(tomtom_od, original_od,zonings[0])
print('gap form model = {}'.format(calculate_gap_RMSE(original_od, approx_matrix)/calculate_gap_RMSE(original_od, tomtom_od)))

