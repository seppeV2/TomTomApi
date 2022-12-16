from operational_functions import import_test_case, calculate_gap_RMSE
from regression_functions import simple_linear_reg, linear_residua_split
from output_functions import bar_outputs
import numpy as np



zonings = ['ZoningSmallLeuven', 'BruggeWithoutZeeBrugge', 'Hasselt']
moves = ['LeuvenExternal', 'BruggeExternal', 'HasseltExternal']
network_properties = ['population_statbel', 'households_statbel', 'cars_statbel']

method = 'sum'

slices_values = {}
gaps = {}
gaps_norm = {}
names = {}
names_norm = {}
bar_analysis = True
all_slopes = []
all_intercepts = []
all_ranges = []
for zone, move in zip(zonings, moves):
    # Import the od matrices
    original_od, tomtom_od = import_test_case(zone)

    # Calculate the original Gap (via RMSE)
    original_gap = calculate_gap_RMSE(original_od, tomtom_od)
    gaps[move] = [original_gap]
    gaps_norm[move] = [1]
    names[move] = ['original gap']
    names_norm[move] = ['Norm. original gap']

    # Simple regression
    simple_approx_gap, slope, intercept = simple_linear_reg(original_od, tomtom_od)
    gaps[move].append(simple_approx_gap)
    gaps_norm[move].append(simple_approx_gap/gaps[move][0])
    names[move].append('simp. reg. gap')
    names_norm[move].append('Norm. simp. reg. gap')

    for network_property in network_properties:
        approx_gap, slopes, intercepts, ranges = linear_residua_split(original_od, tomtom_od, network_property, zone)
        gaps[move].append(approx_gap)
        gaps_norm[move].append(approx_gap/gaps[move][0])
        names[move].append('split. {}_{} reg.'.format(network_property.split('_')[0], zone))
        names_norm[move].append('Norm. split. {}_{} reg.'.format(network_property.split('_')[0], zone))

        all_slopes.append(slopes)
        all_intercepts.append(intercepts)
        all_ranges.append(ranges)

bar_outputs(gaps, gaps_norm, names, names_norm, moves)
print(all_ranges)