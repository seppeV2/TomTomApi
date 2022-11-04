import numpy as np
#import pandas as pd
# import pathlib
import seaborn as sns
import matplotlib.pyplot as plt
import main

#od_mow = pd.read_csv(str(pathlib.Path(__file__).parent)+"/data/results/odMatrixResult.csv")
#print(od_mow)
od = main.setup_test_case()
od_mow = od[0]
od_tomtom = od[1]
print(od_tomtom)

sum_mow = np.sum(np.sum(od_mow))
sum_tomtom = np.sum(np.sum(od_tomtom))

print(sum_mow)
print(sum_tomtom)

norm_od_mow = np.divide(od_mow,sum_mow)
norm_od_tomtom = np.divide(od_tomtom,sum_tomtom)

#heatmap_mow = sns.heatmap(norm_od_mow)
#plt.show()

heatmap_tomtom = sns.heatmap(norm_od_tomtom)
plt.show()
