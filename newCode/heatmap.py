import numpy as np
import pandas as pd
import pathlib
import seaborn as sns
import matplotlib.pyplot as plt

od_mow = pd.read_csv(str(pathlib.Path(__file__).parent)+"/data/results/odMatrixResult.csv")
print(od_mow)
sum = np.sum(np.sum(od_mow))
print(sum)

norm_od_mow = od_mow / sum
print(norm_od_mow)

heatmap_mow = sns.heatmap(norm_od_mow)
plt.show()