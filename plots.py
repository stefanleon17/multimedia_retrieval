from cmath import log
from statistics import stdev

import pandas as pd
import matplotlib.pyplot as plt
from numpy.ma.extras import average

df = pd.read_csv("obj_analysis.csv")
# logdf = {log(i) for i in df["Vertex Count"]}
df2 = df[df["Vertex Count"] > 10000]["Vertex Count"]
plt.hist(df["Vertex Count"], bins=100, color="skyblue", edgecolor="blue")
plt.xlabel("Vertex Count")
plt.xlim(0, None)
plt.ylabel("Frequency")
plt.show()

vertex_avg = average(df["Vertex Count"])
print(vertex_avg)
print(stdev(df["Vertex Count"]))

