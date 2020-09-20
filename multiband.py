import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pyvis
from pyvis.network import Network

# Import precision matrices and create abs value version
mb2 = np.load("/Users/sohailnizam/Desktop/mb2.npy")
mb2_abs = np.absolute(mb2)

mb3 = np.load("/Users/sohailnizam/Desktop/mb3.npy")
mb3_abs = np.absolute(mb3)

mb4 = np.load("/Users/sohailnizam/Desktop/mb4.npy")
mb4_abs = np.absolute(mb4)

mb5 = np.load("/Users/sohailnizam/Desktop/mb5.npy")
mb5_abs = np.absolute(mb5)

mb6 = np.load("/Users/sohailnizam/Desktop/mb6.npy")
mb6_abs = np.absolute(mb6)

mb7 = np.load("/Users/sohailnizam/Desktop/mb7.npy")
mb7_abs = np.absolute(mb7)

mb8 = np.load("/Users/sohailnizam/Desktop/mb8.npy")
mb8_abs = np.absolute(mb8)

mb9 = np.load("/Users/sohailnizam/Desktop/mb9.npy")
mb9_abs = np.absolute(mb9)

# cast matrices and absolute matrices to networkx networks
mb4_graph = nx.from_numpy_array(mb4)
mb4_absgraph = nx.from_numpy_array(mb4_abs)
mb8_graph = nx.from_numpy_array(mb8)
mb8_absgraph = nx.from_numpy_array(mb8_abs)

#### Generate some statistics for these graphs (weighted) ####


### Strength (Weighted Degree) Summaries ###

# Strengths (weighted degrees)
mb4_strengths = np.array(list(dict(mb4_absgraph.degree(weight = 'weight')).values()))
mb8_strengths = np.array(list(dict(mb8_absgraph.degree(weight = 'weight')).values()))

# Mean and SD strength
print(np.mean(mb4_strengths))
print(np.std(mb4_strengths))
print(np.mean(mb8_strengths))
print(np.std(mb8_strengths))

# Five number summary of strength
print(np.quantile(mb4_strengths, 0), np.quantile(mb4_strengths, .25),
      np.quantile(mb4_strengths, .5), np.quantile(mb4_strengths, .75),
      np.quantile(mb4_strengths, 1))

print(np.quantile(mb8_strengths, 0), np.quantile(mb8_strengths, .25),
      np.quantile(mb8_strengths, .5), np.quantile(mb8_strengths, .75),
      np.quantile(mb8_strengths, 1))

# Histogram of Strength Distribution

fig, main_ax = plt.subplots()
main_ax.hist(mb4_strengths)
main_ax.set_xlim(4, 17)
#main_ax.set_ylim(1.1 * np.min(s), 2 * np.max(s))
main_ax.set_xlabel('Weighted Degree')
main_ax.set_ylabel('Frequency')
main_ax.set_title('MB4 Weighted Degree Distribution')
plt.show()


fig, main_ax = plt.subplots()
main_ax.hist(mb8_strengths)
main_ax.set_xlim(4, 17)
#main_ax.set_ylim(1.1 * np.min(s), 2 * np.max(s))
main_ax.set_xlabel('Weighted Degree')
main_ax.set_ylabel('Frequency')
main_ax.set_title('MB8 Weighted Degree Distribution')
plt.show()


### Weighted Average Path ###
print(nx.average_shortest_path_length(mb4_absgraph, weight='weight')) # .0012
print(nx.average_shortest_path_length(mb8_absgraph, weight='weight')) # .0007

### Weighted Average Clustering Coefficient ###
print(nx.average_clustering(mb4_absgraph, weight='weight')) #.0067
print(nx.average_clustering(mb8_absgraph, weight='weight')) #.0051


### Small Worldness ###
#Takes too long to run
#nx.sigma(mb4_absgraph, niter=100, nrand=10, seed=None)

### Heat Maps ###
ax = plt.axes()
sns.heatmap(mb4_abs, ax = ax, cmap = "viridis")
ax.set_title('MB4 Acquisition Adjacency Matrix Heatmap')
plt.show()


ax = plt.axes()
sns.heatmap(mb8_abs, ax = ax, cmap = "viridis")
ax.set_title('MB8 Acquisition Adjacency Matrix Heatmap')
plt.show()

#### Set several thresholds to create unweighted graphs ###

# Threshold 0 #
mb4_t0 = np.where(mb4_abs > 0, 1, 0)
mb8_t0 = np.where(mb8_abs > 0, 1, 0)
mb4_t0_graph = nx.from_numpy_array(mb4_t0)
mb8_t0_graph = nx.from_numpy_array(mb8_t0)


# Threshold .02 (median of mb4) #
mb4_t02 = np.where(mb4_abs > .02, 1, 0)
mb8_t02 = np.where(mb8_abs > .02, 1, 0)
mb4_t02_graph = nx.from_numpy_array(mb4_t02)
mb8_t02_graph = nx.from_numpy_array(mb8_t02)

# Threshold .15 #
mb4_t15 = np.where(mb4_abs > .15, 1, 0)
mb8_t15 = np.where(mb8_abs > .15, 1, 0)
mb4_t15_graph = nx.from_numpy_array(mb4_t15)
mb8_t15_graph = nx.from_numpy_array(mb8_t15)


### Calculate some statistics for the unweighted graphs ###

# Degree values
mb4_t0_deg = np.array(list(dict(mb4_t0_graph.degree()).values()))
mb8_t0_deg = np.array(list(dict(mb8_t0_graph.degree()).values()))

mb4_t02_deg = np.array(list(dict(mb4_t02_graph.degree()).values()))
mb8_t02_deg = np.array(list(dict(mb8_t02_graph.degree()).values()))

mb4_t15_deg = np.array(list(dict(mb4_t15_graph.degree()).values()))
mb8_t15_deg = np.array(list(dict(mb8_t15_graph.degree()).values()))

# Mean and SD Degree
print(np.mean(mb4_t0_deg))
print(np.std(mb4_t0_deg))
print(np.mean(mb8_t0_deg))
print(np.std(mb8_t0_deg))

print(np.mean(mb4_t02_deg))
print(np.std(mb4_t02_deg))
print(np.mean(mb8_t02_deg))
print(np.std(mb8_t02_deg))

print(np.mean(mb4_t15_deg))
print(np.std(mb4_t15_deg))
print(np.mean(mb8_t15_deg))
print(np.std(mb8_t15_deg))

# Five number summary of strength
print(np.quantile(mb4_t0_deg, 0), np.quantile(mb4_t0_deg, .25),
      np.quantile(mb4_t0_deg, .5), np.quantile(mb4_t0_deg, .75),
      np.quantile(mb4_t0_deg, 1))
print(np.quantile(mb8_t0_deg, 0), np.quantile(mb8_t0_deg, .25),
      np.quantile(mb8_t0_deg, .5), np.quantile(mb8_t0_deg, .75),
      np.quantile(mb8_t0_deg, 1))

print(np.quantile(mb4_t02_deg, 0), np.quantile(mb4_t02_deg, .25),
      np.quantile(mb4_t02_deg, .5), np.quantile(mb4_t02_deg, .75),
      np.quantile(mb4_t02_deg, 1))
print(np.quantile(mb8_t02_deg, 0), np.quantile(mb8_t02_deg, .25),
      np.quantile(mb8_t02_deg, .5), np.quantile(mb8_t02_deg, .75),
      np.quantile(mb8_t02_deg, 1))

print(np.quantile(mb4_t15_deg, 0), np.quantile(mb4_t15_deg, .25),
      np.quantile(mb4_t15_deg, .5), np.quantile(mb4_t15_deg, .75),
      np.quantile(mb4_t15_deg, 1))
print(np.quantile(mb8_t15_deg, 0), np.quantile(mb8_t15_deg, .25),
      np.quantile(mb8_t15_deg, .5), np.quantile(mb8_t15_deg, .75),
      np.quantile(mb8_t15_deg, 1))


# Average Path
print(nx.average_shortest_path_length(mb4_t0_graph))
print(nx.average_shortest_path_length(mb8_t0_graph))

print(nx.average_shortest_path_length(mb4_t02_graph))
print(nx.average_shortest_path_length(mb8_t02_graph))

print(nx.average_shortest_path_length(mb4_t15_graph))
print(nx.average_shortest_path_length(mb8_t15_graph))

# Average Clustering Coefficient
print(nx.average_clustering(mb4_t0_graph))
print(nx.average_clustering(mb8_t0_graph))

print(nx.average_clustering(mb4_t02_graph))
print(nx.average_clustering(mb8_t02_graph))

print(nx.average_clustering(mb4_t15_graph))
print(nx.average_clustering(mb8_t15_graph))


### Visualize the unweighted graphs ###

### Heat Maps ###
ax = plt.axes()
sns.heatmap(mb4_t0, ax = ax, cmap = "viridis")
ax.set_title('MB4 Acquisition Binary Adjacency Matrix Heatmap, T = 0')
plt.show()

ax = plt.axes()
sns.heatmap(mb8_t0, ax = ax, cmap = "viridis")
ax.set_title('MB8 Acquisition Binary Adjacency Matrix Heatmap, T = 0')
plt.show()

ax = plt.axes()
sns.heatmap(mb4_t02, ax = ax, cmap = "viridis")
ax.set_title('MB4 Acquisition Binary Adjacency Matrix Heatmap, T = .02')
plt.show()

ax = plt.axes()
sns.heatmap(mb8_t02, ax = ax, cmap = "viridis")
ax.set_title('MB8 Acquisition Binary Adjacency Matrix Heatmap, T = .02')
plt.show()


ax = plt.axes()
sns.heatmap(mb4_t15, ax = ax, cmap = "viridis")
ax.set_title('MB4 Acquisition Binary Adjacency Matrix Heatmap, T = .15')
plt.show()

ax = plt.axes()
sns.heatmap(mb8_t15, ax = ax, cmap = "viridis")
ax.set_title('MB8 Acquisition Binary Adjacency Matrix Heatmap, T = .15')
plt.show()


# Node link Diagram #
# first add brain region as node attributes
region_color = []
for i in range(264):
      if 0 <= i <= 12:
            region_color.append(1)
      elif 13 <= i <= 16:
            region_color.append(2)
      elif 17 <= i <= 46:
            region_color.append(3)
      elif 47 <= i <= 51:
            region_color.append(4)
      elif 52 <= i <= 65:
            region_color.append(5)
      elif 66 <= i <= 78:
            region_color.append(6)
      elif  79 <= i <= 136:
            region_color.append(7)
      elif 137 <= i <= 141:
            region_color.append(8)
      elif 142 <= i <= 172:
            region_color.append(9)
      elif 173 <= i <= 197:
            region_color.append(10)
      elif 198 <= i <= 215:
            region_color.append(11)
      elif 216 <= i <= 224:
            region_color.append(12)
      elif 225 <= i <= 235:
            region_color.append(13)
      else:
            region_color.append(14)




nx.draw(mb4_t15_graph, node_color=region_color, with_labels=False, node_size = 10)
plt.show()

nx.draw(mb8_t15_graph, node_color=region_color, with_labels=False, node_size = 10)
plt.show()

