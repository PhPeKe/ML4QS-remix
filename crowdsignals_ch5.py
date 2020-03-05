##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 5                                               #
#                                                            #
##############################################################
import sys
sys.path.append(".")
from util.VisualizeDataset import VisualizeDataset
from Chapter5.DistanceMetrics import InstanceDistanceMetrics
from Chapter5.DistanceMetrics import PersonDistanceMetricsNoOrdering
from Chapter5.DistanceMetrics import PersonDistanceMetricsOrdering
from Chapter5.Clustering import NonHierarchicalClustering
from Chapter5.Clustering import HierarchicalClustering
import copy
import pandas as pd
import util.util as util
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plot
# Of course we repeat some stuff from Chapter 3, namely to load the dataset

#target = ['gyr_phone_x', 'gyr_phone_y', 'gyr_phone_z']
target = ['acc_phone_x', 'acc_phone_y', 'acc_phone_z']

DataViz = VisualizeDataset()

# Read the result from the previous chapter, and make sure the index is of the type datetime.
dataset_path = 'PythonCode/intermediate_datafiles/'

try:
    dataset = pd.read_csv(dataset_path + 'chapter4_ownresult.csv', index_col=0)
except IOError as e:
    print('File not found, try to run previous crowdsignals scripts first!')
    raise e
dataset.index = pd.to_datetime(dataset.index)

# First let us use non hierarchical clustering.

clusteringNH = NonHierarchicalClustering()

# Let us look at k-means first.

k_values = range(2, 10)
silhouette_values = []
#
## Do some initial runs to determine the right number for k
#
print '===== kmeans clustering ====='
for k in k_values:
    print 'k = ', k
    dataset_cluster = clusteringNH.k_means_over_instances(copy.deepcopy(dataset), target, k, 'default', 20, 10)
    silhouette_score = dataset_cluster['silhouette'].mean()
    print 'silhouette = ', silhouette_score
    silhouette_values.append(silhouette_score)

plot.plot(k_values, silhouette_values, 'b-')
plot.xlabel('k')
plot.ylabel('silhouette score')
plot.ylim([0,1])
#plot.show()
plot.savefig("ownkmeans.png")
# And run the knn with the highest silhouette score

k = 6

dataset_knn = clusteringNH.k_means_over_instances(copy.deepcopy(dataset), target, k, 'default', 50, 50)
plot = DataViz.plot_clusters_3d(dataset_knn, target, 'cluster', ['label'])
plot.savefig("own3d.png")
plot = DataViz.plot_silhouette(dataset_knn, 'cluster', 'silhouette')
plot.savefig("ownsilhouette.png")
util.print_latex_statistics_clusters(dataset_knn, 'cluster', target, 'label')
del dataset_knn['silhouette']


k_values = range(2, 10)
silhouette_values = []

# Do some initial runs to determine the right number for k

print '===== k medoids clustering ====='
for k in k_values:
    print 'k = ', k
    dataset_cluster = clusteringNH.k_medoids_over_instances(copy.deepcopy(dataset), target, k, 'default', 20, n_inits=10)
    silhouette_score = dataset_cluster['silhouette'].mean()
    print 'silhouette = ', silhouette_score
    silhouette_values.append(silhouette_score)

plot.plot(k_values, silhouette_values, 'b-')
plot.ylim([0,1])
plot.xlabel('k')
plot.ylabel('silhouette score')
#plot.show()
plot.savefig("ownk_medoids.png")

# And run k medoids with the highest silhouette score

k = 6

dataset_kmed = clusteringNH.k_medoids_over_instances(copy.deepcopy(dataset), target, k, 'default', 20, n_inits=50)
plot = DataViz.plot_clusters_3d(dataset_kmed, target, 'cluster', ['label'])
plot.savefig("crowd_gyr_3d_medoids.png")
DataViz.plot_silhouette(dataset_kmed, 'cluster', 'silhouette')
util.print_latex_statistics_clusters(dataset_kmed, 'cluster', target, 'label')

# And the hierarchical clustering is the last one we try

clusteringH = HierarchicalClustering()

k_values = range(2, 10)
silhouette_values = []

# Do some initial runs to determine the right number for the maximum number of clusters.

print '===== agglomaritive clustering ====='
for k in k_values:
    print 'k = ', k
    dataset_cluster, l = clusteringH.agglomerative_over_instances(copy.deepcopy(dataset), target, k, 'euclidean', use_prev_linkage=True, link_function='ward')
    silhouette_score = dataset_cluster['silhouette'].mean()
    print 'silhouette = ', silhouette_score
    silhouette_values.append(silhouette_score)
    if k == k_values[0]:
        DataViz.plot_dendrogram(dataset_cluster, l)

plot.plot(k_values, silhouette_values, 'b-')
plot.ylim([0,1])
plot.xlabel('max number of clusters')
plot.ylabel('silhouette score')
#plot.show()
plot.savefig("ownagglomaritive.png")

plot = DataViz.plot_clusters_3d(dataset_cluster, target, 'cluster', ['label'])
plot.savefig("crowd_gyr_3d_agglomaritive.png")

# And we select the outcome dataset of the knn clustering....

dataset_knn.to_csv(dataset_path + 'chapter5_result_own_gyr.csv')
