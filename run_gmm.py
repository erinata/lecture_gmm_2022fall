import pandas
import matplotlib.pyplot as pyplot

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

from sklearn.metrics import silhouette_score


dataset  = pandas.read_csv("dataset_2.csv")
print(dataset)

dataset = dataset.values

pyplot.scatter(dataset[:,1], dataset[:,2])
pyplot.savefig("scatterplot.png")
pyplot.close()


dataset = dataset[:,1:3]

print(dataset)

def run_kmeans(n, dataset):
  machine = KMeans(n_clusters=n)
  machine.fit(dataset)
  results = machine.predict(dataset)
  centroids = machine.cluster_centers_
  silhouette = silhouette_score(dataset, results, metric = "euclidean")
  pyplot.scatter(dataset[:,0], dataset[:,1], c=results)
  pyplot.scatter(centroids[:,0], centroids[:,1], c='red', s=200)
  pyplot.savefig("scatterplot_kmeans_" + str(n) + ".png")
  pyplot.close()
  return silhouette

results = [run_kmeans(i+2, dataset) for i in range(7)]
print(results)
pyplot.plot(range(2,9), results)
pyplot.savefig("kmeans_silhouette.png")
pyplot.close()
print(results.index(max(results))+2)


def run_gmm(n, dataset):
  machine = GaussianMixture(n_components=n)
  machine.fit(dataset)
  results = machine.predict(dataset)
  centroids = machine.means_
  silhouette = silhouette_score(dataset, results, metric = "euclidean")
  pyplot.scatter(dataset[:,0], dataset[:,1], c=results)
  pyplot.scatter(centroids[:,0], centroids[:,1], c='red', s=200)
  pyplot.savefig("scatterplot_gmm_" + str(n) + ".png")
  pyplot.close()
  return silhouette
  
results = [run_gmm(i+2, dataset) for i in range(7)]
print(results)
pyplot.plot(range(2,9), results)
pyplot.savefig("gmm_silhouette.png")
pyplot.close()
print(results.index(max(results))+2)




  
  
  
  
  
  