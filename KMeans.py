"""
Własna implementacja algorytmu K-Means w Pythonie.
Wykonanie: Daniel Madejski
"""

import pandas as pd
import numpy as np
from matplotlib import cm

class K_Means():

    def __init__(self, cluster_num):
        """Inicjalizacja klasy K_Means"""
        self.k = cluster_num
        self.conv = 0

    def data_download(self, file_path):
        """Pobranie danych do klasteryzacji za pmocą biblioteki Pandas"""
        self.data = pd.read_csv(file_path, header=None)
        #print(self.data.dtypes)

    def data_validation(self):
        """Walidacja danych"""
        self.numeric_data = self.data.select_dtypes(include=['float64', 'int64'])
        #print(self.numeric_data)

    def centroids_draw(self):
        """Losowanie centroidów"""
        self.centroids = self.numeric_data.sample(n=self.k)
        self.centroids = self.centroids.values.tolist()
        #print(self.centroids)

    def finding_clusters(self):
        """Dzielenie danych na klastry."""
        data_list = self.numeric_data.values.tolist()
        self.clusters = []
        for r in range(self.k):
            self.clusters.append([])
        for d in data_list:
            cluster_num = 0
            min_dist = 100
            for centroid in range(self.k):
                dist_to_centroid = np.linalg.norm(np.array(d) - np.array(self.centroids[centroid]))
                if dist_to_centroid < min_dist:
                    min_dist = dist_to_centroid
                    cluster_num = centroid
            self.clusters[cluster_num].append(d)
        #print(self.clusters[0], len(self.clusters[0]))
        #print(self.clusters[1], len(self.clusters[1]))
        #print(self.clusters[2], len(self.clusters[2]))

    def visualisation(self, ax):
        """Wizualizacja danych."""
        ax.clear()
        colors = [cm.tab10(i / self.k) for i in range(self.k)]
        clust = []
        for k in range(self.k):
            c = np.array(self.clusters[k])
            clust.append(c)

        for clusters in range(self.k):
            ax.scatter(clust[clusters][:, 0], clust[clusters][:, 1],
                        color=colors[clusters], label=f'Cluster {clusters+1}')

        centroids = np.array(self.centroids)
        ax.scatter(centroids[:, 0], centroids[:, 1],
                    color=colors, marker='x', s=200)

        ax.set_xlabel('Sepal Length in cm')
        ax.set_ylabel('Sepal Width in cm')
        ax.set_title('K-Means Clustering Visualisation')
        ax.legend()
        ax.grid(True)

    def finding_new_centroids(self):
        new_centroid = []
        for k in range(self.k):
            lista = []
            for p in range(len(self.centroids[k])):
                sum =0
                for l in range(len(self.clusters[k])):
                    sum += self.clusters[k][l][p]
                lista.append(sum/len(self.clusters[k]))
            new_centroid.append(lista)
        self.previous_centroid = self.centroids[:]
        self.centroids = new_centroid[:]

    def converged(self):
        """Sprawdzenie czy poprzednie i nowe centroidy się pokrywają."""
        if self.previous_centroid == self.centroids:
            self.conv = 1

    def data_normalization(self):
        length = len(self.numeric_data.columns)
        for l in range(length):
            col_mean = self.numeric_data[l].mean()
            col_std = self.numeric_data[l].std()
            self.numeric_data[l] = (self.numeric_data[l] - col_mean) / col_std

    def visualisation_original_data(self, ax):
        """Wizualizacja oryginalnych danych."""
        ax.clear()
        colors = [cm.tab10(i / self.k) for i in range(self.k)]
        original_data = self.data.values.tolist()
        unique_class = []
        clust = []
        for o_data in original_data:
            for element in o_data:
                if isinstance(element,str) and element not in unique_class:
                    unique_class.append(element)
                    clust.append([])

        for o_d in original_data:
            for ele in o_d:
                if isinstance(ele,str):
                    for i in range(len(unique_class)):
                        if ele == unique_class[i]:
                            o_d.remove(unique_class[i])
                            clust[i].append(o_d)

        c = []
        for r in range(len(unique_class)):
            c.append(np.array(clust[r]))

        for clusters in range(len(unique_class)):
            ax.scatter(c[clusters][:, 0], c[clusters][:, 1],
                        color=colors[clusters], label=unique_class[clusters])


        ax.set_xlabel('Sepal Length in cm')
        ax.set_ylabel('Sepal Width in cm')
        ax.set_title('Original data.')
        ax.legend()
        ax.grid(True)
