import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

class cMeans:

    def prepareData(self, data):
        # Extracting X
        X = data[:,:-2]

        # Extracting y
        goals = data[:,-2:]
        results = goals[:,0] - goals[:,1]
        y = np.array([[1,0,0] if result > 0 else [0,1,0] if result == 0 else [0,0,1] for result in results])
        return X


    def cluster(self, X, label, clusters=3):
        kmeans = KMeans(n_clusters=clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        cl = kmeans.fit_predict(X)
        plt.scatter(X[:,0], X[:,1],c=cl,cmap='rainbow')
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='black')
        plt.title(label)
        plt.show()

