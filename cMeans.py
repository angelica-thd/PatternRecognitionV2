import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans

class cMeans:

    def prepareData(self, data):
        # Extracting X
        X = data[:,:-2]

        # Adding bias column x0
        X = np.column_stack((np.ones(X.shape[0]), X))

        # Extracting y
        goals = data[:,-2:]
        results = goals[:,0] - goals[:,1]
        y = np.array([[1,0,0] if result > 0 else [0,1,0] if result == 0 else [0,0,1] for result in results])
        return y


    def cluster(self, X, clusters=3):
        kmeans = KMeans(n_clusters=clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
        pred_y = kmeans.fit_predict(X)
        plt.scatter(X[:,0], X[:,1])
        plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red')
        plt.show()