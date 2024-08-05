import numpy as np

class KMeansClustering(object):
    def __init__(self, k=3):
        self.k = k
        self.centroids = None
        
    @staticmethod
    def eucledian_distance(point, centroids):
        return np.sqrt(np.sum((centroids - point) ** 2, axis=1))
    
    def fit(self, X, max_iter=1000, learning_rate=0.0001):    
        X = np.array(X)
        self.centroids = np.random.uniform(np.amin(X, axis=0), np.amax(X, axis=0),
                                           size=(self.k, X.shape[1]))
        
        for _ in range(max_iter):
            y = []
            
            for point in X:
                distances = KMeansClustering.eucledian_distance(point, self.centroids)
                cluster_num = np.argmin(distances)
                y.append(cluster_num)
                
            y = np.array(y)
            cluster_indices = []
            
            for i in range(self.k):
                cluster_indices.append(np.argwhere(y == i)) # assigning all points to their cluster
                
            cluster_centers = [] # new centroids
            for i, indices in enumerate(cluster_indices):
                if len(indices) == 0:
                    cluster_centers.append(self.centroids[i])
                else:
                    cluster_centers.append(np.mean(X[indices], axis=0)[0])
            
            if np.max(self.centroids - np.array(cluster_centers)) < learning_rate:
                break
            else:
                self.centroids = np.array(cluster_centers)
                    
        return y
    
    def predict(self, X):
        y = []
        
        for point in X:
            distances = KMeansClustering.eucledian_distance(point, self.centroids)
            cluster_num = np.argmin(distances)
            y.append(cluster_num)
        
        return y