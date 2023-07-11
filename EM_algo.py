import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
data = pd.read_csv('Prog9_EM.csv')
X = data.values
num_clusters = 3
gmm = GaussianMixture(n_components=num_clusters)
gmm.fit(X)
labels = gmm.predict(X)
print('Cluster Labels:')
print(labels)