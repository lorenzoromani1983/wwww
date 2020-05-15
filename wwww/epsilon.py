import numpy as np
from elasticsearch import Elasticsearch
from sklearn.datasets.samples_generator import make_blobs
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
import seaborn as sns
sns.set()

es = Elasticsearch('localhost:9200')

data = es.search(index="face_recognition", size=1000, body={"query": {"match_all": {}}}, request_timeout=1000)

arrays = []

for row in data['hits']['hits']:   
    arrays.append(row['_source']['my_vector'])

data = np.array(arrays)

neigh = NearestNeighbors(n_neighbors=len(data))
nbrs = neigh.fit(data)
distances, indices = nbrs.kneighbors(data)
distances = np.sort(distances, axis=0)
distances = distances[:,1]
mean = distances.mean()
print("Use this value for eps: "+str(mean))

plt.plot(distances)
plt.show()
    


