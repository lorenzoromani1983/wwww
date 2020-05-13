from elasticsearch import Elasticsearch
from sklearn.cluster import DBSCAN
import numpy as np
import networkx
import cv2
import os.path
import shutil
from os import path
import sys

print("[!] Defining clusters...this may require time, depending on your dataset's size!")

es = Elasticsearch('localhost:9200')
G=networkx.DiGraph()
data = es.search(index="face_recognition", size=1000000, body={"query": {"match_all": {}}}, request_timeout=1000)
cwd = sys.path[0]
output = cwd+"\\output"
temp_dir = cwd+"\\temp"

arrays = []
paths = []
coords = []

for row in data['hits']['hits']:   
    array = np.array(row['_source']['my_vector'])
    image = row['_source']['my_text']
    coord = row['_source']['coord']
    arrays.append(array)
    paths.append(image)
    coords.append(coord)

arrayTuple = tuple(arrays)
data = np.vstack(arrayTuple)
clt = DBSCAN(metric="euclidean", n_jobs=2, min_samples=2, eps=0.4)
clt.fit(data)
labelIDs = np.unique(clt.labels_) 
numUniqueFaces = len(np.where(labelIDs > -1)[0])

print("[*] Number of unique faces: {}".format(numUniqueFaces))

for labelID in labelIDs:
    if labelID > -1:
        indices = np.where(clt.labels_ == labelID)[0]
        for i in indices:
 
            inputImage = shutil.copy(paths[i], temp_dir)
            image = cv2.imread(inputImage)
            top, right, bottom, left = coords[i]
            face = image[top:bottom, left:right]
            newFace = cv2.rectangle(image, (left,bottom), (right,top), (0, 255, 66), 2)
            faceFinal = cv2.putText(newFace, str(labelID), org=(left+5,bottom+30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 66), thickness=2)
            newFile = '{!s}/{!s}'.format(output, os.path.basename(paths[i]))

            if path.exists(newFile):
                print("[*] Overwriting file:")
                os.remove(paths[i])
                directory = os.path.dirname(paths[i])
                replaced = shutil.move(newFile, directory) 
                image = cv2.imread(replaced)
                top, right, bottom, left = coords[i]
                face = image[top:bottom, left:right]
                newFace = cv2.rectangle(image, (left,bottom), (right,top), (0, 255, 66), 2)
                faceFinal = cv2.putText(newFace, str(labelID), org=(left+5,bottom+30), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1, color=(0, 255, 66), thickness=2)
                newFile = '{!s}/{!s}'.format(output, os.path.basename(paths[i]))
                cv2.imwrite(newFile, faceFinal)
                print("Identity n. "+ str(labelID) + " > " + newFile)
                file = os.path.basename(paths[i])
                G.add_node(str(labelID))
                G.add_node(file)
                G.add_edge(str(labelID),file) 
                networkx.write_gexf(G, "face_graph.graphml")

            if not path.exists(newFile):
                cv2.imwrite(newFile, faceFinal)
                print("Identity n. "+str(labelID) + " > " + newFile)
                file = os.path.basename(paths[i])
                G.add_node(str(labelID))
                G.add_node(file)
                G.add_edge(str(labelID),file) 
                networkx.write_gexf(G, "face_graph.graphml")




        








				                                                                                                                     	