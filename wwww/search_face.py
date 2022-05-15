from imutils import paths
from elasticsearch import Elasticsearch
import face_recognition
import argparse
import numpy as np
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True, help="path to the input image(s)")
args = vars(ap.parse_args())

imagePaths = list(paths.list_images(args["input"]))

es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])

mean_values = []

def search():
    for (i, imagePath) in enumerate(imagePaths):
        try:
            print("[*] Encoding input image(s) {}/{}".format(i + 1,len(imagePaths)))
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            print(Exception)
            continue 
        boxes = face_recognition.face_locations(rgb,model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)
        d = [{"imagePath": imagePath, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
        for key in d:
            vectors = key['encoding']
            path = key['imagePath']
            vector_values = list(vectors)
            mean_values.append(vector_values)

search()

arrays = [np.array(x) for x in mean_values]
mean_list = [np.mean(k) for k in zip(*arrays)]

print("[*] Searching for matches...")

query = {
  "size" : 1000,
  "query": {
    "script_score": {
      "query" : {
        "match_all" : {}
     },
      "script": {
        "source": "1 / (1 + l2norm(params.queryVector, 'my_vector'))",
        "params": {
          "queryVector": mean_list
        }
      }
    }
  }
}

res = es.search(index="face_recognition", body=query)

for row in res['hits']['hits']:

    print("[*] Possible match: " + row['_source']['my_text'] + " >> score: " + str(row['_score']))
         
