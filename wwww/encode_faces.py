from imutils import paths
from elasticsearch import Elasticsearch
import face_recognition
import argparse
import cv2
import os

ap = argparse.ArgumentParser()
ap.add_argument("-d", "--dataset", required=True, help="Path to your local dataset of images")
args = vars(ap.parse_args())
print("[*] Processing faces...")
imagePaths = list(paths.list_images(args["dataset"]))

es = Elasticsearch([{'host': 'localhost', 'port': '9200'}])

def _init_():
    for (i, imagePath) in enumerate(imagePaths):
        try:
            print("[*] Encoding image {}/{}".format(i + 1,len(imagePaths)))
            image = cv2.imread(imagePath)
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        except Exception:
            print(Exception)
            continue 
        boxes = face_recognition.face_locations(rgb,model='hog')
        encodings = face_recognition.face_encodings(rgb, boxes)
        d = [{"imagePath": imagePath, "loc": box, "encoding": enc} for (box, enc) in zip(boxes, encodings)]
        for key in d:
            filename = imagePath
            vectors = key['encoding']
            path = key['imagePath']
            coord = key['loc']
            vector_values = list(vectors)
            vector = {"my_text" : path, "my_vector" : vector_values, "text" : filename, "coord" : coord}
            es.index(index='face_recognition', body=vector, doc_type='_doc')

_init_()
