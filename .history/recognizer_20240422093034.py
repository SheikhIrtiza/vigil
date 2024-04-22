import os
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace


class Recognize_verify:
    def __init__(self):
        path_to_json_file = "dataset/recorded_encodings/face_encodings.json"
        with open(path_to_json_file, "r+") as json_file:
            self.data = json_file.read()

    def verify_faces(self):
        models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'Dlib']
        for _ in range(len(self.df)):
        # model = DeepFace.build(models)
            if DeepFace.find(img_path = self.df["encoding"].iloc[_], db_path = "dataset" , detector_backend = "ssd")["verified"] == True:
                self.df[["status"]] = True
            else:
                self.df[["status"]] = True

            
        return


obj = Recognize_verify()
obj.verify_faces()