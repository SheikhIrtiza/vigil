import os
import cv2
import numpy as np
import pandas as pd
from deepface import DeepFace


class Recognize_verify:
    def __init__(self):
        self.df = pd.read_csv("/Volumes/Extended/projects/Vigil/dataset/recorded_encodings/recorded_encode.csv")
        self.df_comp = pd.read_csv("place_holder")
        with open("", "r+") as json_file:
            data = json_file.read()

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