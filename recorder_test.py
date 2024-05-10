import os
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace
from datetime import datetime
# import json
from recognizer import *


class Detect_verify:
    def __init__(self):
        self.model_names = ["opencv", "ssd", "Dlib", "mtcnn", "VGG-Face"]
        self.gen_path = "dataset/"

    def from_source(self, path_source):
        self.encoding_data = []
        cap = cv2.VideoCapture(path_source)

        if not cap.isOpened():
            print("not able to open the source")
            exit()
        
        while True:
            ret, frames = cap.read()

            if not ret:
                break

            # try:
            #     faces = DeepFace.extract_faces(frames, detector_backend = self.model_names[1])[0]["facial_area"]
            #     cv2.rectangle(frames, (faces["x"],faces["y"]), (faces["x"]+faces["w"], faces["y"]+faces["h"]), (255, 0, 0), 2)
            #     cv2.imshow("Detect Faces", frames)
            
            # except Exception as E:
            #     cv2.imshow("Detect Faces", frames)
            try:
                detected_faces = DeepFace.extract_faces(frames, detector_backend=self.model_names[1], enforce_detection=False)
                for face in detected_faces:
                    facial_area = face["facial_area"]
                    cv2.rectangle(frames, (facial_area["x"], facial_area["y"]), (facial_area["x"]+facial_area["w"], facial_area["y"]+facial_area["h"]), (255, 0, 0), 2)
                cv2.imshow("Detect Faces", frames)
            except Exception as E:
                print(f"An error occurred: {E}")
                cv2.imshow("Detect Faces", frames)

            # cv2.imshow('Source Video', frames)

            self.encoding_data.append(frames)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        # return self.encoding_data



    def capture_live(self):
        self.encoding_data = []
        # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        cam = cv2.VideoCapture(0)
        k = 1
        while True:
            cam.set(3, 640)     #set video width
            cam.set(4, 480)     #set video height
            ret, frames = cam.read()

            try:
                faces = DeepFace.extract_faces(frames, detector_backend = self.model_names[1])[0]["facial_area"]
                cv2.rectangle(frames, (faces["x"],faces["y"]), (faces["x"]+faces["w"], faces["y"]+faces["h"]), (255, 0, 0), 2)
                # cv2.imshow("Detect Faces", frames)
            
            except Exception as E:
                # cv2.imshow("Detect Faces", frames)
                continue

            self.encoding_data.append(frames)
            k+=1
            # cv2.imshow("Live Feed", frames)
            # using "SSD(single shot detector)" for its faster performance
            # if cv2.waitKey(1) & 0xFF == ord("q"):
            if k>5:
                break
            else:
                continue

        cam.release() 
        cv2.destroyAllWindows()

        # return self.encoding_data


    def process_frames(self):
        counter = 1
        for _ in self.encoding_data:
            try:
                self.detected_face = DeepFace.extract_faces(_, detector_backend = self.model_names[0])
                # faces = self.detected_face[0]["facial_area"]
                self.final_detected_face = self.detected_face[0]["facial_area"]
                # cv2.rectangle(_, (faces["x"],faces["y"]), (faces["x"]+faces["w"], faces["y"]+faces["h"]), (255, 0, 0), 2)
                # cv2.imshow("Detect Faces", _)

                if len(os.listdir(self.gen_path+"recorded_encodings")) == 0: 
                    cv2.imwrite(self.gen_path+f"recorded_encodings/candidate_{counter}"+".jpg", _)
                    counter += 1


                else:
                    # if DeepFace.verify(np.load(self.gen_path+"recorded_encodings/" + os.listdir(self.gen_path+"recorded_encodings")[-1]), _, model_name = 'Facenet512', enforce_detection=False)["verified"] == True:
                    if len(DeepFace.find(_, db_path = "dataset/recorded_encodings", enforce_detection = True, model_name = "Facenet512")[0]["identity"]) != 0:
                        continue

                    else:
                        cv2.imwrite(self.gen_path+f"recorded_encodings/candidate_{counter}"+".jpg", _)
                        counter += 1    
            
            except Exception as E:
                # cv2.imshow("Detect Faces", _)
                continue

obj = Detect_verify()
obj.from_source('dataset/vids/vid1.mp4')
# obj.capture_live()
obj.process_frames()

obj1 = Recognize_verify()
obj1.verify_faces()