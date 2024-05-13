import os
import numpy as np
import pandas as pd
import cv2
from deepface import DeepFace
from datetime import datetime
from recognizer import *
import streamlit as st

st.title("VIGIL")

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
        frame_placeholder = st.empty()
        itera = 0  # Initialize itera 
        while True:
            ret, frames = cap.read()
            if not ret:
                break
            try:
                detected_faces = DeepFace.extract_faces(frames, detector_backend = self.model_names[1])
                for face in detected_faces:
                    facial_area = face["facial_area"]
                    cv2.rectangle(frames, (facial_area["x"],facial_area["y"]), (facial_area["x"]+facial_area["w"], facial_area["y"]+facial_area["h"]), (255, 0, 0), 2)
                frame_placeholder.image(frames, channels = "BGR")
            except Exception as E:
                frame_placeholder.image(frames, channels = "BGR")
            self.encoding_data.append(frames)
            itera+=1
            # cv2.imshow("Live Feed", frames)
            # using "SSD(single shot detector)" for its faster performance
            if itera>10:
                break
            else:
                continue
        # cam.release() 
        cv2.destroyAllWindows()
        # return self.encoding_data
   
 

    def process_frames(self):
        saved_faces = set()  # Set to keep track of saved faces (using bounding box coordinates)
        counter = 1
        for frame in self.encoding_data:
            try:
                detected_faces = DeepFace.extract_faces(frame, detector_backend=self.model_names[0])
                for index, face in enumerate(detected_faces):
                    facial_area = face["facial_area"]
                    bbox_coordinates = (facial_area["x"], facial_area["y"], facial_area["w"], facial_area["h"])
                    if bbox_coordinates not in saved_faces:  # Check if the face has already been saved
                        face_image = frame[facial_area["y"]:facial_area["y"]+facial_area["h"], 
                                            facial_area["x"]:facial_area["x"]+facial_area["w"]]  # Extract the detected face from the frame
                        cv2.imwrite(self.gen_path + f"recorded_encodings/candidate_{counter}_{index+1}.jpg", face_image)
                        counter = 1
                        
                        saved_faces.add(bbox_coordinates)  # Add the face's bounding box coordinates to the set of saved faces
            except Exception as E:
                # Handle exceptions or continue processing
                continue



if st.button("capture live"):
    obj = Detect_verify()
    # obj.from_source('/Users/zuhaib/Code/iQ/vigil/dataset/vids/MOVIE.mp4')
    obj.capture_live()
    obj.process_frames()
    obj1 = Recognize_verify()
    obj1.verify_faces()
    with open("generated_data/dump.txt", "r") as file:
        reso = file.read()
        st.text(f"analytics for the faces from video:-> {reso}")
elif st.button("load video"):
    obj = Detect_verify()
    obj.from_source('dataset/vids/vid1.mp4')
    obj.process_frames()
    obj1 = Recognize_verify()
    obj1.verify_faces()
    with open("generated_data/dump.txt", "r") as file:
        reso = file.read()
        st.text(f"analytics for the faces from video:-> {reso}")