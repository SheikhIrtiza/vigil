import numpy as np
import pandas as pd
import datetime
import cv2
from deepface import DeepFace

# df = pd.read_csv("dataset/recorded_encodings/recorded_encode.csv")
# df[["title", "encoding", "timestamp"]] = [["fgjfg", "dfgg", "345"]]

# print(df["encoding"].iloc[-1])



# def capture_live_faces():
#         counter = 0
#         model_names = ["opencv", "ssd", "dlib", "mtcnn"]
#         while True:
#             # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
#             cap = cv2.VideoCapture(0)
#             cap.set(3, 640)     #set video width
#             cap.set(4, 480)     #set video height
#             ret, frame = cap.read()
#             mirrored_frame = cv2.flip(frame, 1)
                            
#             try:
#                 # using "SSD(single shot detector)" for its faster performance
#                 detected_face = DeepFace.extract_faces(mirrored_frame, detector_backend = "ssd")
#                 faces = detected_face[0]["facial_area"]
#                 cv2.rectangle(mirrored_frame, (faces["x"],faces["y"]), (faces["x"]+faces["w"], faces["y"]+faces["h"]), (255, 0, 0), 2)
#                 cv2.imshow("Detect Faces", mirrored_frame)

#             except:
#                   cv2.imshow("Detect Faces", mirrored_frame)
                  
            

#             if cv2.waitKey(1) & 0xFF == ord("c"):
#                   break
            

# capture_live_faces()
df = pd.DataFrame(columns = ["title", "encoding", "timestamp"])
df.reset_index(drop=True, inplace=True)
df.to_csv("dataset/recorded_encodings/recorded_encode.csv")



        