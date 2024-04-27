import os
import cv2
from deepface import DeepFace



    
if DeepFace.extract_faces("generated_data/candidate_2.jpg", detector_backend = "ssd")[0]["confidence"] > 90:
    print("detected face")

else:
    print("face not detected or might be that high confidence is not achieved")
    