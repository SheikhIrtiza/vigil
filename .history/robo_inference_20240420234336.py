import os
from dotenv import load_dotenv, find_dotenv
import cv2
import supervision as sv
from inference.models.utils import get_roboflow_model


load_dotenv(find_dotenv())

cap  = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()


    model = get_roboflow_model(model_id="yolov8s-640", api_key = os.getenv("roboloflow_api_key"))

    result = model.infer(frame)[0]
    detections = sv.Detections.from_inference(result)
    
    print(len(detections))

    cv2.imshow("live feed", frame)

    if cv2.waitKey(1) & 0xFF == ord("x"):
        break


cap.release()
cv2.destroyAllWindows()