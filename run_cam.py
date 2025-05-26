#from astra_camera import Camera
import cv2
import numpy as np
from canbus import post

#cam = Camera()
def get_info():
    try:
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        
        person_idx = classes.index("person")
        # Load mô hình YOLOv4-tiny
        net = cv2.dnn.readNet("model/yolov4-tiny.weights", "model/yolov4-tiny.cfg")
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU) 
        
        cap = cv2.VideoCapture(0) 
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            height, width, _ = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 1/255.0, (320,320), swapRB=True, crop=False)
            net.setInput(blob)
        
            output_layers = net.getUnconnectedOutLayersNames()
            outputs = net.forward(output_layers)

            for output in outputs:
                for detection in output:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    if class_id == person_idx and confidence > 0.5:
                        center_x, center_y, w, h = (detection[0:4] * [width, height, width, height]).astype("int")
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        #tinh angle 
                        #x_mid = int((x+w)/2)
                        #x_mid = 
                        post(25, 0, 1)
                        
                        label = f"Person: {float(confidence):.2f}"
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                        break
                 
            cv2.imshow("Human Detection - YOLOv4-tiny", frame)
            key = cv2.waitKey(1)
            if key == 27:
                cv2.destroyAllWindows()
                break
        
        cap.release()
    except Exception as e:
        print(e)
    #finally:
        #cam.unload()

get_info()