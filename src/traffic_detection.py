import numpy as np
import cv2
import time
import os
from tensorflow.keras.models import load_model


'''
Yolo v3를 이용해서 프레임 상의 Traffic Sign을 detection한다.
OpenCV는 4.3.0.36 버전으로 사용한다.

!pip uninstall opencv-python
!pip uninstall opencv-contrib-python

!pip install opencv-python==4.3.0.36
!pip install opencv-contrib-python==4.3.0.36

config, weight 등의 파일들의 경로는 Repository의 main.py 위치 기준으로 설정하였다.

TrafficDet 클래스를 호출하고 run 메소드에 frame을 인자로 실행하면 detection을 끝낸 frame을 반환한다.
'''

class TrafficDet():
    
    def __init__(self):
        # config파일 및 가중치 파일 경로 설정.
        self.cfg = "./weights/yolov3_training.cfg"
        self.weight = "./weights/yolov3_training_last.weights"
        
        # Darnet으로 부터 Yolo v3 모델 불러오기.
        self.net = cv2.dnn.readNetFromDarknet(self.cfg, self.weight)
        
        # Class 이름 가져오기.
        self.classes = []
        with open("./weights/signs.names.txt", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
            
            
    def run(self, frame):

        HEIGHT, WIDTH, CHANNEL = frame.shape
        img = frame

        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
        check_time = True
        confidence_threshold = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX

        detection_confidence = 0.5
        cap = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        
        classification_model = load_model('./src/traffic.h5')
        classes_classification = []
        
        with open("./weights/signs_classes.txt", "r") as f:
            classes_classification = [line.strip() for line in f.readlines()]

            # detection
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(output_layers)

            # 객체에 박스치기
            class_ids = []
            confidences = []
            boxes = []
            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]
                    
                    if confidence > confidence_threshold:
                        
                        # Object detected
                        center_x = int(detection[0] * WIDTH)
                        center_y = int(detection[1] * HEIGHT)
                        w = int(detection[2] * WIDTH)
                        h = int(detection[3] * HEIGHT)
                        
                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
                        
            indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
            
            for i in range(len(boxes)):
                
                if i in indexes:
                    
                    x, y, w, h = boxes[i]
                    
                    label = str(self.classes[class_ids[i]]) + "=" + str(round(confidences[i]*100, 2)) + "%"
                    img = cv2.rectangle(img, (x, y), (x + w, y + h), (255,0,0), 2)
                    crop_img = img[y:y+h, x:x+w]
                    
                    if len(crop_img) >0:
                        
                        crop_img = cv2.resize(crop_img, (WIDTH, HEIGHT))
                        crop_img =  crop_img.reshape(-1, WIDTH,HEIGHT,3)
                        prediction = np.argmax(classification_model.predict(crop_img))
                        print(prediction)
                        label = str(classes_classification[prediction])
                        img = cv2.putText(img, label, (x, y), font, 0.5, (255,0,0), 2)
                        
        return img