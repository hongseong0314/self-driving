import numpy as np
import cv2
import time
import os
from tensorflow.keras.models import load_model

class traffic_detection:
    
    def __init__(self, image):
        # Pretrained 파일 불러와서 yolo load.
        self.net = cv2.dnn.readNet("./../weights/yolov3_training_last.weights", "./../weights/yolov3_training.cfg")
        self.image = image
        
    def get_boxed_image(self):

        # Main에서 받은 이미지의 크기를 할당.
        HEIGHT, WIDTH, _ = self.image.shape
        
        # Traffic Sign에 대한 이름이 적혀있는 text파일 불러오기.
        classes = []
        with open("./../weights/signs.names.txt", "r") as f:
            classes = [line.strip() for line in f.readlines()]

        #
        layer_names = self.net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in self.net.getUnconnectedOutLayers()]
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        check_time = True
        confidence_threshold = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX



        detection_confidence = 0.5
        cap = cv2.VideoCapture(0)
        font = cv2.FONT_HERSHEY_SIMPLEX

        classification_model = load_model('traffic.h5') #load mask detection model
        classes_classification = []
        with open("./../weights/signs_classes.txt", "r") as f:
            classes_classification = [line.strip() for line in f.readlines()]

            
            
            #get image shape
            height, width, channels = self.image.shape

            # Detecting objects (YOLO)
            blob = cv2.dnn.blobFromImage(self.image, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(output_layers)

            # Showing informations on the screen (YOLO)
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
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
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
                    label = str(classes[class_ids[i]]) + "=" + str(round(confidences[i]*100, 2)) + "%"
                    img = cv2.rectangle(self.image, (x, y), (x + w, y + h), (255,0,0), 2)
                    crop_img = img[y:y+h, x:x+w]
                    if len(crop_img) >0:
                        crop_img = cv2.resize(crop_img, (WIDTH, HEIGHT))
                        crop_img =  crop_img.reshape(-1, WIDTH,HEIGHT,3)
                        prediction = np.argmax(classification_model.predict(crop_img))
                        print(prediction)
                        label = str(classes_classification[prediction])
                        img = cv2.putText(img, label, (x, y), font, 0.5, (255,0,0), 2)

            #     cv2.imshow("Image", img)
            #     if cv2.waitKey(1) & 0xFF == ord ('q'):
            #        break
            # cv2.destroyAllWindows()
            
            return img


# if __name__ == '__main__':
#     a = traffic_detection('img.jpeg')
#     a.get_boxed_image()
    