import cv2
import numpy as np

class yolo_detection():
    def __init__(self):
        # YOLO 가중치 파일과 CFG 파일 로드
        self.YOLO_net = cv2.dnn.readNet("./../weights/yolov3.weights","./../weights/yolov3.cfg")

        # YOLO NETWORK 재구성
        layer_names = self.YOLO_net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.YOLO_net.getUnconnectedOutLayers()]

        # 원하는 class만 detection
        self.classes = []
        with open("./../weights/yolo.names", "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        self.class_only_name = [
            "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat", 
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench"
        ]
        self.class_only_index = [i for i, name in enumerate(self.classes) if name in self.class_only_name]

    def run(self, frame):
        h, w, c = frame.shape
        """YOLO 입력
        image: 입력 영상
        scalefactor: 입력 영상 픽셀 값에 곱할 값. 기본값은 1.
        size: 출력 영상의 크기. 기본값은 (0, 0).
        mean: 입력 영상 각 채널에서 뺄 평균 값. 기본값은 (0, 0, 0, 0).
        swapRB: R과 B 채널을 서로 바꿀 것인지를 결정하는 플래그. 기본값은 False.
        crop: 크롭(crop) 수행 여부. 기본값은 False.
        """
        blob = cv2.dnn.blobFromImage(image=frame, scalefactor=0.00392, size=(416, 416), mean=(0, 0, 0), swapRB=True, crop=False)
        self.YOLO_net.setInput(blob)
        outs = self.YOLO_net.forward(self.output_layers)
        class_ids = []
        confidences = []
        boxes = []

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                if class_id not in self.class_only_index:
                    continue
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * w)
                    center_y = int(detection[1] * h)
                    dw = int(detection[2] * w)
                    dh = int(detection[3] * h)
                    # Rectangle coordinate
                    x = int(center_x - dw / 2)
                    y = int(center_y - dh / 2)
                    boxes.append([x, y, dw, dh])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indexes = cv2.dnn.NMSBoxes(bboxes=boxes, scores=confidences, score_threshold=0.45, nms_threshold=0.4)

        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(self.classes[class_ids[i]])
                score = confidences[i]
                # 경계상자와 클래스 정보 이미지에 입력
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_ITALIC, 0.5, (0, 0, 255), 2)
                cv2.putText(frame, (f"{score:.2f}"), (x, y + h + 20), cv2.FONT_ITALIC, 0.5, (255, 200, 0), 2)
        cv2.imshow("YOLOv3", frame)
        return frame

if __name__ == "__main__":
    yoo = yolo_detection()
    VideoSignal = cv2.VideoCapture("./../donghyeoncho/carcar.mp4")

    i=0
    while True:
        # 웹캠 프레임
        ret, frame = VideoSignal.read()
        if i<200:
            i+=1
            continue

        ff = yoo.run(frame)
        # cv2.imwrite("result.jpg", ff)

        # input('sleep')
        if cv2.waitKey(100) > 0:
            break