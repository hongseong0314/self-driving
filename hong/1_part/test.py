import cv2
import matplotlib.pyplot as plt
import os
from glob import glob
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import line_dectection

def camload(w, h, path, avi=False):
    """카메라 켜기"""
    capture = cv2.VideoCapture(path, apiPreference=None)#cv2.CAP_DSHOW
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    if avi:
        fps = capture.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적

        # fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')

        # 1프레임과 다음 프레임 사이의 간격 설정
        delay = round(1000/fps)

        # 웹캠으로 찰영한 영상을 저장하기
        # cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
        out = cv2.VideoWriter('output.avi', fourcc, fps, (w, h))
    
        if not out.isOpened():
            print('File open failed!')
            capture.release()
            sys.exit()

    while cv2.waitKey(33) < 0:
        ret, frame = capture.read()
        if not ret:
            continue
        # edge_img = edge_dec(frame)
        # mask_img = region_mask(edge_img)
        blend_img = line_dectection(frame)
        cv2.imshow("frame", blend_img)
        if avi:
            out.write(blend_img)

    return capture, frame

def img_save(img, path):
    """카메라 이미지 저장"""
    cv2.imwrite(path, img)

def cam_close(cam):
    """카메라 끄기"""
    cam.release()
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(data_dir, '../data/original_vid.mp4')
    print(path)
    w = 1280
    h = 720
    cam, img = camload(w, h, path=path, avi=False)
    # img_save(img, r"hong/test1.png")
    cam_close(cam)
