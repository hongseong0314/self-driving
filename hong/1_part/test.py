import cv2
import matplotlib.pyplot as plt
import os
from glob import glob
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from utils import line_dectection

def camload(w, h, path):
    """카메라 켜기"""
    capture = cv2.VideoCapture(path, apiPreference=None)#cv2.CAP_DSHOW
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
    while cv2.waitKey(33) < 0:
        ret, frame = capture.read()
        if not ret:
            continue
        # edge_img = edge_dec(frame)
        # mask_img = region_mask(edge_img)
        blend_img = line_dectection(frame)
        cv2.imshow("frame", blend_img)

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
    path = os.path.join(data_dir, '../data/video/original_vid.mp4')
    w = 1280
    h = 720
    cam, img = camload(w, h, path=path)
    # img_save(img, r"hong/test1.png")
    cam_close(cam)
