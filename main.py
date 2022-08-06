import os
import cv2

from src.self_model import self_drving_model

def camload(w, h, path, avi=False):
    """카메라 켜기"""
    capture = cv2.VideoCapture(path, apiPreference=None)#cv2.CAP_DSHOW
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, w)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

    # 모델선언        
    self_model =self_drving_model(buffer_len=10, w=w, h=h, capture=capture, avi=False)
    self_model.setup()
    steper = self_model.frame_step()

    while cv2.waitKey(33) < 0:
        next(steper)
        
    return capture

def img_save(img, path):
    """카메라 이미지 저장"""
    cv2.imwrite(path, img)

def cam_close(cam):
    """카메라 끄기"""
    cam.release()
    cv2.destroyAllWindows() 

if __name__ == "__main__":
    data_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(data_dir, 'data/original_vid.mp4')
    w = 1280
    h = 720
    
    cam = camload(w, h, path=path, avi=True)
    # img_save(img, r"hong/test1.png")
    cam_close(cam)
