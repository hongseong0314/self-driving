import cv2
from cv2 import line
import matplotlib.pyplot as plt
import os
from glob import glob
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from _2part_utils import processed_frames, compute_offset_from_center, prepare_out_blend_frame, binarize, birdeye, get_fits_by_previous_fits, get_fits_by_sliding_windows
from _2part_utils import Line, draw_back_onto_the_road
from calibration import undistort

import pickle
with open(os.path.join(os.path.dirname(os.path.abspath("HONG")), './camera_matrix.pickle'), 'rb') as f:
    data = pickle.load(f)
mtx = data['mtx']
dist = data['dist']

# init line state
line_lt, line_rt = Line(buffer_len=10), Line(buffer_len=10)

def process_pipeline(frame, keep_state=True):
    """
    Apply whole lane detection pipeline to an input color frame.
    :param frame: input color frame
    :param keep_state: if True, lane-line state is conserved (this permits to average results)
    :return: output blend with detected lane overlaid
    """

    global line_lt, line_rt, processed_frames

    # undistort the image using coefficients found in calibration
    img_undistorted = undistort(frame, mtx, dist, img_plot=False)

    # binarize the frame s.t. lane lines are highlighted as much as possible
    img_binary = binarize(img_undistorted, verbose=False)

    # compute perspective transform to obtain bird's eye view
    img_birdeye, M, Minv = birdeye(img_binary, verbose=False)

    # fit 2-degree polynomial curve onto lane lines found
    if processed_frames > 0 and keep_state and line_lt.detected and line_rt.detected:
        line_lt, line_rt, img_fit = get_fits_by_previous_fits(img_birdeye, line_lt, line_rt, verbose=False)
    else:
        line_lt, line_rt, img_fit = get_fits_by_sliding_windows(img_birdeye, line_lt, line_rt, n_windows=9, verbose=False)

    # compute offset in meter from center of the lane
    offset_meter = compute_offset_from_center(line_lt, line_rt, frame_width=frame.shape[1])

    # draw the surface enclosed by lane lines back onto the original frame
    blend_on_road = draw_back_onto_the_road(img_undistorted, Minv, line_lt, line_rt, keep_state)

    # stitch on the top of final output images from different steps of the pipeline
    blend_output = prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, img_fit, line_lt, line_rt, offset_meter)

    processed_frames += 1

    return blend_output

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
    
    import pickle
    with open('./camera_matrix.pickle', 'rb') as f:
        data = pickle.load(f)
    mtx = data['mtx']
    dist = data['dist']

    while cv2.waitKey(33) < 0:
        ret, frame = capture.read()
        if not ret:
            continue

        pre_img = process_pipeline(frame, keep_state=False)

        cv2.imshow("frame", pre_img)
        if avi:
            out.write(pre_img)

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
    w = 1280
    h = 720
    
    cam, img = camload(w, h, path=path, avi=False)
    # img_save(img, r"hong/test1.png")
    cam_close(cam)
