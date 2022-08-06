import os
import pickle
from PIL import Image
import cv2
from src.utill import undistort, Line, binarize, birdeye, get_fits_by_previous_fits, get_fits_by_sliding_windows
from src.utill import compute_offset_from_center, draw_back_onto_the_road, prepare_out_blend_frame
class self_drving_model():
    COMOOENT = []
    def __init__(self, buffer_len,w,h,capture,avi,
                        **kwargs):
        super().__init__()
        self.line_lt, self.line_rt = Line(buffer_len), Line(buffer_len)
        self.weight, self.height = w, h
        self.capture = capture
        self.avi = avi
        pass

    def setup(self):
        # camera calibaration
        self.processed_frames = 0
        self.open_camera_matrix()
        self.undistort = lambda frame, mtx, dist, img_plot : undistort(frame, mtx, dist, img_plot)
        # linde detection
        # tarffic detection
        # segmentation
        if self.avi:
            fps = self.capture.get(cv2.CAP_PROP_FPS) # 카메라에 따라 값이 정상적, 비정상적

            # fourcc 값 받아오기, *는 문자를 풀어쓰는 방식, *'DIVX' == 'D', 'I', 'V', 'X'
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')

            # 1프레임과 다음 프레임 사이의 간격 설정
            delay = round(1000/fps)

            # 웹캠으로 찰영한 영상을 저장하기
            # cv2.VideoWriter 객체 생성, 기존에 받아온 속성값 입력
            out = cv2.VideoWriter('output.avi', fourcc, fps, (self.w, self.h))
        
            if not out.isOpened():
                print('File open failed!')
                self.capture.release()
                self.sys.exit()

        pass

    def frame_run(self):
        pass
    def frame_step(self):
        while True:
            ret, frame = self.capture.read()
            if not ret:
                continue
            # camera calibaration
            copy_frame = frame.copy()
            blend_img = self.preprocess(copy_frame)
            cv2.imshow("frame", blend_img)
            if self.avi:
                self.out.write(blend_img)
            # linde detection
            # tarffic detection
            # segmentation
            #xoutput = frame
            yield blend_img

    def open_camera_matrix(self):
        with open(os.path.join('data', 'camera_matrix.pickle'), 'rb') as f:
            data = pickle.load(f)
        self.mtx = data['mtx']
        self.dist = data['dist']
        del data
        pass

    def preprocess(self, frame, keep_state=True):
        img_undistorted = self.undistort(frame, self.mtx, self.dist, False)

        # binarize the frame s.t. lane lines are highlighted as much as possible
        img_binary = binarize(img_undistorted, verbose=False)
        # compute perspective transform to obtain bird's eye view
        img_birdeye, M, Minv = birdeye(img_binary, verbose=False)
        # fit 2-degree polynomial curve onto lane lines found
        if self.processed_frames > 0 and keep_state and self.line_lt.detected and self.line_rt.detected:
            self.line_lt, self.line_rt, img_fit = get_fits_by_previous_fits(img_birdeye, self.line_lt, self.line_rt, 
                                                                            verbose=False)
        else:
            self.line_lt, self.line_rt, img_fit = get_fits_by_sliding_windows(img_birdeye, self.line_lt, self.line_rt, 
                                                                                n_windows=9, verbose=False)

        # compute offset in meter from center of the lane
        offset_meter = compute_offset_from_center(self.line_lt, self.line_rt, frame_width=self.weight)

        # draw the surface enclosed by lane lines back onto the original frame
        blend_on_road = draw_back_onto_the_road(img_undistorted, Minv, self.line_lt, self.line_rt, keep_state)

        # stitch on the top of final output images from different steps of the pipeline
        blend_output = prepare_out_blend_frame(blend_on_road, img_binary, img_birdeye, 
                                                img_fit, self.line_lt, self.line_rt, 
                                                offset_meter)

        self.processed_frames += 1
        del blend_on_road, offset_meter
        return blend_output
