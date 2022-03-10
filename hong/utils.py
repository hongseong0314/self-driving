import numpy as np
import cv2
import os

def edge_dec(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # perform gaussian blur
    img_blur = cv2.GaussianBlur(img_gray, (17, 17), 0)

    # perform edge detection
    img_edge = cv2.Canny(img_blur, threshold1=50, threshold2=80)
    return img_edge

def hough_lines_detection(img, rho, theta, threshold, min_line_len, max_line_gap):
    """
    `img` should be the output of a Canny transform.
    """
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len,
                            maxLineGap=max_line_gap)
    return lines


def region_mask(img):
    # Image Size -> 720, 1180
    img_h = img.shape[0]
    img_w = img.shape[1]
    
    # Set Region
    region = np.array([
        [(100, img_h), (1180, img_h), (img_w / 2, img_h / 2)]
    ], dtype = np.int32) # dtype = np.int32 for fillPoly error
    
    # Apply Mask to the Image
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, region, 1)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

class Line:
    """
    open cv line으로 생성된 line에 기울기와 y절편 정보를 추가한다.
    """
    def __init__(self, x1, y1, x2, y2):

        self.x1 = np.float32(x1)
        self.y1 = np.float32(y1)
        self.x2 = np.float32(x2)
        self.y2 = np.float32(y2)

        self.slope = self.compute_slope()
        self.bias = self.compute_bias()

    def compute_slope(self):
        return (self.y2 - self.y1) / (self.x2 - self.x1 + np.finfo(float).eps)

    def compute_bias(self):
        return self.y1 - self.slope * self.x1

    def get_coords(self):
        return np.array([self.x1, self.y1, self.x2, self.y2])

    def set_coords(self, x1, y1, x2, y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def draw(self, img, color=[255, 0, 0], thickness=10):
        cv2.line(img, (int(self.x1), int(self.y1)), (int(self.x2), int(self.y2)), color, thickness)


def compute_lane_from_candidates(line_candidates, img_shape):

    # separate candidate lines according to their slope
    pos_lines = [l for l in line_candidates if l.slope > 0]
    neg_lines = [l for l in line_candidates if l.slope < 0]

    # interpolate biases and slopes to compute equation of line that approximates left lane
    # median is employed to filter outliers
    neg_bias = np.median([l.bias for l in neg_lines]).astype(int)
    neg_slope = np.median([l.slope for l in neg_lines])
    x1, y1 = 0, neg_bias
    x2, y2 = -np.int32(np.round(neg_bias / neg_slope)), 0
    left_lane = Line(x1, y1, x2, y2)

    # interpolate biases and slopes to compute equation of line that approximates right lane
    # median is employed to filter outliers
    lane_right_bias = np.median([l.bias for l in pos_lines]).astype(int)
    lane_right_slope = np.median([l.slope for l in pos_lines])
    x1, y1 = 0, lane_right_bias
    x2, y2 = np.int32(np.round((img_shape[0] - lane_right_bias) / lane_right_slope)), img_shape[0]
    right_lane = Line(x1, y1, x2, y2)

    return left_lane, right_lane

def weighted_img(img, initial_img, α=0.8, β=1., λ=0.):
    """
    Returns resulting blend image computed as follows:
    initial_img * α + img * β + λ
    """
    img = np.uint8(img)
    if len(img.shape) is 2:
        img = np.dstack((img, np.zeros_like(img), np.zeros_like(img)))

    return cv2.addWeighted(initial_img, α, img, β, λ)

def line_dectection(img):
    img_h, img_w = img.shape[:-1]

    img_edge = edge_dec(img)
    detected_lines = hough_lines_detection(img=img_edge,
                                       rho=2,
                                       theta=np.pi / 180,
                                       threshold=1,
                                       min_line_len=15,
                                       max_line_gap=5)

    detected_lines = [Line(l[0][0], l[0][1], l[0][2], l[0][3]) for l in detected_lines]

    candidate_lines = []
    for line in detected_lines:
        # consider only lines with slope between 30 and 60 degrees
        if 0.5 <= np.abs(line.slope) <= 2:
            candidate_lines.append(line)

    lane_lines = compute_lane_from_candidates(candidate_lines, img_edge.shape)

    line_img = np.zeros(shape=(img_h, img_w))
    for lane in lane_lines:
        lane.draw(line_img)

    # Set Region
    region = np.array([
        [(100, img_h), (1180, img_h), (img_w / 2, img_h / 2)]
    ], dtype = np.int32) # dtype = np.int32 for fillPoly error

    # Apply Mask to the Image
    mask = np.zeros_like(line_img)
    cv2.fillPoly(mask, region, 255)
    masked_img = cv2.bitwise_and(line_img, mask)

    img_blend = weighted_img(masked_img, img, α=0.8, β=1., λ=0.)
    return img_blend
