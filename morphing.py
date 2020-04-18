import cv2
import numpy as np
import math
import sys
from argparse import ArgumentParser

# line structure, store its start point and end point
class Line(object):
    def __init__(self, start_point, end_point):
        self.start_point = start_point
        self.end_point = end_point
    
    def get_vector(self):   # return vector
        return self.end_point - self.start_point

    def get_perpendicular(self):    # return perpendicular vector
        vector = self.get_vector()
        return np.array([vector[1], -vector[0]])

    def get_length(self):   # calculate the length of the vector
        vector = self.get_vector()
        length = np.sum(np.square(vector))
        return length
    
    def print_content(self):    # for debugging
        vector = self.get_vector()
        perpen = self.get_perpendicular()
        print('start {} end {} vector {} perpen {}'.format(self.start_point, self.end_point, vector, perpen))

# mouse event call back function
def get_feature_line(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(param[0], (x, y), 2, (0, 0, 255), thickness=-3)
        param[1].append(np.array([y, x]))   # store current point

        if len(param[1]) % 2 == 0:  # if 2 points, make a line
            cv2.line(param[0], (param[1][-2][1], param[1][-2][0]), (x, y), (0, 0, 255), 2)
            line = Line(np.array(param[1][-2]), np.array([y, x]))   # create a line object and store it
            param[2].append(line)

# calculate line interpolation for given ratio
def lineInterpolate(src_lines, dst_lines, ratio):
    inter_vectors = []
    for i in range(len(src_lines)):
        # interpolate of start_point and point
        start_point = (1 - ratio) * src_lines[i].start_point + ratio * dst_lines[i].start_point
        end_point = (1 - ratio) * src_lines[i].end_point + ratio * dst_lines[i].end_point
        # create line object to store
        inter_line = Line(start_point, end_point)
        inter_vectors.append(inter_line)
    
    return inter_vectors

# map the P in destination image to source image
def mapping(cur_point, src_vector, inter_vector, p=0, a=1, b=2):
    src_perpen = src_vector.get_perpendicular() # perpendicular vector
    inter_perpen = inter_vector.get_perpendicular()
    inter_start_point = inter_vector.start_point

    PX = cur_point - inter_start_point  # PX vector
    PQ = inter_vector.get_vector()      # PQ vector, destination vector

    inter_len = inter_vector.get_length()   # len of destination vector

    u = np.inner(PX, PQ) / inter_len    # calculate u and v
    v = np.inner(PX, inter_perpen) / np.sqrt(inter_len)
    
    PQt = src_vector.get_vector()       # PQ vector in src img
    src_len = np.sqrt(src_vector.get_length())  # its length
    xt = src_vector.start_point + u * PQt + v * src_perpen / src_len    # Xt point

    # calculate the distance from Xt to PQ vector in src img depend on u
    dist = 0
    if u < 0:
        dist = np.sqrt(np.sum(np.square(xt - inter_start_point)))
    elif u > 1: 
        dist = np.sqrt(np.sum(np.square(xt - inter_vector.end_point)))
    else:
        dist = abs(v)
    
    # calculate weight of this point
    weight = 0
    length = pow(inter_vector.get_length(), p)
    weight = pow((length / (a + dist)), b)

    return xt, weight

# do bilinear of given point and img on its color
def bilinear(img, point, h, w):
    x, y = point[0], point[1]
    x1, x2 = math.floor(x), math.ceil(x)    # ceiling and floor point
    y1, y2 = math.floor(y), math.ceil(y)
    if x2 >= h:                             # limit the range
        x2 = h - 1
    if y2 >= w:
        y2 = w - 1
    a, b = x - x1, y - y1
    # bilinear, get the color array (3,)
    val = (1 - a) * (1 - b) * img[x1, y1] + a * (1 - b) * img[x2, y1] + (1 - a) * b * img[x1, y2] + a * b *img[x2, y2]
    
    return val

# warping image
def warpImg(img , src_vectors, inter_vectors, p=0, a=1, b=2):
    h, w, _ = img.shape
    warp_img = np.empty_like(img)   # result img

    # loop every pixel
    for i in range(h):
        for j in range(w):
            psum = np.array([0, 0])
            wsum = 0
            # calculate the mapping point on src img of this point
            for idx, inter_vector in enumerate(inter_vectors):  # for each line vector
                xt, weight = mapping(np.array([i, j]), src_vectors[idx], inter_vector, p, a, b)
                psum = psum + xt * weight   # weighted point amd sum up
                wsum = wsum + weight        # weight sum up
            point = psum / wsum             # final point

            if point[0] < 0:                # limit the range
                point[0] = 0
            elif point[0] >= h:
                point[0] = h - 1
            if point[1] < 0:
                point[1] = 0
            elif point[1] >= w:
                point[1] = w - 1
            
            warp_img[i, j] = bilinear(img, point, h, w) # calulate the color by bilinear
    return warp_img
    

if __name__ == '__main__':
    # argument parser
    parser = ArgumentParser()
    parser.add_argument('--p', type=int, help='Parameter p for weight', default=0)
    parser.add_argument('--a', type=int, help='Parameter a for weight', default=1)
    parser.add_argument('--b', type=int, help='Parameter b for weight', default=2)
    parser.add_argument('--src', type=str, help='Path to src image', default='./img/women.jpg')
    parser.add_argument('--dst', type=str, help='Path to dst image', default='./img/cheetah.jpg')
    parser.add_argument('--frames', type=int, help='# of frames for animation', default=21)
    
    args = parser.parse_args()
    src_path, dst_path = args.src, args.dst
    p, a, b, frames = args.p, args.a, args.b, args.frames
    
    src_points = []
    dst_points = []
    src_lines = []
    dst_lines = []    
    
    # read images
    src_origin = cv2.imread(src_path)
    src_img = src_origin.copy()
    dst_origin = cv2.imread(dst_path)
    dst_img = dst_origin.copy()
    h, w, channel = src_img.shape

    # set mouse event
    param = [src_img, src_points, src_lines]
    cv2.namedWindow("Source Image")
    cv2.setMouseCallback("Source Image", get_feature_line, param=param)
    param = [dst_img, dst_points, dst_lines]
    cv2.namedWindow("Destination Image")
    cv2.setMouseCallback("Destination Image", get_feature_line, param=param)

    # create windows to get feature vectors
    while True:
        cv2.imshow("Source Image", src_img)
        cv2.imshow("Destination Image", dst_img)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    # exit if not matched
    if len(src_lines) != len(dst_lines):
        print("Control lines do not match!")
        sys.exit()
    
    animation = []  # save images for animation
    for i in range(frames):
        t = i / (frames - 1)
        print('{} / {} frame, t = {}'.format(i+1, frames, t))   

        # get vectors of line interpolation between src and dst lines for given t
        inter_vectors = lineInterpolate(src_lines, dst_lines, t)

        # get warp images
        src_warp = warpImg(src_origin, src_lines, inter_vectors, p, a, b)
        dst_warp = warpImg(dst_origin, dst_lines, inter_vectors, p, a, b)

        # dissolving
        img = np.empty_like(src_origin)
        for i in range(h):
            for j in range(w):
                img[i, j] = (1 - t) * src_warp[i, j] + t * dst_warp[i, j]
        animation.append(img)
    
    # play animation
    for img in animation:
        cv2.imshow('Animation', img)
        cv2.waitKey(500)
    
    cv2.destroyAllWindows()