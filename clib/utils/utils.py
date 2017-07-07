import chainer.functions as F
from chainer import function
import numpy as np
import cv2
from chainer import Variable

def reorg(input, stride=2):
    batch_size, input_channel, input_height, input_width = input.data.shape
    output_height, output_width, output_channel = int(input_height/stride), int(input_width/stride), input_channel*stride*stride
    output = F.transpose(F.reshape(input, (batch_size, input_channel, output_height, stride, output_width, stride)), (0, 1, 2, 4, 3, 5)) 
    output = F.transpose(F.reshape(output, (batch_size, input_channel, output_height, output_width, -1)), (0, 4, 1, 2, 3)) 
    output = F.reshape(output, (batch_size, output_channel, output_height, output_width))
    return output

# 2本の線の情報を受取り、被ってる線分の長さを返す。あくまで線分
def overlap(x1, len1, x2, len2):
    len1_half = len1/2
    len2_half = len2/2

    left = max(x1 - len1_half, x2 - len2_half)
    right = min(x1 + len1_half, x2 + len2_half)

    return right - left

# chainerのVariable用のoverlap
def multi_overlap(x1, len1, x2, len2):
    len1_half = len1/2
    len2_half = len2/2

    left = F.maximum(x1 - len1_half, x2 - len2_half)
    right = F.minimum(x1 + len1_half, x2 + len2_half)

    return right - left

# 2つのboxを受け取り、被ってる面積を返す(intersection of 2 boxes)
def box_intersection(a, b):
    w = overlap(a.x, a.w, b.x, b.w)
    h = overlap(a.y, a.h, b.y, b.h)
    if w < 0 or h < 0:
        return 0

    area = w * h
    return area

# chainer用
def multi_box_intersection(a, b):
    w = multi_overlap(a.x, a.w, b.x, b.w)
    h = multi_overlap(a.y, a.h, b.y, b.h)
    zeros = Variable(np.zeros(w.shape, dtype=w.data.dtype))
    zeros.to_gpu()

    w = F.maximum(w, zeros)
    h = F.maximum(h, zeros)

    area = w * h
    return area

# 2つのboxを受け取り、合計面積を返す。(union of 2 boxes)
def box_union(a, b):
    i = box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u

# chianer用
def multi_box_union(a, b):
    i = multi_box_intersection(a, b)
    u = a.w * a.h + b.w * b.h - i
    return u

# compute iou
def box_iou(a, b):
    return box_intersection(a, b) / box_union(a, b)

# chainer用
def multi_box_iou(a, b):
    return multi_box_intersection(a, b) / multi_box_union(a, b)


# 画像を読み込んで、hue, sat, val空間でランダム変換を加える関数
def random_hsv_image(bgr_image, delta_hue, delta_sat_scale, delta_val_scale):
    hsv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV).astype(np.float32)

    # hue
    hsv_image[:, :, 0] += int((np.random.rand() * delta_hue * 2 - delta_hue) * 255)

    # sat
    sat_scale = 1 + np.random.rand() * delta_sat_scale * 2 - delta_sat_scale
    hsv_image[:, :, 1] *= sat_scale

    # val
    val_scale = 1 + np.random.rand() * delta_val_scale * 2 - delta_val_scale
    hsv_image[:, :, 2] *= val_scale

    hsv_image[hsv_image < 0] = 0 
    hsv_image[hsv_image > 255] = 255 
    hsv_image = hsv_image.astype(np.uint8)
    bgr_image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2BGR)
    return bgr_image

