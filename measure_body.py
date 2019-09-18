import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

from get_depth import depth_predict
from bbox_extraction import bbox_extraction


def point_cloud(depth):
    f_mm = 3.519
    width_mm = 4.61
    height_mm = 3.46
    tan_horFov = width_mm / (2 * f_mm)
    tan_verFov = height_mm / (2 * f_mm)

    width = 640
    height = 480

    cx, cy = width / 2, height / 2
    fx = width / (2 * tan_horFov)
    fy = height / (2 * tan_verFov)
    xx, yy = np.tile(range(width), height).reshape(height, width), np.repeat(range(height), width).reshape(height,
                                                                                                           width)
    xx = (xx - cx) / fx
    yy = (yy - cy) / fy

    xyz = np.dstack((xx * depth, yy * depth, depth))

    return xyz


def calculate_grad(image, filter='scharr', smoothen=False):
    ddepth = cv2.CV_16S
    if smoothen:
        image = cv2.GaussianBlur(image, (3, 3), 0)
    if filter == 'scharr':
        grad_x = cv2.Scharr(image, ddepth, 1, 0)
    elif filter == 'sobel':
        grad_x = cv2.Sobel(image, ddepth, 1, 0)

    return grad_x


def get_boundaries(grad_x, bbox):
    x1 = bbox['x1']
    y1 = bbox['y1']
    x2 = bbox['x2']
    y2 = bbox['y2']
    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)

    row = cy
    col_start = x1
    col_end = x2

    x = np.arange(col_end - col_start)
    y = grad_x[row, col_start:col_end]
    # plt.plot(x, y)
    # plt.grid()
    # plt.show()

    min_grad = 0
    max_grad = 0
    centre_left_x = col_start
    centre_right_x = col_end
    for i in range(col_start, col_end, 1):
        #     print(i)
        if grad_x[row, i] < min_grad:
            min_grad = grad_x[row, i]
            centre_left_x = i
        if grad_x[row, i] >= max_grad:
            max_grad = grad_x[row, i]
            centre_right_x = i
    # print(grad_x[row, col_start:col_end])
    # print('row', row)
    # print('clx', centre_left_x)
    # print('crx', centre_right_x)
    for i in range(centre_left_x + 1, cx, 1):
        if grad_x[row, i] > grad_x[row, centre_left_x] and grad_x[row, i] < 0:
            centre_left_x = i
    centre_left_x += 1

    for i in range(centre_right_x - 1, cx, -1):
        if grad_x[row, i] < grad_x[row, centre_right_x] and grad_x[row, i] > 0:
            centre_right_x = i
    centre_right_x -= 1

    # print('After shift: clx', centre_left_x)
    # print('After shift: crx', centre_right_x)
    return row, centre_left_x, centre_right_x


def dist(p1, p2):
    d = np.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2 + (p2[2] - p1[2]) ** 2)
    return d


def size(xyz, row, left, right):
    # row = 250
    # left = 320
    # right = 360
    p1 = xyz[row, left, :]
    p2 = xyz[row, left + 1, :]
    tot = 0
    for i in range(left + 2, right, 1):
        d = dist(p1, p2)
        if d <= 0.01:
            tot += d
        #         print(d)
        p1 = p2
        p2 = xyz[row, i, :]

    return tot


def visualize(depth):
    row = 250
    col1 = 310
    col2 = 370
    x = np.arange(col2 - col1)
    y = depth[row, col1:col2]
    plt.plot(x, y)
    plt.grid()
    plt.show()

def measure(depth=None, bbox=None):
    # depth = np.load('./test_data/viz_predictions/sample_test/11_depth.npy')
    # depth_img = cv2.imread('./test_data/viz_predictions/sample_test/11.jpg',0)
    # visualize(depth)
    xyz = point_cloud(depth)
    grad_x = calculate_grad(image=depth, filter='scharr', smoothen=False)

    # bbox['neck'] = {'x1': 289, 'y1': 141, 'x2': 355, 'y2': 181}
    # bbox['stomach'] = {'x1': 249, 'y1': 382, 'x2': 445, 'y2': 440}

    row, left, right = get_boundaries(grad_x, bbox['neck'])
    neck_half = size(xyz, row, left, right)

    row, left, right = get_boundaries(grad_x, bbox['stomach'])
    stomach_half = size(xyz, row, left, right)

    # print(neck_half)
    # print(stomach_half)

    return neck_half*2, stomach_half*2

def navy_body_fat(neck,waist,height,sex):
    neck *= 100
    waist *= 100
    height *= 100
    if sex == 'male':
        body_fat = (495 / (1.0324 - 0.19077 * math.log10(waist - neck) + 0.15456 * math.log10(height))) - 450
    elif sex == 'female':
        print('Measure feature for females to be added soon')
        body_fat = 0.0
    return body_fat



if __name__ == '__main__':
    depth_array = depth_predict(image_pathfile='./data/image.txt')
    cv2.imshow('Depth', depth_array/10.0)
    cv2.waitKey(0)
    bbox = bbox_extraction(file_list='./data/image2.csv')
    print(bbox)
    neck, waist = measure(depth=depth_array, bbox=bbox)
    bf = navy_body_fat(neck,waist,height=1.82,sex='male')
    print('Your Waist:{:.2f}cm\nNeck:{:.2f}cm\nBody Fat percentage:{:.2f}%'.format(waist*100, neck*100, bf))












import matplotlib.image as mpimg












# img = cv2.imread('./11.jpg')
# img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_LINEAR)
# # img_depth = mpimg.imread('./11_depth.jpg')
# img_depth = cv2.imread('./11_depth.jpg', 0)
# img_parts = cv2.imread('./11_parts.jpg')
# cv2.imshow('Image', img)
# cv2.waitKey(0)
# cv2.imshow('Depth', img_depth)
# cv2.waitKey(0)
# # plt.imshow(img_depth)
# # plt.show()
# cv2.imshow('Body parts', img_parts)
# cv2.waitKey(0)
#
# print('\n\n\n')
# print('Your Waist:{:.2f}cm\nNeck:{:.2f}cm\nBody Fat percentage:{:.2f}%'.format(waist*100, neck*100, bf))


# print(bf)
# measure()