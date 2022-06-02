import cv2
import numpy as np


def get_area_ratio(img):
    """去除轮廓内的干扰区域
    :param img: 滑动窗口
    :return: 面积比
    """
    img = np.array(img)

    img = cv2.GaussianBlur(img, (3, 3), 0)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh, img_binary = cv2.threshold(img_gray, 200, 255, cv2.THRESH_BINARY)

    # 得到轮廓
    contous, heriachy = cv2.findContours(img_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_list = []
    for contou in contous:
        area = cv2.contourArea(contou)
        area_list.append(area)

    img_w = img.shape[0]
    img_h = img.shape[1]
    area_ratio = sum(area_list)/(img_h * img_w)
    return area_ratio
