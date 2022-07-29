import numpy as np
import cv2
import os
from preprocessing.slide_window import slide_window
from utils.common import logger


def get_ret_ifo(xy_list: list, slide, svs_address,
                image_dir_split, window_size: int, stride: int,
                points_con_thre: int, area_ratio_thre: float):
    """From the xy_list ,getting the information which can help get a min circumscribed rectangle
    :param xy_list: 点的坐标列表，坐标以列表的形式表示
    :param slide:读取的svs文件
    :param image_dir_split 存储分割后的图片路径
    :param window_size:窗口大小
    :param stride:窗口步长
    :param points_con_thre: 轮廓内点的个数阈值
    :param area_ratio_thre: 面积阈值
    """
    (filepath, filename) = os.path.split(svs_address)
    image_address = image_dir_split + '/' + filename

    for i in range(len(xy_list)):
        if len(xy_list[i]) == 0:
            continue
        cancer = 0
        health = 0

        for points in xy_list[i]:
            if i == 0:
                cancer += 1
                logger.info("Dealing with the {0}th Cancer area of {1}".format(cancer, svs_address.split('/')[-1]))
            if i == 1:
                health += 1
                logger.info("Dealing with Health area....")
            contours = np.array(points)
            x, y, w, h = cv2.boundingRect(contours)

            try:
                slide_window(slide, image_address, x, y, w, h,
                             window_size, stride, i,
                             cancer, health, i, contours,
                             points_con_thre, area_ratio_thre, svs_address)
            except Exception as e:
                logger.warn(e)


