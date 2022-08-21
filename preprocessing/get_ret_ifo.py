import numpy as np
import cv2
import os
from preprocessing.slide_window import slide_window
from utils.common import logger


def get_ret_ifo(xy_list: list, slide, svs_address,
                image_dir_split, window_size: int, stride: int,
                points_con_thre: int, area_ratio_thre: float):
    """From the xy_list ,getting the information which can help get a min circumscribed rectangle
    :param xy_list: List of coordinates for the point, the coordinates are expressed as a list
    :param slide:read svs file
    :param image_dir_split: Store the split image path
    :param window_size:window size
    :param stride:window step
    :param points_con_thre: The threshold of the number of points in the contour
    :param area_ratio_thre: area threshold
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


