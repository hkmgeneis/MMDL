import cv2


def judge_position(contours, point_list):
    """Remove selected interference areas outside the outline
    :param contours: contour
    :param point_list: List of vertex and center coordinates of the sliding window
    :return: List of states if in contour, 1 in contour -1 not in contour
    """
    value_list = []
    for point in point_list:
        value = cv2.pointPolygonTest(contours, point, False)
        # yield value
        value_list.append(value)
    return value_list

