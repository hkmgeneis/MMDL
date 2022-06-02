import cv2


def judge_position(contours, point_list):
    """去除轮廓以外选中的干扰区域
    :param contours: 轮廓
    :param point_list: 滑窗的顶点和中心坐标列表
    :return: 是否在轮廓中的状态列表 1在轮廓中 -1 不在轮廓中
    """
    value_list = []
    for point in point_list:
        value = cv2.pointPolygonTest(contours, point, False)
        # yield value
        value_list.append(value)
    return value_list

