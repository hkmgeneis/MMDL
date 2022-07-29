from preprocessing.judge_position import judge_position
from preprocessing.get_area_ratio import get_area_ratio
from tiatoolbox.dataloader import wsireader
from tiatoolbox.utils import misc
from utils.common import logger


def slide_window(image, image_address, x_begin: int, y_begin: int, w: int, h: int, window_size: int, stride: int,
                 num_name: int, luncancer: int, health: int, i, contours, points_con_thre, area_ratio_thre,
                 svs_address):
    """通过滑窗的方式得到小窗口
    :param image: svs文件
    :param x_begin: 滑窗的左上角坐标x
    :param y_begin: 滑窗的左上角坐标y
    :param w: 外接矩形的宽
    :param h: 外接矩形的高
    :param window_size: 滑窗大小
    :param stride: 滑窗步长
    :param num_name: 图片索引
    :param luncancer: 癌症区域索引
    :param health: 健康区域索引
    :param i: 癌症类型标志
    :param contours: 轮廓
    :param points_con_thre: 轮廓内点的个数阈值
    :param area_ratio_thre: window内面积比率阈值
    """
    # 统计一个区域得到小窗口的个数
    i_name = 0

    # 异常控制
    if w < window_size:
        w = window_size+10
    if h < window_size:
        h = window_size+10

    [m, n] = image.dimensions
    x_end = x_begin+w-window_size
    y_end = y_begin+h-window_size

    wsi = wsireader.OpenSlideWSIReader(input_path=svs_address)

    for x in range(x_begin, x_end, stride):
        for y in range(y_begin, y_end, stride):
            i_name += 1

            # 越界控制
            if x+window_size > m or y+window_size > n:
                continue

            # 去除轮廓外干扰区域
            point_list = [(x+int(window_size/2), y+int(window_size/2)), (x, y), (x+window_size, y),
                          (x, y+window_size), (x+window_size, y+window_size)]
            count_list = judge_position(contours, point_list)
            if count_list.count(1.0) < points_con_thre or count_list[0] == -1.0:
                continue

            ret = image.read_region((x, y), 0, (window_size, window_size)).convert('RGB')

            # 去除轮廓内的白色
            ratio = get_area_ratio(ret)

            if i == 0:
                if ratio < area_ratio_thre:
                    logger.info("get the {0}th cancer picture".format(i_name))
                    # img_resion.append(ret)
                    target_area_img = wsi.read_region(x, y, x+window_size, y+window_size)
                    misc.imwrite(image_address+"_"+str(luncancer)+"_"+str(i_name)+".orig.png", target_area_img)


