from preprocessing.judge_position import judge_position
from preprocessing.get_area_ratio import get_area_ratio
from tiatoolbox.dataloader import wsireader
from tiatoolbox.utils import misc
from utils.common import logger


def slide_window(image, image_address, x_begin: int, y_begin: int, w: int, h: int, window_size: int, stride: int,
                 num_name: int, luncancer: int, health: int, i, contours, points_con_thre, area_ratio_thre,
                 svs_address):
    """Get a small window by sliding window
    :param image: svs files
    :param x_begin: The coordinate x of the upper left corner of the sliding window
    :param y_begin: The coordinate y of the upper left corner of the sliding window
    :param w: the width of the bounding rectangle
    :param h: height of bounding rectangle
    :param window_size: Sliding window size
    :param stride: Sliding window step size
    :param num_name: image index
    :param luncancer: Cancer Region Index
    :param health: healthy area index
    :param i: Cancer Type Signs
    :param contours: contour
    :param points_con_thre: The threshold of the number of points in the contour
    :param area_ratio_thre: Area ratio threshold within the window
    """
    # Count an area to get the number of small windows
    i_name = 0

    # exception control
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

            # Out of bounds control
            if x+window_size > m or y+window_size > n:
                continue

            # Remove the interference area outside the contour
            point_list = [(x+int(window_size/2), y+int(window_size/2)), (x, y), (x+window_size, y),
                          (x, y+window_size), (x+window_size, y+window_size)]
            count_list = judge_position(contours, point_list)
            if count_list.count(1.0) < points_con_thre or count_list[0] == -1.0:
                continue

            ret = image.read_region((x, y), 0, (window_size, window_size)).convert('RGB')

            # remove white inside outline
            ratio = get_area_ratio(ret)

            if i == 0:
                if ratio < area_ratio_thre:
                    logger.info("get the {0}th cancer picture".format(i_name))
                    # img_resion.append(ret)
                    target_area_img = wsi.read_region(x, y, x+window_size, y+window_size)
                    misc.imwrite(image_address+"_"+str(luncancer)+"_"+str(i_name)+".orig.png", target_area_img)


