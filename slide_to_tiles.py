import argparse
import openslide
import os
import sys
from glob import glob
from os.path import join
from preprocessing.load_xml import load_xml
from preprocessing.get_ret_ifo import get_ret_ifo
from utils.common import logger
import multiprocessing
from pathlib import Path

from multiprocessing import Process
import os, time
import psutil

#global status
#status = 0

path_wd = os.path.dirname(sys.argv[0])
sys.path.append(path_wd)
if not path_wd == '':
    os.chdir(path_wd)
need_save = False


def multiprocessing_segmentation(xml, image_dir_root, images_dir_split, size_square, prepare_type):
    xy_list = load_xml(xml)
    try:
    	slide = openslide.open_slide(image_dir_root)
    	get_ret_ifo(xy_list, slide, image_dir_root, images_dir_split,
                size_square, size_square, 3, 0.3)
    except Exception as e:
	      print(e)
        #break

def prepare_data(images_dir_root, images_dir_split, size_square, prepare_type):
    
    samp_all = glob(images_dir_root+"/*.tif")
    segmentation_pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for samp in samp_all:
        configuration_file = str(Path(samp).parent) + '/' + Path(samp).stem + ".xml"
        #print(Path(samp).stem[:12])
        if os.path.exists(os.path.join(images_dir_split, Path(samp).stem)):
            continue
        print(configuration_file)
        segmentation_pool.apply(multiprocessing_segmentation,
                (configuration_file, samp, images_dir_split, size_square, prepare_type))


    logger.info('tiles are done.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='svs to tiles')
    parser.add_argument('--slide_image_root', type=str, default="/media/zw/colorectal/hkm/colorectal/tmb/verify/geneis/corectal202101_08/210910annotation/")
    # parser.add_argument("--configuration_file", type=str)
    parser.add_argument('--tiles_image_root', type=str, default="/media/zw/colorectal/hkm/colorectal/tmb/verify/geneis/corectal202101_08/tiles/")
    parser.add_argument('--size_square', type=int, default=512)
    parser.add_argument('--prepare_types', type=str, default="tif")
    args = parser.parse_args()

    logger.info('Processing tif images to tiles')
    available_policies = ["svs", "tif"]
    # assert os.path.exists(args.slide_image_root), "the target H&E file not found "
    # assert os.path.exists(args.configuration_file), "the H&E configuration xml file not found "
    # assert args.prepare_types in available_policies, "svs or ndpi slide support only"
    prepare_data(args.slide_image_root, args.tiles_image_root, args.size_square, args.prepare_types)
