# -*- coding: UTF-8 -*-
from posixpath import basename
#from slide_to_tiles import prepare_data
import subprocess
import pathlib
import click
import multiprocessing
import argparse
import os
from cv2 import transform

from tiatoolbox import utils
import tiatoolbox.tools.stainnorm as sn
from tiatoolbox.utils.exceptions import MethodNotSupported


def stainnorm(source_input, target_input, method, stain_matrix, output_dir, file_types):
    file_types = tuple(file_types.split(", "))

    if os.path.isdir(source_input):
        files_all = utils.misc.grab_files_from_dir(
            input_path=source_input, file_types=file_types
        )
    elif os.path.isfile(source_input):
        files_all = [
            source_input,
        ]
    else:
        raise FileNotFoundError
    
    if method not in ["reinhard", "custom", "ruifork", "macenko", "vahadane"]:
        raise MethodNotSupported

    norm = sn.get_normaliser(method, stain_matrix)
    
    norm.fit(utils.misc.imread(target_input))

    for curr_file in files_all:
        basename = os.path.basename(curr_file)
        transform = norm.transform(utils.misc.imread(curr_file))
        print(os.path.join(output_dir, basename))
        utils.misc.imwrite(os.path.join(output_dir, basename), transform)


def batch_cn(source_input, target_input, output_dir, file_types="*.png"):
    """External package tiatoolbox,
       multi-threaded thread pool automatically completes stainnorm.
    """

    pool_multi = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    stainnorm(source_input, target_input, "macenko", None, output_dir, file_types)

    # pool_multi.close()
    # pool_multi.join()
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tils to colorNormalization')
    parser.add_argument('--source_input', type=str, default="/media/zw/GeneisJY/geneis/geneis202104-08-tiles")
    parser.add_argument('--target_input', type=str, default="/home/huangkm/tmbpredictor-master/demo/tmb/asset/Template.png")
    parser.add_argument('--output_dir', type=str, default='/media/zw/GeneisJY/geneis/tiles_color_normalized')
    parser.add_argument('--file_types', type=str, default="*.png")
    args = parser.parse_args()
    # open('log.txt','a').write(args.output_dir)
    batch_cn(args.source_input, args.target_input, args.output_dir, args.file_types)


