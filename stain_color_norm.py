# -*- coding: UTF-8 -*-
import subprocess
from pathlib import Path
import click
import multiprocessing
import argparse
import os

# from utils.process import Process
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

def batch_cn(source_input, target_input, output_dir, file_types='*CN.png'):
    """External package tiatoolbox,
       multi-threaded thread pool automatically completes stainnorm.
    """
    if Path(source_input).is_dir():
        paths = Path(source_input).glob('*')
        pool_multi = multiprocessing.Pool(processes=multiprocessing.cpu_count())
        
        for path in paths:
            name = path.stem
            Path(output_dir).joinpath(name).mkdir(parents=True, exist_ok=True)
            if len(os.listdir(os.path.join(output_dir, name))) > 0:
                continue
            pool_multi.apply(stainnorm,
                         (path,target_input,"macenko",None,Path(output_dir).joinpath(name),file_types))

        pool_multi.close()
        pool_multi.join()
    elif Path(source_input).is_file():
        name = '-'.join(Path(source_input).parent.stem.split('-')[:3])
        Path(output_dir).joinpath(name).mkdir(parents=True, exist_ok=True)
        stainnorm(Path(source_input),target_input,"macenko",None,Path(output_dir).joinpath(name),file_types)
    else:
        raise FileNotFoundError


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tils to colorNormalization')
    parser.add_argument('--source_input', type=str, default="/home/zyj/Desktop/hkm/code/crctcga/tiles/")
    parser.add_argument('--target_input', type=str, default="/home/zyj/Desktop/hkm/code/mmdl/asset/Template.png")
    parser.add_argument('--output_dir', type=str, default='//home/zyj/Desktop/hkm/code/crctcga/tiles_cn/')
    parser.add_argument('--file_types', type=str, default="*.png")
    args = parser.parse_args()
    # open('log.txt','a').write(args.output_dir)
    batch_cn(args.source_input, args.target_input, args.output_dir, args.file_types)
