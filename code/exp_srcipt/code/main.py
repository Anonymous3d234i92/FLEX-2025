from util import Configuration
import util
import os
import random
from tqdm import tqdm 
import shutil
import multiprocessing
import argparse
import time 
import shutil


config = Configuration()

def is_crash_file(err_result_file):
    err_message = util.get_file_content(err_result_file)
    if not '../llvm-project/install/bin/mlir-opt' in error_message:
        return False
    return True

def process_seed(seed_file):
    print(f'-- Process {seed_file}')
    file_name = os.path.basename(seed_file)
    tmp_dir = os.path.join(config.tmp_dir, file_name)
    util.mkdir_dir(tmp_dir)
    for option in config.opts:
        result_file = os.path.join(config.tmp_dir, util.random_file_prefix(file_name))
        err_result_file = f'{result_file}.err'
        cmd = (' ').join([config.mlir_opt, seed_file, '1>', result_file, '2>', err_result_file])
        util.execmd(cmd)
        if is_crash_file(err_result_file):
            # util.append_content_to_file(err_result_file, cmd)
            save_file = os.path.join(config.crash_dir, os.path.basename(err_result_file))
            shutil.move(err_result_file, save_file)
    util.delete_mlir_files()
    util.remove_dir(tmp_dir)


def parse_args():
    parser = argparse.ArgumentParser(description="Running Configuration")
    parser.add_argument(
        '--multiprocess', type=int, default=16, 
        help="Number of processes for multiprocessing (must be an integer)"
    )
    parser.add_argument(
        '--seedname', type=str, default='tosa_seed_v13', 
        help="Directory name of tosa seed to process"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    config.init(args.seedname) 
    # Use a Pool of workers to process the files in parallel
    with multiprocessing.Pool(processes=args.multiprocess) as pool:
        list(tqdm(pool.imap(process_seed, config.all_mlirfiles), 
                  desc="Processing All MLIR Files", 
                  total=len(config.all_mlirfiles), 
                  unit="files", 
                  ncols=100))
