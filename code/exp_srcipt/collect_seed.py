import argparse
import json
from pathlib import Path
import os
import subprocess
from tqdm import tqdm
import shutil
import multiprocessing
import random
import time




root_path = '..'
seed_dir = f'{root_path}/seed/seed_final'
if not os.path.exists(seed_dir):
    os.makedirs(seed_dir)
max_generated_file_id = 7

def put_file_content(path, content):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open(path, 'a+')
    f.write(content)
    f.flush()
    f.close()

def random_file_prefix(prefix_name='tmp'):
    random_bytes = os.urandom(8)  # Generate 8 random bytes
    random_str = random_bytes.hex()
    temp_filename = f"{prefix_name}.{random_str}.mlir"
    return temp_filename



def convert_generated_to_single_file():
    for id in range(max_generated_file_id +1):
        file_path = f'{root_path}/model/generated' + str(id) + '.txt'
        print(f'parse file path {file_path}')
        read_and_parse_json(file_path)

def read_and_parse_json(file_path):
    f = open(file_path)
    lines = f.readlines()
    #if os.path.exists(seed_dir):
    #    shutil.rmtree(seed_dir)
    # os.makedirs(seed_dir)
    for line in tqdm(lines, desc="Parsing a generated file"):
        try:
            file_name = seed_dir + os.sep + random_file_prefix()
            put_file_content(file_name,json.loads(line.strip()))
        except Exception as e:
            print(f"Skipping line due to error: {line.strip()}")
            print(f"Error: {e}")
    entries = os.listdir(seed_dir)
    file_count = sum(1 for entry in entries if os.path.isfile(os.path.join(seed_dir, entry)))
    print('=========== Total file number: ' + str(file_count) + ' ================')


def main():
    convert_generated_to_single_file()




if __name__ == "__main__":
    main()
