import json
import random
import os
from tqdm import tqdm 
import shutil
from datetime import datetime
import multiprocessing


def mkdir_dir(dir):
    if not os.path.exists(dir):
        try:
            os.makedirs(dir, exist_ok=True)
        except Exception as e:
            print(f"Error creating folder {dir}: {e}")

def remove_dir(dir):
    if os.path.exists(dir):
        try:
            shutil.rmtree(dir)
        except Exception as e:
            print(f"Error deleting folder {dir}: {e}")
def get_content_from_json(json_file_path):
    with open(json_file_path, 'r') as json_file:
        result_dict = json.load(json_file)
    return result_dict

def get_file_content(file_path):
    f = open(file_path)
    return f.read();

def append_content_to_file(path, content):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    f = open(path, 'a+')
    f.write(content)
    f.flush()
    f.close()

def random_file_prefix(prefix_name):
    random_bytes = os.urandom(8)  # Generate 8 random bytes
    random_str = random_bytes.hex()
    temp_filename = f"{prefix_name}.{random_str}.mlir"
    return temp_filename


def execmd(cmd, timeout_time=10):
    # print('[execmd] ' + cmd)
    timeout_cmd = ' '.join(['timeout', str(timeout_time), cmd])
    try:
        pipe = os.popen(timeout_cmd)
        reval = pipe.read()
        pipe.close()
        return reval
    except BlockingIOError:
        print("[execmd] trigger BlockingIOError")
        return "None"

def get_all_files_in_directory(directory,postfix='.mlir'):
    files = []
    for item in os.listdir(directory):
        full_path = os.path.join(directory, item)
        if os.path.isfile(full_path) and item.endswith(postfix):
            files.append(full_path)
    return files



class Configuration:
    def __init__(self):
        self.project_dir = '../exp_srcipt/'
        self.scan_func = self.project_dir + '/third_party_tools/SplitFunc'
        self.mlir_opt = '../llvm-project/install/bin/mlir-opt'
        
        self.opt_file =  '../opt.txt'
        self.stacktrace_file = self.project_dir + '/crash.json'

       
    def init(self,seed_name):
        print('[Config.init] Init all options and mlir seeds.')
        self.seed_dir = os.path.join(self.project_dir, seed_name)
        # current_datetime = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        # self.log_dir = os.path.join(self.project_dir, 'multiple_log', seed_name, current_datetime)
        self.tmp_dir = self.project_dir + '/sample_tmp/'
        self.crash_dir = self.project_dir + '/crash_file/'
        
 
        self.stacktrace = get_content_from_json(self.stacktrace_file)
        self.all_mlirfiles =  get_all_files_in_directory(self.seed_dir)



    # def process_stacktrace(self, stacktrace):
    #     new_stacktrace = []
    #     for line in stacktrace:
    #         line = line.strip()
    #         if not line.startswith('#'):
    #             continue
    #         new_stacktrace.append(line)
    #     return '\n'.join(new_stacktrace)


def delete_mlir_files(directory='/tmp'):
    print(f"[delete_mlir_files]: Start delete the mlir files in /tmp")
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        if os.path.isfile(file_path) and filename.endswith('.mlir'):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"Failed to delete {file_path}: {e}")


