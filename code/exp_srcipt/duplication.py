import os
import json
from tqdm import tqdm  



root_path = '..'
mlir_opt = '../llvm-project/build/bin/mlir-opt'

def read_content(file_path):
    with open(file_path, 'r') as f:
        content = f.read();
    return content

def read_lines(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines();
    return lines


def process_stacktrace(stacktrace):
    new_stacktrace = []
    for line in stacktrace:
        line = line.strip()
        if not line.startswith('#') or not 'mlir::' in line:
            continue
        line = line.split(' ', 2)[2].split(f'({mlir_opt}')[0]
        new_stacktrace.append(line)
    return '\n'.join(new_stacktrace)

def find_opt_mlir(line):
    words = line.split()
    pass_opt = words[1] if len(words) > 1 else None
    mlir_file = words[2] if len(words) > 2 else None
    return pass_opt,mlir_file

def find_all_crashfile():
    # find all result dir
    result_dir = []
    for i in range(0,32):
        result_dir.append(root_path + '/result/result' + str(i)) 
    # result_dir = [root_path + '/result/result31',root_path + '/result/result30']
    # find all crash_files 
    crash_files = []
    for dir in result_dir:
        for root, dirs, files in os.walk(dir):
            # only find all files in current dir,not sub dir
            for file in files:
                crash_files.append(os.path.join(root, file))
            break;
    return crash_files

def execmd(cmd):
    import os
    # print('[execmd] ' + cmd)
    try:
        pipe = os.popen(cmd)
        reval = pipe.read()
        pipe.close()
        return reval
    except BlockingIOError:
        print("[execmd] trigger BlockingIOError")
        return "None"

def build_stack_map(crash_files):
    stacktrace_map = {}
    for crash_file in tqdm(crash_files,total=len(crash_files), desc="Processing crash files"):
        lines = read_lines(crash_file)
        key = ''
        for line in lines:
            if line.startswith('mlir-opt:'):
                key = line
                break
        if not key:
            key = process_stacktrace(lines)
        if key not in stacktrace_map:
            stacktrace_map[key] = []
        stacktrace_map[key].append(crash_file)
    with open('crash.json', 'w') as json_file:
        json.dump(stacktrace_map, json_file, indent=4) 
        

if __name__ == '__main__':
    crash_files = find_all_crashfile()
    build_stack_map(crash_files)
