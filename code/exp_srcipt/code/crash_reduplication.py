import util
import json
import os
from tqdm import tqdm 
from datetime import datetime



def process_stacktrace(stacktrace):
    new_stacktrace = []
    for line in stacktrace:
        line = line.strip()
        if not line.startswith('#'):
            continue
        line = line.split(' ', 2)[2].split('(../llvm-project/install/bin/mlir-opt')[0]
        new_stacktrace.append(line)
    return '\n'.join(new_stacktrace)
        
def reduplicate_crash(all_files):
    # all_files = util.get_all_files_in_directory(config.crash_dir, '.err')
    print(f'Total {len(all_files)} crash files')
    assert_file_dict = {}
    for file in tqdm(all_files, total=len(all_files), desc="Processing crash files", unit="%", ncols=100): 
        lines = util.get_file_content(file).split('\n')
        lines[0] = lines[0].replace('Testing: ', '')
        if 'mlir-opt:' in lines[0]:
            key = lines[0]
        else:
            key = process_stacktrace(lines)
        if key not in assert_file_dict:
            assert_file_dict[key] = []
        assert_file_dict[key].append(file)
    crash_json_file = 'crash_' + datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.json'
    with open(crash_json_file, 'w') as json_file:
        json.dump(assert_file_dict, json_file, indent=4)

def find_all_crashfile():
    root_path = '..'
    # find all result dir
    result_dir = [root_path + '/result/result-1'] 
    # find all crash_files 
    crash_files = []
    for dir in result_dir:
        for root, dirs, files in os.walk(dir):
            # only find all files in current dir,not sub dir
            for file in files:
                crash_files.append(os.path.join(root, file))
            break;
    return crash_files

if __name__ == '__main__':
    crash_files = find_all_crashfile()
    reduplicate_crash(crash_files)
