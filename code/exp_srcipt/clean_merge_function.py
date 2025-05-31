import runMLIRMultiple
import argparse
import json
from pathlib import Path
import os
from tqdm import tqdm
import multiprocessing
import shutil


def random_file_prefix():
    # pattern: "/tmp/tmp.WeWmILaF2A"
    cmd = "mktemp -p tmp_func" 
    random_name = runMLIRMultiple.execmd(cmd)
    random_name = random_name[:-1] if random_name.endswith('\n') else random_name
    res_name = random_name.split('/')[-1]
    cmd = "rm -f " + random_name
    os.system(cmd)
    return res_name


root = '../exp_srcipt/tmp_func'
# def read_and_parse_json(file_path):
#     all_functions_name = {}
#     with open(file_path, 'r', encoding='utf-8') as json_file:
#         all_functions = json.load(json_file)
#     for func in all_functions:
#         file_name = root + os.sep + random_file_prefix() + '.mlir'
#         runMLIRMultiple.put_file_content(file_name, func)
#         all_functions_name[file_name] = func
#     return all_functions_name

def build_name_content_map():
    print('Start loading all mlir files')
    file_content_map = {}
    for filename in os.listdir(root):
        filepath = os.path.join(root, filename)
        if os.path.isfile(filepath):
            with open(filepath, 'r', encoding='utf-8') as infile:
                content = infile.read()
                file_content_map[filepath] = content
    print(f'End loading all mlir files, total {len(file_content_map)} files')
    return file_content_map

def get_file_content(file_path):
    f = open(file_path)
    return f.read();

def get_file_lines(file_path):
    f = open(file_path)
    lines = f.readlines()
    return lines

def true_crash(report):
    return 'Stack dump:' in get_file_content(report)

def run_opt(item):
    funcname, func_content = item
    opts = get_file_lines(args.optfile)
    opts = [_[:-1] if _.endswith('\n') else _ for _ in opts]
    out_mlir = 'tmp.mlir'
    report = 'crash.txt'
    has_crash = False
    for option in opts:
        cmd = ' '.join([args.mlir_opt, option, funcname, '>', out_mlir, '2>', report])
        runMLIRMultiple.execmd(cmd)
        if true_crash(report):
            print(f'{funcname} file failed')
            has_crash = True
    if not has_crash:
        process_id = multiprocessing.current_process().pid
        with open(f'correct_pid_{process_id}.txt', 'a+', encoding='utf-8') as json_file:
            json_file.write(json.dumps(func_content, ensure_ascii=False) + '\n')


def main():
    all_functions = build_name_content_map()
    first_item = next(iter(all_functions.items()))
    print(f'First item in file_content_map: {first_item}')
    new_all_functions = []
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    for _ in tqdm(pool.imap_unordered(run_opt, all_functions.items()), total=len(all_functions.keys()), desc="Processing all programs"):
        pass
    
    # for func_name in tqdm(all_functions.keys(), desc="Processing funcs"):
    #     if run_opt(func_name):
    #         new_all_functions.append(all_functions[func_name])
    # print(f'new_all_functions\'s size: {len(new_all_functions)}')
    # with open(output_file, 'w', encoding='utf-8') as json_file:
    #     json.dump(new_all_functions, json_file, indent=2, ensure_ascii=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, help="timeout of milr-opt.", default=10)
    parser.add_argument("--mlir_opt", type=str, help="Path of mlir_opt",
                        default="../llvm-project/install/bin/mlir-opt")
    parser.add_argument("--optfile", type=str, help="Path of opt cmd file",
                        default="../opt.new.txt")
    args = parser.parse_args()
    main()
