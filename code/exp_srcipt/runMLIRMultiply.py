import argparse
import json
from pathlib import Path
import os
from tqdm import tqdm
import shutil
import multiprocessing


root_path = '..'

class Configuration:
    def __init__(self, iterator):
        self.seed_dir = args.seed_dir + os.sep + 'seed' + str(iterator)
        self.result_dir = args.result_dir + os.sep + 'result' + str(iterator)
        self.new_gen_file = root_path + os.sep + 'exp_srcipt' + os.sep +  'new_generated_' + str(iterator) + '.txt'
        self.correct_gen_file = root_path + os.sep + 'exp_srcipt' + os.sep + 'correct_generated_' + str(iterator) + '.txt'
        


config = None
def read_and_parse_json(file_path):
    f = open(file_path)
    lines = f.readlines()
    for line in tqdm(lines, desc="Parsing a generated file"):
        try:
            file_name = config.seed_dir + os.sep + random_file_prefix()
            put_file_content(file_name,json.loads(line.strip()))
        except Exception as e:
            print(f"Skipping line due to error: {line.strip()}")
            print(f"Error: {e}")
    entries = os.listdir(config.seed_dir)
    file_count = sum(1 for entry in entries if os.path.isfile(os.path.join(config.seed_dir, entry)))
    print('=========== Total file number: ' + str(file_count) + ' ================')

def convert_generated_to_single_file():
    if os.path.exists(config.seed_dir):
        shutil.rmtree(config.seed_dir)
    os.makedirs(config.seed_dir, exist_ok=True)
    for id in range(args.max_generated_file_id+1):
        file_path = f'{root_path}/model/generated' + str(id) + '.txt'
        print(f'parse file path {file_path}')
        read_and_parse_json(file_path)

def get_all_file_paths(file_path):
    print('========== Loading all mlir file path ===================')
    path = Path(file_path)
    return [str(file.resolve()) for file in path.rglob('*') if file.is_file()]

def get_file_lines(file_path):
    f = open(file_path)
    lines = f.readlines()
    return lines

def get_file_line_num(file_path):
    f = open(file_path)
    content = f.read().strip()
    return len(content.split('\n'))

def get_file_content(file_path):
    f = open(file_path)
    return f.read();

def put_file_content(path, content):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open(path, 'a+')
    f.write(content)
    f.flush()
    f.close()

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

def execmd_limit_time(cmd, time_limit):
    import time
    start = time.time()
    execmd("timeout " + str(time_limit) + " " + cmd)
    end = time.time()
    return (end - start) >= time_limit

# def random_file_prefix():
#     # pattern: "/tmp/tmp.WeWmILaF2A"
#     cmd = "mktemp -p " + config.seed_dir
#     random_name = execmd(cmd)
#     random_name = random_name[:-1] if random_name.endswith('\n') else random_name
#     res_name = random_name.split('/')[-1]
#     cmd = "rm -f " + random_name
#     os.system(cmd)
#     return res_name

def random_file_prefix(prefix_name='tmp'):
    random_bytes = os.urandom(8)  # Generate 8 random bytes
    random_str = random_bytes.hex()
    temp_filename = f"{prefix_name}.{random_str}.mlir"
    return temp_filename




def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)



class OptRuner:
    def __init__(self, seed):
        lines = get_file_lines(args.optfile)
        self.opts = [_[:-1] if _.endswith('\n') else _ for _ in lines]
        self.seed = seed
        self.process_id = multiprocessing.current_process().pid

    # find all syntactically correct mlir programs
    def count_files_in_directory(self, directory):
        file_count = 0
        for root, dirs, files in os.walk(directory):
            file_count += len(files)
        return file_count

    def false_crash(self, report):
        return 'Stack dump:' not in get_file_content(report)

    def is_new_generated(self, out_mlir):
        return abs(get_file_line_num(out_mlir) - get_file_line_num(self.seed)) > 2


    def count_files_with_prefix(self, directory, prefix):
        files = os.listdir(directory)
        matching_files = []
        for file in files:
            if file.startswith(prefix):
                matching_files.append(file)
        # matching_files = [file for file in files if file.startswith(prefix)]
        return len(matching_files)


    def run_opt(self):
        file_name = os.path.basename(self.seed)
        # result/result0/tmp.1Vp5kMxXc2.mlir
        out_mlir_dir = config.result_dir + os.sep + file_name

        # validate the syntax of the mlir program
        out_mlir = out_mlir_dir + os.sep + file_name
        report = out_mlir_dir + os.sep + file_name + '.crash.txt'
        ensure_directory_exists(out_mlir)
        ensure_directory_exists(report)
        cmd = ' '.join([args.mlir_opt, self.seed, '1>', out_mlir, '2>', report])
        execmd_limit_time(cmd, args.timeout)
        if not get_file_content(out_mlir).strip() == '':
            with open(f'{root_path}/exp_srcipt/correct_generated_pid_{self.process_id}.txt', 'a+', encoding='utf-8') as json_file:
                json_file.write(json.dumps(get_file_content(self.seed), ensure_ascii=False) + '\n')
        new_generated_program = set()
        for option in self.opts:
            option = option.strip()
            # result/result0/tmp.1Vp5kMxXc2.mlir/after_-canonicalize.mlir
            out_mlir = out_mlir_dir + os.sep + 'after_' + option + '.mlir'
            # result/result0/tmp.0aej7JTl8s.mlir_-tosa-to-tensor.crash.txt
            report = config.result_dir + os.sep  + file_name + '_' + option + '.crash.txt'
            # ensure_directory_exists(out_mlir)
            # ensure_directory_exists(report)
            cmd = ' '.join([args.mlir_opt, option, self.seed, '1>', out_mlir, '2>', report])
            execmd_limit_time(cmd, args.timeout)
            #execmd(cmd)
            put_file_content(report, cmd)
            # only retain the true crash file
            if os.path.exists(report) and self.false_crash(report):
                os.remove(report)
            # if out_milr is empty, remove it
            if os.path.exists(out_mlir) and os.path.getsize(out_mlir) == 0:
                os.remove(out_mlir)
            # collect new generated mlir program
            elif self.is_new_generated(out_mlir):
                new_generated_program.add(get_file_content(out_mlir))
        if len(new_generated_program) > 0:
            for program in new_generated_program:
                with open(f'{root_path}/exp_srcipt/new_generated_pid_{self.process_id}.txt', 'a+', encoding='utf-8') as json_file:
                    json_file.write(json.dumps(program, ensure_ascii=False) + '\n')

        shutil.rmtree(out_mlir_dir)

        # move the crash report which fails in all options to the dir ’all_crash‘
        # this case maybe failed in parser
        if self.count_files_with_prefix(config.result_dir, file_name+'_') > 10:
            new_dir = config.result_dir + os.sep + 'all_crash'
            if new_dir and not os.path.exists(new_dir):
                os.makedirs(new_dir)
            cmd = 'mv ' + config.result_dir + os.sep + file_name + '_* ' + new_dir
            execmd(cmd)


def merge_file_with_prefix(dir, prefix, new_file_name):
    print(f'=========== Merging All {prefix} file ================')
    with open(new_file_name, 'w', encoding='utf-8') as new_gen_file:
        for filename in os.listdir(dir):
            # 'new_generated_pid' or correct_generated_pid
            if filename.startswith(prefix):
                filepath = os.path.join(dir, filename)
                if os.path.isfile(filepath):
                    with open(filepath, 'r', encoding='utf-8') as infile:
                        new_gen_file.write(infile.read())



def process_seed(seed):
    optRunner = OptRuner(seed)
    optRunner.run_opt()

def main():
    seeds = get_all_file_paths(config.seed_dir)
    pool = multiprocessing.Pool(processes=args.multiprocess)
    for _ in tqdm(pool.imap_unordered(process_seed, seeds), total=len(seeds), desc="Processing seeds"):
        pass
    pool.close()
    pool.join()
    print('====== End running options ==============')
    merge_file_with_prefix(root_path + os.sep + 'exp_srcipt', 'new_generated_pid_', config.new_gen_file )
    merge_file_with_prefix(root_path + os.sep + 'exp_srcipt', 'correct_generated_pid_', config.correct_gen_file)
    execmd(f'rm {root_path}/exp_srcipt/new_generated_pid_*')
    execmd(f'rm {root_path}/exp_srcipt/correct_generated_pid_*')





if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, help="timeout of milr-opt.", default=10)
    parser.add_argument("--mlir_opt", type=str, help="Path of mlir_opt",
                        default=f"{root_path}/llvm-release/install/mlir-opt")
    parser.add_argument("--optfile", type=str, help="Path of opt cmd file",
                        default=f"{root_path}/opt.new.txt")
    # generated files: generated0.txt generated1.txt generated2.txt
    parser.add_argument("--max_generated_file_id", type=int, help="the max id of generated file",
                        default="1")
    parser.add_argument("--iterator", type=int, help="the iterator of generation",
                        default="1")                    
    parser.add_argument("--result_dir", type=str, help="Result directory.",
                        default=f"{root_path}/result")
    parser.add_argument("--seed_dir", type=str, help="Seed directory.",
                        default=f"{root_path}/seed")
    parser.add_argument(
        '--multiprocess', type=int, default=12, 
        help="Number of processes for multiprocessing (must be an integer)"
    )
    args = parser.parse_args()
    config = Configuration(args.iterator)
    convert_generated_to_single_file()
    main()

