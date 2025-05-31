import argparse
import json
from pathlib import Path
import os
from tqdm import tqdm

def read_and_parse_json(file_path):
    f = open(file_path)
    lines = f.readlines()
    for line in lines:
        try:
            file_name = args.seed_dir + os.sep + random_file_prefix() + '.mlir'
            put_file_content(file_name,json.loads(line.strip()))
        except Exception as e:
            print(f"Skipping line due to error: {line.strip()}")
            print(f"Error: {e}")
    entries = os.listdir(args.seed_dir)
    file_count = sum(1 for entry in entries if os.path.isfile(os.path.join(args.seed_dir, entry)))
    print('=========== Total file number: ' + str(file_count) + ' ================')


def get_all_file_paths(file_path):
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

def random_file_prefix():
    # pattern: "/tmp/tmp.WeWmILaF2A"
    cmd = "mktemp -p " + args.seed_dir
    random_name = execmd(cmd)
    random_name = random_name[:-1] if random_name.endswith('\n') else random_name
    res_name = random_name.split('/')[-1]
    cmd = "rm -f " + random_name
    os.system(cmd)
    return res_name


def ensure_directory_exists(file_path):
    directory = os.path.dirname(file_path)
    if directory and not os.path.exists(directory):
        os.makedirs(directory)


class SeedPool:
    def __init__(self):
        self.pool = []
    def init(self):
        self.pool = get_all_file_paths(args.seed_dir)


class OptRuner:
    def __init__(self, seed):
        self.opts = get_file_lines(args.optfile)
        self.opts = [_[:-1] if _.endswith('\n') else _ for _ in self.opts]
        self.seed = seed

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


    def run_opt(self):
        file_name = os.path.basename(self.seed)
        out_mlir_dir = args.result_dir + os.sep + file_name
        error_count = 0
        new_generated_program = set()
        for option in self.opts:
            option = option.strip()
            out_mlir = out_mlir_dir + os.sep + 'after_' + option + '.mlir'
            report = args.result_dir + os.sep  + file_name + '_' + option + '.crash.txt'
            ensure_directory_exists(out_mlir)
            ensure_directory_exists(report)
            cmd = ' '.join([args.mlir_opt, option, self.seed, '>', out_mlir, '2>', report])
            #execmd_limit_time(cmd, args.timeout)
            execmd(cmd)
            put_file_content(report, cmd)
            # only retain the true crash file
            if os.path.exists(report) and self.false_crash(report):
                os.remove(report)
            if os.path.exists(out_mlir) and os.path.getsize(out_mlir) == 0:
                error_count += 1
                os.remove(out_mlir)
                if error_count == 20:
                    break
            # collect new generated mlir program
            elif self.is_new_generated(out_mlir):
                new_generated_program.add(get_file_content(out_mlir))
        if len(new_generated_program) > 0:
            for program in new_generated_program:
                with open('new_generated.txt', 'a+', encoding='utf-8') as json_file:
                    json_file.write(json.dumps(program, ensure_ascii=False) + '\n')

        if self.count_files_in_directory(out_mlir_dir) > 0:
            with open('correct_generated.txt', 'a+', encoding='utf-8') as json_file:
                json_file.write(json.dumps(get_file_content(self.seed), ensure_ascii=False) + '\n')
        else:
            os.rmdir(out_mlir_dir)



def main():
    seedpool = SeedPool()
    seedpool.init()
    for seed in tqdm(seedpool.pool):
        optRunner = OptRuner(seed)
        optRunner.run_opt()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, help="timeout of milr-opt.", default=10)
    parser.add_argument("--mlir_opt", type=str, help="Path of mlir_opt",
                        default="../llvm-project/install/bin/mlir-opt")
    parser.add_argument("--optfile", type=str, help="Path of opt cmd file",
                        default="../opt.new.txt")
    # parser.add_argument("--generated_file", type=str, help="Path of generated mlir programs",
    parser.add_argument("--result_dir", type=str, help="Result directory.",
                        default="../result")
    parser.add_argument("--seed_dir", type=str, help="Seed directory.",
                        default="../seed")
    args = parser.parse_args()
    read_and_parse_json()
    main()
