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



root_path = '/data/szy/extension/MLIR-NEW-extension-sample'
# mliropt_prefix = f'{root_path}/llvm-cov/llvm-project'
target='MLIR-ASE_'
class Configuration:
    def __init__(self):
        self.total_seed_dir = args.seed_dir + os.sep + f'{target}seed_20000'
        self.seed_dir = args.seed_dir + os.sep + f'{target}seed_24hours_cov'
        self.result_dir = args.result_dir + os.sep + 'result_24hours_cov'
        self.cov_dir = args.cov_dir + os.sep + f'{target}24hours_cov'
        if os.path.exists(self.cov_dir):
            shutil.rmtree(self.cov_dir)
        os.makedirs(self.cov_dir)
        if os.path.exists(self.seed_dir):
            shutil.rmtree(self.seed_dir)
        os.makedirs(self.seed_dir, exist_ok=True)
         
config = None
def read_and_parse_json(file_path, all_seed_dir):
    f = open(file_path)
    lines = f.readlines()
    for line in tqdm(lines, desc="Parsing a generated file"):
        try:
            file_name = all_seed_dir + os.sep + random_file_prefix()
            put_file_content(file_name,json.loads(line.strip()))
        except Exception as e:
            print(f"Skipping line due to error: {line.strip()}")
            print(f"Error: {e}")
    entries = os.listdir(all_seed_dir)
    file_count = sum(1 for entry in entries if os.path.isfile(os.path.join(all_seed_dir, entry)))
    print('=========== Total file number: ' + str(file_count) + ' ================')

def convert_generated_to_single_file():
    if os.path.exists(config.total_seed_dir):
        shutil.rmtree(config.total_seed_dir)
    os.makedirs(config.total_seed_dir, exist_ok=True)
    for id in range(args.max_generated_file_id):
        file_path = f'../model/generated' + str(id) + '.txt'
        print(f'parse file path {file_path}')
        read_and_parse_json(file_path, config.total_seed_dir)

def get_all_file_paths(file_path):
    print(f'========== Loading all mlir file path from {file_path} ===================')
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
        self.opts = get_file_lines(args.optfile)
        self.opts = [_[:-1] if _.endswith('\n') else _ for _ in self.opts]
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

    def start_process(self, method, *args):
        process = multiprocessing.Process(target=method, args=args)
        process.start()
        return process

    def collect_cov(self, cov_mlir_dir):
        print(f'Start merge for {cov_mlir_dir}')
        all_profraw = f'{cov_mlir_dir}/*.profraw'
        merge_cmd = (' ').join(['llvm-profdata', 'merge -j 16', all_profraw, '-o', f'{cov_mlir_dir}.profdata'])
        print(f'[merge_cmd] {merge_cmd}')
        execmd_limit_time(merge_cmd, 3600)
        shutil.rmtree(cov_mlir_dir)
        # report_cmd = (' ').join(['llvm-cov report', config.mlir_opt, f'-instr-profile={cov_mlir_dir}.profdata', f"--ignore-filename-regex='^{mliropt_prefix}/(llvm|build)'", '1>', report_file])

    def run_opt(self):
        file_name = os.path.basename(self.seed)
        # result/result0/tmp.1Vp5kMxXc2.mlir
        out_mlir_dir = config.result_dir + os.sep + file_name
        cov_mlir_dir = config.cov_dir + os.sep  + file_name
        for option in tqdm(self.opts, desc="Running opt"):
            option = option.strip()
            # result/result0/tmp.1Vp5kMxXc2.mlir/after_-canonicalize.mlir
            out_mlir = out_mlir_dir + os.sep + 'after_' + option + '.mlir'
            # result/result0/tmp.0aej7JTl8s.mlir_-tosa-to-tensor.crash.txt
            report = out_mlir_dir + os.sep + 'after_' + option + '.crash.txt'
            ensure_directory_exists(out_mlir)
            ensure_directory_exists(report)
            cmd = ' '.join(["timeout",  str(args.timeout), args.mlir_opt, option, self.seed, '1>', out_mlir, '2>', report])
            env = os.environ.copy()
            env['LLVM_PROFILE_FILE'] = f'{cov_mlir_dir}/option{option}.profraw'
            result = subprocess.run(cmd, shell=True, env=env)
        shutil.rmtree(out_mlir_dir)
        self.start_process(self.collect_cov, cov_mlir_dir)

def process_seed(seed):
    print(f'Processing seed: {seed}')
    optRunner = OptRuner(seed)
    optRunner.run_opt()


def runforseed():
    all_seeds = get_all_file_paths(config.total_seed_dir)

    start_time = time.time()
    for file_path in all_seeds:
        print(f'Processing generated file: {file_path}', flush=True)
        content = get_file_content(file_path)
        if content.startswith('"'):
            try:
                new_item = json.loads(content)
            except json.JSONDecodeError:
                continue
        else:
            new_item = content
        
        seed = config.seed_dir + os.sep + random_file_prefix()
        put_file_content(seed, new_item)
        process_seed(seed)
        already_run = time.time() - start_time
        print(f'[process_generated_file] --- Already run {already_run/3600:.2f}')
        if already_run > 24 * 60 * 60:
            print(f'Time out, already run {already_run/3600:.2f}')
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, help="timeout of milr-opt.", default=60)
    parser.add_argument("--mlir_opt", type=str, help="Path of mlir_opt",
                        default=f"/data/szy/extension/llvm-dir/llvm-cov/install/mlir-opt-cov-13c6abfa")  # the instrumented mlir-opt
    parser.add_argument("--optfile", type=str, help="Path of opt cmd file",
                        default=f"{root_path}/opt.new.txt")
    # generated files: generated0.txt generated1.txt generated2.txt
    parser.add_argument("--max_generated_file_id", type=int, help="the max id of generated file",
                        default="7")
    parser.add_argument("--iterator", type=int, help="the iterator of generation",
                        default="1")                    
    parser.add_argument("--result_dir", type=str, help="Result directory.",
                        default=f"{root_path}/result")
    parser.add_argument("--seed_dir", type=str, help="Seed directory.",
                        default=f"{root_path}/seed")
    parser.add_argument("--cov_dir", type=str, help="Coverage directory.",
                        default=f"{root_path}/cov_collection")
    parser.add_argument(
        '--multiprocess', type=int, default=1, 
        help="Number of processes for multiprocessing (must be an integer)"
    )
    args = parser.parse_args()
    config = Configuration()
    runforseed()



