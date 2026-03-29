import argparse
import json
from pathlib import Path
import os
import subprocess
from tqdm import tqdm
import shutil
from multiprocessing import Process
from math import ceil
import random
import time



root_path = '/data/szy/extension/MLIR-NEW-extension-sample'
version = '13c6abfa'
# version = '0f68dc6cf'
hours = 24

class Configuration:
    def __init__(self, iterator):
        self.total_seed_dir = args.seed_dir + os.sep + 'seed_20000'
        self.seed_dir = args.seed_dir + os.sep + f'seed_{hours}hours_bug'
        self.result_dir = args.result_dir + os.sep + 'result_{hours}hours_bug'
        self.cov_dir = args.cov_dir + os.sep + f'{hours}hours_bug'
        if os.path.exists(self.cov_dir):
            shutil.rmtree(self.cov_dir)
        os.makedirs(self.cov_dir, exist_ok=True)
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
        # self.process_id = multiprocessing.current_process().pid

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
    
    def process_stacktrace(self, stacktrace):
        new_stacktrace = []
        for line in stacktrace:
            line = line.strip()
            if not line.startswith('#'):
                continue
            line = line.split(' ', 2)[2].split(f'({root_path}/exp_script/thirdlibs/mlir-opt-{version}')[0]
            new_stacktrace.append(line)
        return '\n'.join(new_stacktrace)
    
    def get_crash_key(self, error_message):
        lines = error_message.split('\n')
        for line in lines:
            if f'mlir-opt-{version}:' in line:
                return line.replace('Testing: ', '')
        new_lines = error_message.split('PLEASE submit a bug report')[1].split('\n')
        return self.process_stacktrace(new_lines)
    
    def run_opt(self, bug_dict):
        file_name = os.path.basename(self.seed)
        # result/result0/tmp.1Vp5kMxXc2.mlir
        out_mlir_dir = config.result_dir + os.sep + file_name
        for option in tqdm(self.opts, desc="Running opt"):
            option = option.strip()
            # result/result0/tmp.1Vp5kMxXc2.mlir/after_-canonicalize.mlir
            out_mlir = out_mlir_dir + os.sep + 'after_' + option + '.mlir'
            # result/result0/tmp.0aej7JTl8s.mlir_-tosa-to-tensor.crash.txt
            report = out_mlir_dir + os.sep + 'after_' + option + '.crash.txt'
            ensure_directory_exists(out_mlir)
            ensure_directory_exists(report)
            cmd = ' '.join(["timeout",  str(args.timeout), args.mlir_opt, option, self.seed, '1>', out_mlir, '2>', report])
            result = subprocess.run(cmd, shell=True)
            error_message = get_file_content(report)
            if not 'PLEASE submit a bug report' in error_message:
                continue
            # print('error_message: ' + error_message)
            crash_key = self.get_crash_key(error_message)
            crash_value = f'{self.seed}_{option}'
            bug_dict.setdefault(crash_key, [crash_value]).append(crash_value)
            # if a file trigers a crash with an option, we don't try other options in this file.
            #continue
        shutil.rmtree(out_mlir_dir)
        # self.start_process(self.collect_cov, cov_mlir_dir)

def process_seed(seed,bug_dict):
    # print(f'Processing seed: {seed}')
    optRunner = OptRuner(seed)
    optRunner.run_opt(bug_dict)

def run_for_partition(seed_subset, process_id):
    start_time = time.time()
    bug_dict = {}
    # all_processed_mlirs = set()

    for file_path in tqdm(seed_subset, desc=f"Process {process_id}"):
        print(f'[Process {process_id}] Processing file: {file_path}', flush=True)
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
        process_seed(seed,bug_dict)
        already_run = time.time() - start_time
        print(f'[Process {process_id}] --- Runtime: {already_run / 3600:.2f} hours')
        if already_run > 3600:  # 每个进程最多运行1小时
            bug_num_file = config.cov_dir + os.sep + f'bug_num_proc_{process_id}.json'
            print(f'process_id {process_id}, run_files, {len(all_processed_mlirs)}')
            with open(bug_num_file, 'w', encoding='utf-8') as f:
                json.dump(bug_dict, f, ensure_ascii=False, indent=4)
            print(f'[Process {process_id}] Timeout: {already_run / 3600:.2f} hours')
            break

def run_parallel():
    all_seeds = get_all_file_paths(config.total_seed_dir)
    print(f'Total seeds to process: {len(all_seeds)}')
    print(f'{config.total_seed_dir}')
    # random.seed(42)
    # random.shuffle(all_seeds)

    num_processes = hours
    chunk_size = ceil(len(all_seeds) / num_processes)
    processes = []

    for i in range(num_processes):
        subset = all_seeds[i * chunk_size : (i + 1) * chunk_size]
        p = Process(target=run_for_partition, args=(subset, i))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()  

def merge_bugnum():
    merged_results = []
    cumulative_dict = {}

    for hour in range(24):
        file_path = os.path.join(config.cov_dir, f'bug_num_proc_{hour}.json')
        if not os.path.exists(file_path):
            print(f'Warning: {file_path} not found, skipping.')
            merged_results.append(cumulative_dict.copy())
            continue

        with open(file_path, 'r', encoding='utf-8') as f:
            current_data = json.load(f)

        cumulative_dict.update(current_data)
        merged_results.append(cumulative_dict.copy())

    with open(f"{config.cov_dir}/merged_summary.txt", 'w', encoding='utf-8') as out_f:
        # out_f.write("hour, bug_count, value_count\n") 
        for i, result in enumerate(merged_results, 1):
            bug_count = len(result)
            value_count = sum([len(v) if isinstance(v, list) else 1 for v in result.values()])
            out_f.write(f"{i}, {bug_count}, {value_count}\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, help="timeout of milr-opt.", default=60)
    parser.add_argument("--mlir_opt", type=str, help="Path of mlir_opt",
                        default=f"{root_path}/exp_script/thirdlibs/mlir-opt-{version}")  # the mlir-opt
    parser.add_argument("--optfile", type=str, help="Path of opt cmd file",
                        default=f"{root_path}/opt.new.txt")
    # generated files: generated0.txt generated1.txt generated2.txt
    parser.add_argument("--max_generated_file_id", type=int, help="the max id of generated file",
                        default="8")
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
    config = Configuration(args.iterator)
    # convert_generated_to_single_file()
    run_parallel()
    merge_bugnum()
