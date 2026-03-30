import json
import sys
import subprocess
import os

with open("./all_ll_files_ori.json", 'r') as file:
    data = json.load(file)


def adddata(fpath):
    global data
    try:
        with open(fpath, 'r') as f:
            lines = json.load(f)
    except:
        return
    lines = [line.strip() for line in lines]
    result_set = set(data).union(set(lines))
    result_list = list(result_set)
    data = result_list

def write_train(train_data):
    with open("./all_ll_files.json", 'w') as file:
        json.dump(train_data, file, indent=4)
    with open("./traindata_size.log", "a") as log_file:
        log_file.write(f"{len(train_data)}\n")


index = int(sys.argv[1]) - 1


adddata(f"../llvm_exp_script/correct_generated_{index}.json")
adddata(f"../llvm_exp_script/new_generated_{index}.json")

with open("./all_ll_files_ori.json", 'w') as file:
    json.dump(data, file, indent=4)

# cluster and filter
if len(data) > 100000:
    print(f"[Info] Data size {len(data)} > 100000, using diversity_sampler for sampling")
    
    # 调用 diversity_sampler.sh 脚本进行多样性采样
    script_path = os.path.join(os.path.dirname(__file__), 'diversity_sampler.sh')
    try:
        result = subprocess.run(
            ['bash', script_path],
            check=True,
            capture_output=True,
            text=True
        )
        print(f"[Info] diversity_sampler.sh completed successfully")
        with open("./all_ll_files.json", 'r') as file:
            train_data = json.load(file)
            with open("./traindata_size.log", "a") as log_file:
                log_file.write(f"{len(train_data)}\n")
        if result.stdout:
            print(result.stdout)
    except subprocess.CalledProcessError as e:
        print(f"[Error] diversity_sampler.sh failed with exit code {e.returncode}")
        if e.stderr:
            print(f"Error output: {e.stderr}")
        with open("./traindata_size.log", "a") as log_file:
            log_file.write("failed sample\n")
        print(f"[Warning] Using original data due to sampling failure")
        exit(1)
    
else:
    print(f"[Info] Data size {len(data)} <= 100000, no sampling needed")
    write_train(data)