import os
import json
import subprocess
from pathlib import Path
from tqdm import tqdm

project_dir = "/data/szy/extension/MLIR-NEW-extension-sample"
seed_directory = f"{project_dir}/seed/seed_24hours_bug"
# mlir_opt = "/data/szy/MLIR/llvm-release/install/mlir-opt-13c6abfa"

# if not os.path.exists(result_dir):
#     os.makedirs(result_dir)

def get_file_content(file_path):
    f = open(file_path)
    return f.read()

def compile_failed(error_message):
    if error_message.strip() == "":
        return False
    if 'error:' in error_message:
        return True
    else:
        print(f"[Error] Unknown error message: {error_message}")
    return False

def run_mlir_opt_on_seeds():
    seed_dir = Path(seed_directory)
    if not seed_dir.exists() or not seed_dir.is_dir():
        print(f"[Error] Seed directory not found: {seed_dir}")
        return

    mlir_files = list(seed_dir.glob("*.mlir"))
    failed_num = 0
    tokens_num = 0
    for mlir_file in tqdm(mlir_files, desc="Processing MLIR files"):
        # print(f"\n[Running] {mlir_file.name}")
        # crash_file = f'{result_dir}/{mlir_file.name}.txt'
        # command = f"{mlir_opt} {mlir_file} 2> {crash_file}"
        # # print(command)
        # result = subprocess.run(command, shell=True, capture_output=True, text=True)
        # if compile_failed(get_file_content(crash_file)):
        #     failed_num += 1
        # else:
        #     print(f"[Success] {mlir_file.name} compiled successfully.")

        # Get the number of tokens in the file
        code_str = get_file_content(mlir_file)
        tokens_num += tokenize_simple(code_str)
    
    print(f"[Info] Running mlir-opt on seed directory: {seed_directory}")
    print(f"Total files: {len(mlir_files)}")
    # print(f"Total failed: {failed_num}")
    # print(f"Total successful: {len(mlir_files) - failed_num}")
    print(f"Total tokens: {tokens_num}")
    print(f"Average tokens per file: {tokens_num / len(mlir_files) if mlir_files else 0:.2f}")

def tokenize_simple(code_str):
    import re
    tokens = re.findall(r'\w+|\S', code_str)
    return len(tokens)

if __name__ == "__main__":
    run_mlir_opt_on_seeds()