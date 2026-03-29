import os
import math
import subprocess
from pathlib import Path
import json


profdata_folder = Path('../cov_collection/line_cov_final')
output_folder = profdata_folder / 'merged_outputs'
output_folder.mkdir(exist_ok=True)
cov_result = output_folder / 'cov_result.txt'



def put_file_content(path, content):
    directory = os.path.dirname(path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    f = open(path, 'a+')
    f.write(content)
    f.flush()
    f.close()

def extract_coverage_metrics(json_path):
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            branches = data["data"][0]["totals"]["branches"]["percent"]
            lines = data["data"][0]["totals"]["lines"]["percent"]
            return branches, lines
    except Exception as e:
        print(f"⚠️ Failed to process {json_path.name}: {e}")
        return None, None


def split_files_equally(file_list, num_parts):
    avg = math.ceil(len(file_list) / num_parts)
    return [file_list[i * avg:(i + 1) * avg] for i in range(num_parts)]

def merge_profdata(merge_files, merge_output, export_output):
    command = (' ').join(['llvm-profdata merge -j 32', (' ').join(merge_files), '-o', merge_output])
    print(f"Running merge... ")
    subprocess.run(command, text=True, check=True, shell=True)
    cov_dir = '../llvm-cov/llvm-project'
    report_cmd = (' ').join(['llvm-cov export -j 32 -summary-only', '../llvm-cov/llvm-project/build/bin/mlir-opt', f'-instr-profile={merge_output}', f"--ignore-filename-regex='^{cov_dir}/(llvm|build)'", '1>', export_output])
    print(f"Running export... ")
    subprocess.run(report_cmd, text=True, check=True, shell=True)    

def main():
    all_files = [str(f) for f in profdata_folder.glob('tmp.*.profdata')]
    if len(all_files) == 0:
        print("❌ No .profdata files found.")
        return
    parts = split_files_equally(all_files, 24)

    for i in range(1, 25):
        print(f"The {i}th merge step...")
        files_to_merge = sum(parts[:i], [])  
        merge_path = output_folder / f'merged_{i:02d}.profdata'
        export_path = output_folder / f'merged_{i:02d}.json'
        merge_profdata(files_to_merge, str(merge_path), str(export_path))
        # print("files_to_merge", files_to_merge)
        print(f"✅ Merging {len(files_to_merge)} files into {merge_path}")
        print(f"✅ Exporting to {export_path}")
        branches, lines = extract_coverage_metrics(str(export_path))
        if branches is not None and lines is not None:
            put_file_content(cov_result, f'{i} {branches} {lines} {len(files_to_merge)}\n')

    print("✅ All merge steps completed.")

if __name__ == "__main__":
    main()


