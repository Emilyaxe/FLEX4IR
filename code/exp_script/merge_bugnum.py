import json
import os

input_dir = '../cov_collection/bug_num3'  
output_file = '../cov_collection/bug_num3/merged_summary.txt'
merged_results = []
cumulative_dict = {}

for hour in range(24):
    file_path = os.path.join(input_dir, f'bug_num_proc_{hour}.json')
    
    if not os.path.exists(file_path):
        print(f'Warning: {file_path} not found, skipping.')
        merged_results.append(cumulative_dict.copy())
        continue

    with open(file_path, 'r', encoding='utf-8') as f:
        current_data = json.load(f)

    cumulative_dict.update(current_data)

    merged_results.append(cumulative_dict.copy())

with open(output_file, 'w', encoding='utf-8') as out_f:
    # out_f.write("hour, bug_count, value_count\n") 
    for i, result in enumerate(merged_results, 1):
        bug_count = len(result)
        value_count = sum([len(v) if isinstance(v, list) else 1 for v in result.values()])
        out_f.write(f"{i}, {bug_count}, {value_count}\n")


