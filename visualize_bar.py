import os
import json
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm

def list_files_in_directory(directory_path):
    try:
        with os.scandir(directory_path) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.endswith('.npy'):
                    yield entry.name
    except Exception as e:
        print(f"An error occurred: {e}")
        return []

def parse_file(file):
    try:
        number, id_with_ext = file.split('-')
        id = id_with_ext.replace('.npy', '')
        return number, id
    except ValueError:
        print(f"Filename {file} does not match the expected format")
        return None, None

def process_files(directory_path, num_threads):
    result = defaultdict(list)
    files_gen = list_files_in_directory(directory_path)
    total_files = sum(1 for _ in files_gen)
    files_gen = list_files_in_directory(directory_path)  # 重新生成生成器对象

    with tqdm(total=total_files, desc="Processing files", unit="files") as pbar:
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            for number, id in executor.map(parse_file, files_gen):
                if number and id:
                    result[number].append(id)
                pbar.update(1)  # 更新进度条

    return result

def save_to_json(data, output_file):
    try:
        with open(output_file, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=4)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process files in a directory and save results to JSON.')
    parser.add_argument('-j', '--jobs', type=int, default=1, help='Number of concurrent jobs (threads)')
    args = parser.parse_args()

    directory_path = '/data/work-gcp-europe-west4-a/bin_ren/point-cloud/ShapeNet55-34/shapenet_pc'
    num_threads = args.jobs

    parsed_data = process_files(directory_path, num_threads)
    
    output_file = 'output.json'  # 输出的JSON文件路径
    save_to_json(parsed_data, output_file)
    print(f"Parsed file names have been written to {output_file}")
