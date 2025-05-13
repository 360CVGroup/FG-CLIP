import json
import os
from multiprocessing import Pool, cpu_count
from tqdm import tqdm
import shutil
import argparse



def process_parent_folder(parent_idx):
    parent_folder_path = os.path.join(down_file_root, f"coyo_image_{parent_idx}")

    resave_parent_folder_path = os.path.join(resave_file_root, f"coyo_image_{parent_idx}")
    os.makedirs(resave_parent_folder_path, exist_ok=True)

    for k in tqdm(range(num_subfolders_per_parent), desc=f"Processing subfolders in {parent_idx}"):
        folder_name = f"{k:05d}"
        folder_path = os.path.join(parent_folder_path, folder_name)

        if not os.path.exists(folder_path):
            print(f"Folder {folder_path} does not exist. Skipping.")
            continue

        nested_folder_path = os.path.join(resave_parent_folder_path, folder_name)
        os.makedirs(nested_folder_path, exist_ok=True)

        for filename in os.listdir(folder_path):
            if filename.endswith('.json'):
                file_path = os.path.join(folder_path, filename)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)

                    url = data.get('url')
                    if url in url2prekey:
                        down_img_name = data.get('key') + ".jpg"
                        full_down_image_name = os.path.join(folder_path, down_img_name)
                        
                        rename_key = url2prekey[url]
                        image_destination_path = os.path.join(nested_folder_path, rename_key + ".jpg")
                        if os.path.exists(full_down_image_name) and not os.path.exists(image_destination_path):
                            shutil.copy(full_down_image_name, image_destination_path)
                        data["key"] = rename_key

                        json_destination_path = os.path.join(nested_folder_path, rename_key + ".json")
                        with open(json_destination_path, 'w', encoding='utf-8') as f:
                            json.dump(data, f, indent=4, ensure_ascii=False)

                except (json.JSONDecodeError, IOError) as e:
                    print(f"Error processing {file_path}: {e}")
                except Exception as e:
                    print(f"Unexpected error while processing {file_path}: {e}")


def main(args):

    global url2prekey
    global down_file_root
    global resave_file_root
    global num_parent_folders
    global num_subfolders_per_parent

    url2key_json_root = args.url2key_json
    url2prekey = []

    for filename in os.listdir(url2key_json_root):
        url2key_json_name_full = os.path.join(url2key_json_root, filename)

        with open(url2key_json_name_full, 'r', encoding='utf-8') as file:
            url2key = json.load(file)

        url2prekey += url2key

    down_file_root = args.down_file_root
    resave_file_root = args.resave_file_root
    num_parent_folders = args.num_parent_folders  
    num_subfolders_per_parent = args.num_subfolders_per_parent    


    num_processes = min(cpu_count(), num_parent_folders)
    print(f"Using {num_processes} processes...")

    pool = Pool(processes=num_processes)
    results = [pool.apply_async(process_parent_folder, (idx,)) for idx in range(num_parent_folders)]
    
    # 使用 tqdm 显示进度条
    for result in tqdm(results, total=len(results), desc="Processing chunks"):
        try:
            result.get()
        except Exception as e:
            print(f"Chunk processing failed with error: {e}")

    pool.close()
    pool.join()





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url2key_json", type=str, default="url2key.json")
    parser.add_argument("--down_file_root", type=str, default="down_file_root")
    parser.add_argument("--num_parent_folders", type=int, default=20)
    parser.add_argument("--resave_file_root", type=str, default="resave_file_root")
    parser.add_argument("--num_subfolders_per_parent", type=int, default=100)
    args = parser.parse_args()

    main(args)