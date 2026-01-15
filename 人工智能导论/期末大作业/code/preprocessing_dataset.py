import os
import sys
import zipfile
from PIL import Image
import shutil

_DATASET_NAME = "PetImages"
IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.tif', '.tiff']

def is_image_file(filename):
    """Judge whether it is a picture."""
    return any(filename.lower().endswith(extension) for extension in IMG_EXTENSIONS)

def delete_file(img_path):
    if os.path.isdir(img_path):
        os.rmdir(img_path)
    else:
        os.remove(img_path)
        print(f"Invalid image file, delete {img_path}")

def is_jpg(img_path):
    try:
        i=Image.open(img_path)
        return i.format =='JPEG'
    except IOError:
        return False

def filter_dataset(dataset_path):
    for sub_dir in os.listdir(dataset_path):
        for file_name in os.listdir(os.path.join(dataset_path, sub_dir)):
            img_path = os.path.join(os.path.join(dataset_path, sub_dir, file_name))
            if not (is_image_file(file_name) and is_jpg(img_path)):
                delete_file(img_path)
                continue

def split_dataset(dataset_path, eval_split=0.1):
    os.makedirs(os.path.join(dataset_path, "train"))
    os.makedirs(os.path.join(dataset_path, "eval"))
    for sub_dir in os.listdir(dataset_path):
        if sub_dir in ["train", "eval"]:
            continue
        cls_list = os.listdir(os.path.join(dataset_path, sub_dir))
        train_size = int(len(cls_list) * (1 - eval_split))
        os.makedirs(os.path.join(dataset_path, "train", sub_dir))
        os.makedirs(os.path.join(dataset_path, "eval", sub_dir))
        for i, file_name in enumerate(os.listdir(os.path.join(dataset_path, sub_dir))):
            source_file = os.path.join(dataset_path, sub_dir, file_name)
            if i <= train_size:
                target_file = os.path.join(dataset_path, "train", sub_dir, file_name)
            else:
                target_file = os.path.join(dataset_path, "eval", sub_dir, file_name)
            shutil.move(source_file, target_file)
        delete_file(os.path.join(dataset_path, sub_dir))

def extract_dataset(zip_file, save_dir):
    if not os.path.exists(zip_file):
        ValueError(f"{zip_file} is not a valid zip file!")
    try:
        print("extract dataset")
        zip_file = zipfile.ZipFile(zip_file)
        for names in zip_file.namelist():
            zip_file.extract(names, save_dir)
        zip_file.close()
        print(f"extract dataset at {os.path.join(save_dir, _DATASET_NAME)}")
    except:
        ValueError(f"{save_path} is not a valid zip file!")

if __name__ == '__main__':
    save_dir = os.path.abspath("./dataset")
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    zip_file = sys.argv[1]
    extract_dataset(zip_file, save_dir)
    print("filter invaild images!")
    dataset_path = os.path.join(save_dir, _DATASET_NAME)
    filter_dataset(dataset_path)
    print("filter invaild images done, then split dataset to train and eval")
    split_dataset(dataset_path, eval_split=0.1)
    print(f"final dataset at {dataset_path}")
