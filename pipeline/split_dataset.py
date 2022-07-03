from ruamel.yaml import YAML
import os
import shutil
import glob
import random


def copy_files(imgs, set_folder, copy_label):
    to_copy = []
    for img_file in imgs:
        img_file2 = f'Data/Dataset/{set_folder}/images/' + img_file.split('images')[1]
        os.makedirs(os.path.split(img_file2)[0], exist_ok=True)
        
        if copy_label:
            shutil.copy(img_file, img_file2)
            label_file = img_file.replace('images', 'labels').replace('.jpg', '.txt')
            label_file2 = f'Data/Dataset/{set_folder}/labels/' + img_file.split('images')[1].replace('.jpg', '.txt')
            os.makedirs(os.path.split(label_file2)[0], exist_ok=True)
            shutil.copy(label_file, label_file2)
        else:
            to_copy.append((img_file, img_file2))
            
    if not copy_label:
        return to_copy


# Load Params
with open("params.yaml", "r") as fd:
    yaml = YAML()
    params = yaml.load(fd)

# Smoke Images
folders = glob.glob('Data/Dataset_All/smoke/*')

for folder in folders:
    path = os.path.normpath(folders[0])
    split = models = params["split_dataset"]["smoke"][path.split(os.sep)[-1]]
    imgs = glob.glob(folder + '/smoke/images/*')
    nb_train = int(len(imgs)*split[0])
    nb_val = int(len(imgs)*split[1])
    nb_test = int(len(imgs)*split[2])
    
    copy_files(imgs[:nb_train], set_folder='train', copy_label=True)
    copy_files(imgs[nb_train:nb_train+nb_val], set_folder='val', copy_label=True)
    copy_files(imgs[nb_train+nb_val:nb_train+nb_val+nb_val], set_folder='test', copy_label=True)


nb_smoke = len(glob.glob('Data/Dataset/**/images/*'))
nb_background_required = int(nb_smoke * params["split_dataset"]["background_smoke_ratio"])

# Background Images

folders = glob.glob('Data/Dataset_All/FP/**/*')

nb_background = 0
to_copy = []
for folder in folders:
    path = os.path.normpath(folders[0])
    split = models = params["split_dataset"]["FP"][path.split(os.sep)[-1]]
    imgs = glob.glob(folder + '/images/*')
    nb_train = int(len(imgs)*split[0])
    nb_val = int(len(imgs)*split[1])
    nb_test = int(len(imgs)*split[2])
    
    to_copy += copy_files(imgs[:nb_train], set_folder='train', copy_label=False)
    to_copy += copy_files(imgs[nb_train:nb_train+nb_val], set_folder='val', copy_label=False)
    to_copy += copy_files(imgs[nb_train+nb_val:nb_train+nb_val+nb_val], set_folder='test', copy_label=False)

    nb_background += nb_train+nb_val+nb_val


ratio = nb_background_required/nb_background

for img_file, img_file2 in to_copy:
    if random.random()<ratio:
        shutil.copy(img_file, img_file2)


for set_folder in ['train', 'val', 'test']:
    print(f"{set_folder} images {len(glob.glob(f'Data/Dataset/{set_folder}/images/*.jpg'))} images")

print(f"Total images {len(glob.glob(f'Data/Dataset/**/images/*.jpg'))} images")
 
