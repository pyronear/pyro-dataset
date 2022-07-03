import subprocess
from ruamel.yaml import YAML
import glob
import os


# Load Params
with open("params.yaml", "r") as fd:
    yaml = YAML()
    params = yaml.load(fd)

models = params["auto_label"]["models"]
img_size = params["auto_label"]["img_size"]
conf = params["auto_label"]["conf"]

folders = glob.glob('Data/All_Data/**/**/images')

for folder in folders:
    if not os.path.isdir(folder.replace('images', 'labels')):
        name = folder.split('All_Data')[1].split('images')[0][1:]
        cmd = f"python yolov5/detect.py --weights {models} --img {img_size} --conf {conf} --source {folder}  --nosave \
               --save-txt --project Data/Labels --name {name}"
        subprocess.call(cmd, shell=True)
