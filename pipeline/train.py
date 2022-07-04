import subprocess
import pandas as pd
import json
from ruamel.yaml import YAML


# Load Params
with open("params.yaml", "r") as fd:
    yaml = YAML()
    params = yaml.load(fd)

img_size = params["train"]["img_size"]
batch_size = params["train"]["batch_size"]
epochs = params["train"]["epochs"]
model = params["train"]["model"]
workers = params["train"]["workers"]

cmd = f"python yolov5/train.py --img {img_size} --batch {batch_size} --epochs {epochs} --data ./yolo.yaml --weights {model}.pt --project runs/train --workers {workers}"
print(f"* Command:\n{cmd}")
subprocess.call(cmd, shell=True)

# Log Metrics

df = pd.read_csv("runs/train/exp/results.csv")
metrics = {}
for c in df.columns:
    val = df.iloc[-1][c]
    metrics[c] = val

with open("metrics.json", "w") as outfile:
    json.dump(metrics, outfile)
