![PyroNear Logo](docs/source/_static/img/pyronear-logo-dark.png)

<p align="center">
    <a href="LICENSE" alt="License">
        <img src="https://img.shields.io/badge/License-AGPL%20v3-blue.svg" /></a>
    <a href="https://www.codacy.com/gh/pyronear/pyro-dataset/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=pyronear/pyro-dataset&amp;utm_campaign=Badge_Grade">
        <img src="https://app.codacy.com/project/badge/Grade/7f17d9f2448248dd93d84331e93523e1"/></a>
    <a href="https://github.com/pyronear/pyro-dataset/actions?query=workflow%3Apython-package">
        <img src="https://github.com/pyronear/pyro-dataset/workflows/python-package/badge.svg" /></a>
    <a href="https://codecov.io/gh/pyronear/pyro-dataset">
  		<img src="https://codecov.io/gh/pyronear/pyro-dataset/branch/master/graph/badge.svg" />
	</a>
    <a href="https://pyronear.github.io/pyro-dataset">
  		<img src="https://img.shields.io/badge/docs-available-blue.svg" /></a>
    <a href="https://pypi.org/project/pyrodataset/" alt="Pypi">
        <img src="https://img.shields.io/badge/pypi-v0.1.1-blue.svg" /></a>
</p>




# Pyrodataset: Create dataset about wildfire

This repository gathers everything needed to create and train our smoke detection model.

We have 4 data sources:

Wildfire containing 20K (around 9K with smoke) images from 1000 videos of North American forest observation

Ardeche containing 20K images but no smoke that we aquired in ardeche this year.

Ai for Mankind, 2934 images of smoke and 1440 without smoke

Random, a dataset of images scraped from the internet 329 with smoke and 605 with low clouds

We use [dvc](https://dvc.org/) to track any changes made during dataset generation

Our detection model is a yolov5, we use [ultralytics](https://github.com/ultralytics/yolov5) repository for training

## Setup

Clone this pyro-dataset and [ultralytics](https://github.com/ultralytics/yolov5) repository using:

```shell
git clone --recurse-submodules https://github.com/pyronear/pyro-dataset.git
cd pyro-dataset
pip install -r requirements.txt
pip install -r yolov5/requirements.txt
```

Then pull data from remote using dvc (you well need to ask access using your google account)

```shell
dvc pull
```

Finally, reproduce the dataset generation and training using 

```shell
dvc repro
```

At the moment you shall not use dvc push and we can't store all the dvc cache on the remote because we use google drive which is limited. A bucket will be opened soon allowing to use the full power of dvc

## Test

After training you can validate the performances on test dataset using:

```shell
python yolov5/val.py --task test --weights runs/train/exp/weights/best.pt --img 640  --data yolo.yaml
```

Or just have a look on predictions using:

```shell
python yolov5/detect.py --weights runs/train/exp/weights/best.pt --img 640 
```

For more details please refer to [ultralytics](https://github.com/ultralytics/yolov5) repository



## What else

### Documentation

The full package documentation is available [here](https://pyronear.org/pyro-dataset/) for detailed specifications.

## Citation

If you wish to cite this project, feel free to use this [BibTeX](http://www.bibtex.org/) reference:

```bibtex
@misc{pyrodataset2019,
    title={Pyrodataset: wildfire early detection},
    author={Pyronear contributors},
    year={2019},
    month={October},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/pyronear/pyro-dataset}}
}
```


## Contributing

Please refer to [`CONTRIBUTING`](CONTRIBUTING.md) to help grow this project!



## License

Distributed under the Apache 2 License. See [`LICENSE`](LICENSE) for more information.
