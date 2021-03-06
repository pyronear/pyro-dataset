# Copyright (C) 2019-2022, Pyronear.

# This program is licensed under the Apache License version 2.
# See LICENSE or go to <https://www.apache.org/licenses/LICENSE-2.0.txt> for full license details.

"""
Package installation setup
"""

import os
import re
import subprocess
from pathlib import Path

from setuptools import find_packages, setup

version = '0.0.1.dev0'
sha = 'Unknown'
src_folder = 'pyrodataset'
package_index = 'pyrodataset'

cwd = Path(__file__).parent.absolute()

if os.getenv('BUILD_VERSION'):
    version = os.getenv('BUILD_VERSION')
else:
    try:
        sha = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=cwd).decode('ascii').strip()
    except Exception:
        pass
    if sha != 'Unknown':
        version += '+' + sha[:7]
print(f"Building wheel {package_index}-{version}")

with open(cwd.joinpath(src_folder, 'version.py'), 'w') as f:
    f.write(f"__version__ = '{version}'\n")

with open('README.md') as f:
    readme = f.read()

_deps = [
    "opencv-python>=3.4.5.20",
    "pandas>=0.25.2",
    "torch>=1.8.0",
    "torchvision>=0.9.0",
    "tqdm>=4.20.0",
    "requests>=2.20.0",
    "ipywidgets>=7.5.1",
    "moviepy>=1.0.3",
    "pytesseract>=0.3.4",
    # Testing
    "PyYAML>=5.1.2",
    "youtube-dl>=2020.3.24",
    "pafy>=0.5.5",
    "coverage>=4.5.4",
    # Quality
    "flake8>=3.9.0",
    "isort>=5.7.0",
    "mypy>=0.812",
    "pydocstyle>=6.0.0",
    # Docs
    "sphinx<=3.4.3,<3.5.0",
    "sphinx-rtd-theme==0.4.3",
    "docutils<0.18",
]

# Borrowed from https://github.com/huggingface/transformers/blob/master/setup.py
deps = {b: a for a, b in (re.findall(r"^(([^!=<>]+)(?:[!=<>].*)?$)", x)[0] for x in _deps)}


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


install_requires = [
    deps["opencv-python"],
    deps["pandas"],
    deps["torch"],
    deps["torchvision"],
    deps["tqdm"],
    deps["requests"],
    deps["ipywidgets"],
    deps["moviepy"],
    deps["pytesseract"]
]

extras = {}

extras["testing"] = deps_list(
    "PyYAML",
    "youtube-dl",
    "pafy",
    "coverage",
)

extras["quality"] = deps_list(
    "flake8",
    "isort",
    "mypy",
    "pydocstyle",
)

extras["docs"] = deps_list(
    "sphinx",
    "sphinx-rtd-theme",
    "docutils",
)

extras["dev"] = (
    extras["testing"]
    + extras["quality"]
    + extras["docs"]
)


setup(
    name=package_index,
    version=version,
    author='PyroNear Contributors',
    author_email='pyronear.d4g@gmail.com',
    maintainer='Pyronear',
    description='Datasets and models for wildfire detection in PyTorch',
    long_description=readme,
    long_description_content_type="text/markdown",
    url='https://github.com/pyronear/pyro-dataset',
    download_url='https://github.com/pyronear/pyro-dataset/tags',
    license='Apache',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords=['pytorch', 'deep learning', 'vision', 'models', 'wildfire', 'object detection'],
    packages=find_packages(exclude=('test',)),
    zip_safe=True,
    python_requires='>=3.6.0',
    include_package_data=True,
    install_requires=install_requires,
    extras_require=extras,
    package_data={'': ['LICENSE']},
)
