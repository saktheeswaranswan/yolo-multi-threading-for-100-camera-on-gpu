import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from collections import defaultdict
from io import StringIO
from matplotlib import pyplot as plt
from PIL import Image
from IPython.display import display
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util


!pip install -U --pre tensorflow=="2.*"
!pip install tf_slim
!pip install pycocotools
import os

import pathlib


if "models" in pathlib.Path.cwd().parts:
  while "models" in pathlib.Path.cwd().parts:
    os.chdir('..')
elif not pathlib.Path('models').exists():
  !git clone --depth 1 https://github.com/tensorflow/models
  
  
%%bash
cd models/research/
protoc object_detection/protos/*.proto --python_out=.

cp object_detection/packages/tf2/setup.py .
!python -m pip install .
%%bash 
cd models/research
pip install .

sudo apt-get install libgtk2.0-dev pkg-config


https://github.com/tensorflow/models/blob/master/research/object_detection/colab_tutorials/object_detection_tutorial.ipynb


build open cv from source
After that, you can rebuild and reinstall OpenCV library with the following commands:

bash
Copy code
cd ~/<opencv_build_directory>
cmake <opencv_source_directory>
make -j$(nproc)
sudo make install
Replace <opencv_build_directory> with the directory where you want to build OpenCV, and <opencv_source_directory> with the directory where you downloaded the OpenCV source code.

Once OpenCV is properly installed, you should
