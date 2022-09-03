from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode, Visualizer
import cv2
import numpy
import os
os.environ['DISPLAY'] = ':0'
import glob
all_files = glob.glob("/misc/lmbssd/saikiat/datasets/coco/train2017/*")
import random

dim = []
randomlist = random.sample(range(0, len(all_files)), 10000)
for i in randomlist:
    imagePath = str(all_files[i])
    image = cv2.imread(imagePath)
    dim.append(image.shape)

import matplotlib.pyplot as plt
col = list(zip(*dim))
plt.hist(x = col[0], bins = 20)
plt.savefig("col[0].png")
plt.clf()
plt.hist(x = col[1], bins = 20)
plt.savefig("col[1].png")
plt.clf()
