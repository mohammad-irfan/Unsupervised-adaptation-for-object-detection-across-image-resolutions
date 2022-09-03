import cv2
from pathlib import Path
import glob
import json
import numpy as np
import orjson 
import numpy as np

def np_encoder(object):
    if isinstance(object, np.generic):
        return object.item()


all_files = glob.glob("/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/leftImg8bit/*/*/*.png")
new_folder = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes_.75/leftImg8bit/"
scale = 0.75
for f in all_files:
    img = cv2.imread(f)
    x_ = img.shape[1]
    y_ = img.shape[0]
    resized = cv2.resize(img, (int(x_*scale), int(y_*scale)))
    name = "/".join(f.split("/")[-3:])
    new_path = new_folder + name
    Path(new_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(new_path, resized)


Path("/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes_.75/annotations").parent.mkdir(parents=True, exist_ok=True)
file_cityscapes = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_train_new.json"
a = open(file_cityscapes, "r")
cityscapes = json.load(a)
ann = []
im = []
for i in cityscapes["annotations"]:
    i["bbox"] = list((np.array(i["bbox"])*scale).astype(int))
    ann.append(i)
for i in cityscapes["images"]:
    i["width"] = int(i["width"]*scale)
    i["height"] = int(i["height"]*scale)
    im.append(i)

cityscapes["annotations"] = ann
cityscapes["images"] = im
# cityscapes["categories"] = [{'id': 1, 'name': 'person'}, {'id': 2, 'name': 'bicycle'}, {'id': 3, 'name': 'car'}, {'id': 4, 'name': 'motorcycle'},  {'id': 6, 'name': 'bus'}, {'id': 7, 'name': 'train'}, {'id': 8, 'name': 'truck'}]
with open("/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes_.75/annotations/instancesonly_filtered_gtFine_train_new.json", "w") as f:
    json.dump(cityscapes,f, default=np_encoder)

file_cityscapes = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_val_new.json"
a = open(file_cityscapes, "r")
cityscapes = json.load(a)
ann = []
im = []
for i in cityscapes["annotations"]:
    i["bbox"] = list((np.array(i["bbox"])*scale).astype(int))
    ann.append(i)
for i in cityscapes["images"]:
    i["width"] = int(i["width"]*scale)
    i["height"] = int(i["height"]*scale)
    im.append(i)

cityscapes["annotations"] = ann
cityscapes["images"] = im
# cityscapes["categories"] = [{'id': 1, 'name': 'person'}, {'id': 2, 'name': 'bicycle'}, {'id': 3, 'name': 'car'}, {'id': 4, 'name': 'motorcycle'},  {'id': 6, 'name': 'bus'}, {'id': 7, 'name': 'train'}, {'id': 8, 'name': 'truck'}]
with open("/misc/lmbraid19/mirfan/cityscapes-to-cxco-conversion/data/cityscapes_.75/annotations/instancesonly_filtered_gtFine_val_new.json", "w") as f:
    json.dump(cityscapes,f, default=np_encoder)

