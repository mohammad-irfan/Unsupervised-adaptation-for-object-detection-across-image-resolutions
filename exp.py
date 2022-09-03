from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode, Visualizer
import cv2
import numpy
import os
os.environ['DISPLAY'] = ':0'
from register_datasets import *
# train_name, val_name = register_coco()


cfg = get_cfg()
file = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(file))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(file)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = "cuda"
cfg["INPUT"]["MIN_SIZE_TEST"] = 200
cfg["INPUT"]["MAX_SIZE_TEST"] =  400
from detectron2.engine import DefaultTrainer
# predictor = DefaultPredictor(cfg)
trainer = DefaultTrainer(cfg) 
predictor = trainer
trainer.resume_or_load(resume=True)
def prediction(imagePath, name = "0.png"):
    image = cv2.imread(imagePath)
    predictions = predictor(image)

    viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
    instance_mode = ColorMode.IMAGE)

    output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
    
    cv2.imwrite(name, output.get_image()[:,:,::-1])

import glob
all_files = glob.glob("/misc/lmbssd/saikiat/datasets/coco/train2017/*")
import random
dim = []
randomlist = random.sample(range(0, len(all_files)), 1000)
for i in randomlist:
    imagePath = str(all_files[i])
    prediction(imagePath, "/misc/student/mirfan/coco/"+str(imagePath.split("/")[-1]))
