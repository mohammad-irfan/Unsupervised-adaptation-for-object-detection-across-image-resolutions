from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode, Visualizer
import cv2
import numpy
from pycocotools.coco import COCO
from pathlib import Path
from pycocotools.cocoeval import COCOeval
import os
import contextlib
import matplotlib.pyplot as plt
import json
from thesis.register_datasets import register_cityscapes
from register_datasets import *
import torch
import numpy as np
import numpy as np
import os, json, cv2, random
json_file = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes_.75/annotations/instancesonly_filtered_gtFine_val_new.json"
train_json_file = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes_.75/annotations/instancesonly_filtered_gtFine_train_new.json"

train_name, val_name = register_cityscapes(val_json = json_file, train_json = train_json_file)
cfg = get_cfg()
file = "COCO-Detection/faster_rcnn_R_50_C4_1x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(file))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(file)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = "cuda"
# cfg.merge_from_file("/misc/student/mirfan/config_files/cityscapes_R_50_FPN_eval.yaml")
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_C4_1x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = "cuda"
cfg.DATASETS.TRAIN = "cityscapes_fine_detection_train"
cfg.DATASETS.TEST = "cityscapes_fine_detection_val"
predictor = DefaultPredictor(cfg)

def prediction(imagePath, data, a,name = "0.png"):
    imagePath = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/" +imagePath
    image = cv2.imread(imagePath)
    predictions = predictor(image)

    viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0]), 
    instance_mode = ColorMode.IMAGE)

    output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
    
    cv2.imwrite(name, output.get_image()[:,:,::-1])

coco_api = COCO(train_json_file)

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
from detectron2.engine import DefaultTrainer

# trainer = DefaultTrainer(cfg) 
# trainer.resume_or_load(resume=True)
# loader = trainer.build_test_loader(cfg, val_name)
file = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
loader = coco_api.loadAnns(coco_api.getAnnIds(imgIds=coco_api.getImgIds(), catIds=coco_api.getCatIds()))
coco = coco_api
for data in coco.loadImgs(coco.getImgIds()):
    _,axs = plt.subplots(1,1)
    imagePath = data["file_name"]
    imagePath = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes_.75/" +imagePath
    image = cv2.imread(imagePath)
    annIds = coco.getAnnIds(imgIds=[data['id']])
    anns = coco.loadAnns(annIds)
    coco.loadCats([a["category_id"] for a in anns if a["image_id"]==data["id"] ])
    # v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
    # out = v.draw_box([i["segmentation"] for i in anns])
    # plt.figure(figsize=(20,10))
    # plt.imshow(out.get_image()[..., ::-1][..., ::-1])
    axs.imshow(image)
    plt.sca(axs)
    coco.showAnns(anns, draw_bbox=True)
    a = "/misc/student/mirfan/cityscapeoutputs/.75/"
    os.makedirs(Path(a + data["file_name"]).parent, exist_ok=True)
    plt.savefig(a+data["file_name"])
    # plt.show()
    # prediction(imagePath, data, a, "/misc/student/mirfan/Results/cityscapes_GT/"+str(imagePath.split("/")[-1]))

from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
evaluator = COCOEvaluator(val_name, output_dir="./output")
val_loader = build_detection_test_loader(cfg, val_name)
print(inference_on_dataset(predictor.model, val_loader, evaluator))

print(cfg["INPUT"]["MIN_SIZE_TEST"])
print(cfg["INPUT"]["MAX_SIZE_TEST"])

def onlykeep_person_class(outputs):
  cls = outputs['instances'].pred_classes
  scores = outputs["instances"].scores
  boxes = outputs['instances'].pred_boxes

  # index to keep whose class == 0
  indx_to_keep = (cls == 0).nonzero().flatten().tolist()
    
  # only keeping index  corresponding arrays
  cls1 = torch.tensor(np.take(cls.cpu().numpy(), indx_to_keep))
  scores1 = torch.tensor(np.take(scores.cpu().numpy(), indx_to_keep))
  boxes1 = Boxes(torch.tensor(np.take(boxes.tensor.cpu().numpy(), indx_to_keep, axis=0)))
  
  # create new instance obj and set its fields
  obj = detectron2.structures.Instances(image_size=(oim.shape[0], oim.shape[1]))
  obj.set('pred_classes', cls1)
  obj.set('scores', scores1)
  obj.set('pred_boxes',boxes1)
  return obj

