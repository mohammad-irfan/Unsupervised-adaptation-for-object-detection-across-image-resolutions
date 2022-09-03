from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
from detectron2.utils.visualizer import ColorMode, Visualizer
import cv2
from pycocotools.coco import COCO
import os
import matplotlib.pyplot as plt
import skimage.io as io
import detectron2
from thesis.register_datasets import register_cityscapes
from register_datasets_old import *
import torch
import numpy as np
import numpy as np
import os, json, cv2, random
json_file =  "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_val_new.json"
train_name, val_name = register_cityscapes(train_json= "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_train_new.json",
                                            val_json = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_val_new.json")
cfg = get_cfg()
file = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
cfg.merge_from_file(model_zoo.get_config_file(file))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(file)
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda"
# cfg.merge_from_file("/misc/student/mirfan/config_files/cityscapes_R_50_FPN_eval.yaml")
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.DEVICE = "cuda"
cfg.DATASETS.TRAIN = "cityscapes_fine_detection_train"
cfg.DATASETS.TEST = "cityscapes_fine_detection_val"
from pathlib import Path
predictor = DefaultPredictor(cfg)
from shapely.geometry import Polygon


def calculate_iou(box_1, box_2):
    poly_1 = Polygon(box_1)
    poly_2 = Polygon(box_2)
    iou = poly_1.intersection(poly_2).area / poly_1.union(poly_2).area
    return iou

# def make_ann_box(bbox):
#     box = []
#     box.append([bbox[0],bbox[1]])
#     box.append([bbox[0],bbox[1] + bbox[3]])
#     box.append([bbox[0]+ bbox[2],bbox[1] + bbox[3]])
#     box.append([bbox[0]+ bbox[2],bbox[1]])
#     return box

# def make_pred_box(bbox):
#     if type(bbox) == type(torch.tensor([0])):
#       bbox = list(bbox.to(torch.int).detach().cpu().numpy())  
#     box = []
#     box.append([bbox[0],bbox[1]])
#     box.append([bbox[0],bbox[3] ])
#     box.append([bbox[2],bbox[1]])
#     box.append([bbox[2],bbox[3]])
#     return box

def make_ann_box(bbox):
    box={}
    box["x1"] = bbox[0]
    box["x2"] = bbox[0]+ bbox[2]
    box["y1"] = bbox[1]
    box["y2"] = bbox[1] + bbox[3]
    return box

def make_pred_box(bbox):
    if type(bbox) == type(torch.tensor([0])):
      bbox = list(bbox.to(torch.int).detach().cpu().numpy())  
    box={}
    box["x1"] = bbox[0]
    box["x2"] = bbox[2]
    box["y1"] = bbox[1]
    box["y2"] = bbox[3]
    return box

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def prediction(imagePath, cfg, name = "0.png"):
    name = "/misc/student/mirfan/cityscapeoutputs/" + name
    Path(name).parent.mkdir(parents=True, exist_ok=True)
    image = cv2.imread(imagePath)
    predictions = predictor(image)
    annIds = coco.getAnnIds(imgIds=[data['id']])
    anns = coco.loadAnns(annIds)
    true_pred = [make_ann_box(i["bbox"]) for i in anns]
    pred = [make_pred_box(i) for i in predictions["instances"].pred_boxes.tensor.to(torch.int)]
    index = []
    for i, p in enumerate(pred):
        for j in true_pred:
            if get_iou(p,j) > 0.5:
                index.append(i)
    
    false_positives = list(set([i for i in range(len(pred))]) - set(index))
    # a = detectron2.structures.instances.Instances(predictions["instances"] )
    # predictions["instances"].pred_boxes = predictions["instances"].pred_boxes[false_positives]
    viz = Visualizer(image[:,:,::-1], metadata =  MetadataCatalog.get("coco_2017_train"), scale=1.2, 
    instance_mode = ColorMode.IMAGE)
    if len(predictions["instances"][false_positives]) > 0 :
        output = viz.draw_instance_predictions(predictions["instances"][false_positives].to("cpu"))
        cv2.imwrite(name, output.get_image()[:,:,::-1])

coco_api = COCO(json_file)

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
    imagePaths = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/" +imagePath
    # image = io.imread(imagePath)
    prediction(imagePaths, cfg, name = imagePath)
    # annIds = coco.getAnnIds(imgIds=[data['id']])
    # anns = coco.loadAnns(annIds)
    # coco.loadCats(anns['category_id'])[0]["name"]
    # v = Visualizer(image[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TEST[0]), scale=1.2)
    # out = v.draw_box([i["segmentation"] for i in anns])
    # plt.figure(figsize=(20,10))
    # plt.imshow(out.get_image()[..., ::-1][..., ::-1])

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

