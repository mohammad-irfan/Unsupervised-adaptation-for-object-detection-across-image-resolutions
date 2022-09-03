from detectron.engine import DefaultPredictor
from detectron.config import get_cfg
from detectron.data import MetadataCatalog
from detectron2 import model_zoo
from detectron.utils.visualizer import ColorMode, Visualizer
import cv2
from detectron.engine import default_writers, default_setup, default_argument_parser, launch
import numpy
import detectron.utils.comm as comm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import os
import contextlib
from torch.nn.parallel import DistributedDataParallel
import io
from thesis.register_datasets import register_cityscapes
# from register_datasets_old import *
import torch
import numpy as np
import numpy as np
import os, json, cv2, random

def main(args):
    json_file = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_val_new.json"
    train_name, val_name = register_cityscapes(val_json = json_file)
    cfg = get_cfg()
    cfg.set_new_allowed(True) 
    file = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(file)
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.DEVICE = "cuda"
    cfg.merge_from_file("/misc/student/mirfan/config_files/cityscapes.75_cityscapes_R_50_FPN.yaml")
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")

    from detectron.solver import build_lr_scheduler, build_optimizer
    from detectron.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.DEVICE = "cuda"
    cfg.DATASETS.TRAIN = "cityscapes_fine_detection_train"
    cfg.DATASETS.TEST = "cityscapes_fine_detection_val"
    cfg.INPUT.MIN_SIZE_TEST = 1024
    cfg.INPUT.MAX_SIZE_TEST = 2048
    predictor = DefaultPredictor(cfg)

    def prediction(imagePath, name = "0.png"):
        image = cv2.imread(imagePath)
        predictions = predictor(image)

        viz = Visualizer(image[:,:,::-1], metadata = MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), 
        instance_mode = ColorMode.IMAGE)

        output = viz.draw_instance_predictions(predictions["instances"].to("cpu"))
        
        cv2.imwrite(name, output.get_image()[:,:,::-1])
    with contextlib.redirect_stdout(io.StringIO()):
        coco_api = COCO(json_file)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    from detectron.engine import DefaultTrainer

    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    loader = trainer.build_test_loader(cfg, val_name)
    file = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    # loader = coco_api.loadAnns(coco_api.getAnnIds(imgIds=coco_api.getImgIds(), catIds=coco_api.getCatIds()))
    # for data in loader:
    #     imagePath = data[0]["file_name"]
    #     prediction(imagePath, "/misc/student/mirfan/Results/cityscapes_GT/"+str(imagePath.split("/")[-1]))

    from detectron.evaluation import COCOEvaluator, inference_on_dataset
    from detectron.data import build_detection_test_loader
    evaluator = COCOEvaluator(val_name, output_dir="./output")
    val_loader = build_detection_test_loader(cfg, val_name)
    optimizer = build_optimizer(cfg, predictor.model)
    scheduler = build_lr_scheduler(cfg, optimizer)
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(file)

    checkpointer = DetectionCheckpointer(
        predictor.model, "/misc/lmbraid19/mirfan/", scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=True).get("iteration", -1) + 1
    )
    predictor.model = DistributedDataParallel(
        predictor.model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
        #  find_unused_parameters=True
    )

    DetectionCheckpointer(predictor.model).load("/misc/lmbraid19/mirfan/output/c7_lr0.00125_rpn/model_final.pth")
    print(inference_on_dataset(predictor.model, val_loader, evaluator))

    print("MIN_SIZE_TEST ", cfg["INPUT"]["MIN_SIZE_TEST"])
    print("MAX_SIZE_TEST ", cfg["INPUT"]["MAX_SIZE_TEST"])

if __name__ == "__main__":
    import time

    # Grab Currrent Time Before Running the Code
    start = time.time()

    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
    end = time.time()

    #Subtract Start Time from The End Time
    total_time = end - start
    print("\n"+ str(total_time/(60*60)))