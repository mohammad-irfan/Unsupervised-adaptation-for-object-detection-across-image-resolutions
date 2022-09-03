# from re import L
# from detectron2.config import get_cfg
# from detectron2.data import MetadataCatalog
# from detectron2 import model_zoo
# import os
# from register_datasets import *
# import torch
# import numpy as np
# import os
# import logging
# import os
# import torch
# from torch.nn.parallel import DistributedDataParallel
# from detectron2.evaluation import COCOEvaluator, inference_on_dataset
# from detectron2.data import build_detection_test_loader
# import detectron2.utils.comm as comm
# from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
# from detectron2.config import get_cfg
# from detectron2.data import (
#     MetadataCatalog,
#     build_detection_test_loader,
#     build_detection_train_loader,
# )
# from detectron2.engine import default_writers, default_setup, default_argument_parser, launch
# from detectron2.evaluation import (
#     COCOEvaluator,
#     inference_on_dataset,
# )
# from detectron2.modeling import build_model
# from detectron2.solver import build_lr_scheduler, build_optimizer
# from detectron2.utils.events import EventStorage
from re import L
from detectron.config import get_cfg
from detectron2 import model_zoo
import os
from register_datasets import *
import torch
import os
import logging
import os
import torch
import numpy as np
from torch.nn.parallel import DistributedDataParallel
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
import detectron.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron.config import get_cfg
from detectron.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron.engine import default_writers, default_setup, default_argument_parser, launch
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
)
from detectron.modeling import build_model
from detectron.utils.events import EventStorage
from prettytable import PrettyTable
import torch.nn.functional as F
import torch.nn as nn

logger = logging.getLogger("detectron2")

from prettytable import PrettyTable
para = {}
change = []
notrain_params = []
output_dir = "checkcpu/"
def do_train(cfg, model, val_name, resume=False):
    model.train()
    from time import time
    import multiprocessing as mp
    for num_workers in range(8, 0, -1):  
        cfg.DATALOADER.NUM_WORKERS = num_workers
        data_loader = build_detection_train_loader(cfg, is_source = True)
        start = time()
        for data, iteration in zip(data_loader, range(0, 1000)):
            pass
        end = time()
        print("Finish with:{} second, num_workers={}".format(end - start, num_workers))
    

def main(args):
    train_name, val_name = register_cityscapes(train_json= "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes_.75/annotations/instancesonly_filtered_gtFine_train_new.json",
                                            train_dir = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes_.75/",
                                            val_dir = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes_.75/",
                                            val_json = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes_.75/annotations/instancesonly_filtered_gtFine_val_new.json",
                                            train_name= "cityscapes_fine_detection_train",
                                            val_name="cityscapes_fine_detection_val")
    cfg = get_cfg()
    cfg.merge_from_list(args.opts)
    file = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(file)
    cfg.MODEL.DEVICE = "cuda"
    cfg.merge_from_file("/misc/student/mirfan/config_files/coco_R_50_FPN.yaml")
    cfg.set_new_allowed(True) 
    cfg.merge_from_file("/misc/student/mirfan/config_files/cityscapes_0.75_R_50_FPN_withCocoBase.yaml")
    default_setup(cfg, args)
    cfg.OUTPUT_DIR = "/misc/lmbraid19/mirfan/output/" + output_dir
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    print("Counting parameters before freezing")
    distributed = comm.get_world_size() > 1
    print("Distributed = ")
    print(distributed)
    print("")
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    print(cfg)
    do_train(cfg, model, val_name, resume=True)

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

