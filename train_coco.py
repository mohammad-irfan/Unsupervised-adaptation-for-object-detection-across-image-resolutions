from re import L
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import os
from register_datasets_old import *
import torch
import numpy as np
import os
import logging
import os
import torch
from torch.nn.parallel import DistributedDataParallel
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_writers, default_setup, default_argument_parser, launch
from detectron2.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import EventStorage

logger = logging.getLogger("detectron2")

from prettytable import PrettyTable
para = {}
change = []
notrain_params = []
output_dir = "trial/"
def check_parameters(model):
    global para, change
    for name, parameter in model.named_parameters():
        if name in para and not torch.equal(para[name], parameter):
            change.append(name)
            print("SOMETHING CHANGED!!!")
            print(name)
        elif name not in para:
            para[name] = parameter


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params+=params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def no_train(model, param_names = []):
    if param_names == []:
        return model
    if type(param_names) == str():
        param_names = [param_names] 
    for name, parameter in model.named_parameters():
        if not name.startswith(tuple(param_names)):
            parameter.requires_grad = False
        # else:
        #     parameter.requires_grad = True
def do_train(cfg, model, val_name, resume=False):
    model.train()
    global notrain_params
    # no_train(model, [ "module." + n for n in notrain_params]) 
    # no_train(model, notrain_params)     
    # print(count_parameters(model))
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)

    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load(cfg.MODEL.WEIGHTS, resume=resume).get("iteration", -1) + 1
    )
    max_iter = cfg.SOLVER.MAX_ITER
    # max_iter = 0

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = default_writers(cfg.OUTPUT_DIR, max_iter) if comm.is_main_process() else []

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            # losses = loss_dict["loss_cls"]
            assert torch.isfinite(losses).all(), loss_dict
            
            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            # if iteration % 100 == 0:
            #     print("Iteration:\t", iteration, " \t Loss:\t",losses_reduced)
            optimizer.zero_grad()
            losses.backward()
            del loss_dict
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and (iteration + 1) % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter - 1
            ):
                do_test(cfg, model, val_name)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (
                (iteration + 1) % 20 == 0 or iteration == max_iter - 1
            ):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def do_test(cfg, model, val_name):
    t = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.DEVICE = "cuda:1"
    evaluator = COCOEvaluator(val_name, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, val_name)
    output = inference_on_dataset(model, val_loader, evaluator)
    f = open(cfg.OUTPUT_DIR+"results", "a")
    f.write(str(output))
    f.close()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = t


def main(args):
    global output_dir, notrain_params
    # train_name, val_name = register_coco(val_json = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/coco/annotations/instances_val2017.json",
    # train_json = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/coco/annotations/instances_train2017.json",
    # train_name="coco_2017_train1", val_name="coco_2017_val1" )
    train_name, val_name = register_cityscapes(train_json= "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_train_new.json",
                                        val_json = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_val_new.json")

    cfg = get_cfg()
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    file = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(file)
    cfg.MODEL.DEVICE = "cuda:1"
    cfg.merge_from_file("/misc/student/mirfan/config_files/coco_R_50_FPN.yaml")
    cfg.OUTPUT_DIR = "/misc/lmbraid19/mirfan/output/" + output_dir
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = (val_name,)
    # cfg.DATALOADER.NUM_WORKERS = 30
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    print("Counting parameters before freezing")
    # cfg.INPUT.MIN_SIZE_TEST = 768
    # cfg.INPUT.MAX_SIZE_TRAIN = 1536
    # print(count_parameters(model))
    # # model.backbone.bottom_up.res5.requires_grad_ = False
    # # model.proposal_generator.requires_grad_ = False
    # # model.roi_heads.box_predictor.requires_grad_ = False
    # no_train(model, notrain_params)
    # print("Counting parameters after freezing")
    # print(count_parameters(model))
    distributed = comm.get_world_size() > 1
    print("Distributed = ")
    print(distributed)
    print("")
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    print(cfg)
    # do_train(cfg, model, val_name, resume=True)
    do_test(cfg, model, val_name)
    global change
    print(change)

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
