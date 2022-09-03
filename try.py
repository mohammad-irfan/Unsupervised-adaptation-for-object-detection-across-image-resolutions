from re import L
from detectron.config import get_cfg
from detectron2 import model_zoo
import os
from register_datasets import *
import logging
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
from detectron.evaluation import COCOEvaluator, inference_on_dataset
import detectron.utils.comm as comm
from detectron.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron.engine import default_writers, default_setup, default_argument_parser, launch
from detectron.modeling import build_model
from detectron.solver import build_lr_scheduler, build_optimizer
from detectron.utils.events import EventStorage
import torch
output_dir = "cityscape0.50_cityscape24kiter_sum_norm_mean(1)simsiam/"
# output_dir = "reproduce_trial_p3_p5/"
# output_dir = "trialss/"

logger = logging.getLogger("detectron2")

def do_train(cfg, model, val_name, resume=False):
    model.train()
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
    # cfg.DATALOADER.NUM_WORKERS = 16
    data_loader = build_detection_train_loader(cfg, 
        total_batch_size = cfg.SOLVER.IMS_PER_BATCH , is_source = True)
    train_name, val_name = register_cityscapes(train_json= "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_train_new.json",
                                            train_dir = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/",
                                            val_dir = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/",
                                            val_json = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_val_new.json",
                                            train_name= "cityscapes_fine_detection_train_notsource",
                                            val_name="cityscapes_fine_detection_val_notsource")
    cfg.DATASETS.TRAIN = (train_name,)
    # cfg.DATALOADER.NUM_WORKERS = 16
    target_data_loader = build_detection_train_loader(cfg, 
        total_batch_size = cfg.SOLVER.IMS_PER_BATCH_TARGET , is_source = False)
    
    logger.info("Testing before starting training")
    # DetectionCheckpointer(model).load("/misc/lmbraid19/mirfan/output/25.8/cityscape0.50_allfinetuned_bs8_24kiter/model_final.pth")  # load a file, usually from cfg.MODEL.WEIGHTS
    DetectionCheckpointer(model).load("/misc/lmbraid19/mirfan/output/25.8/cityscape0.50_allfinetuned_bs8/model_final.pth")
    do_test(cfg, model, "cityscapes_fine_detection_val_notsource")

    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, target_data,  iteration in zip(data_loader, target_data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            optimizer.zero_grad()
            loss_dict = model(data, target_data)
            losses = sum(loss_dict.values())
            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            if iteration % 500 == 0 and comm.get_local_rank() == 0:
                print("Iteration:\t", iteration, " \t Loss:\t",loss_dict)
            losses.backward()
            optimizer.step()
            scheduler.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)

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
    cfg.MODEL.DEVICE = "cuda"
    evaluator = COCOEvaluator(val_name, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, val_name)
    output = inference_on_dataset(model, val_loader, evaluator)
    f = open(cfg.OUTPUT_DIR+"results", "a")
    f.write(str(output))
    f.close()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = t


def main(args):
    global output_dir
    train_name, val_name = register_cityscapes(train_json= "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes_.50/annotations/instancesonly_filtered_gtFine_train_new.json",
                                            train_dir = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes_.50/",
                                            val_dir = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes_.50/",
                                            val_json = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes_.50/annotations/instancesonly_filtered_gtFine_val_new.json",
                                            train_name= "cityscapes_fine_detection_train",
                                            val_name="cityscapes_fine_detection_val")

    cfg = get_cfg()
    cfg.merge_from_list(args.opts)
    file = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(file))
    from detectron2.checkpoint import DetectionCheckpointer
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(file)
    cfg.MODEL.DEVICE = "cuda"
    cfg.set_new_allowed(True) 
    
    cfg.merge_from_file("/misc/student/mirfan/config_files/cityscapes.50_cityscapes_R_50_FPN.yaml")
    default_setup(cfg, args)
    cfg.OUTPUT_DIR = "/misc/lmbraid19/mirfan/output/" + output_dir
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = ("cityscapes_fine_detection_val",)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    model = build_model(cfg)
    
    logger.info("Model:\n{}".format(model))
    print("Counting parameters before freezing")
    distributed = comm.get_world_size() > 1
    print("Distributed = ")
    print(distributed)
    print("")
    if args.num_gpus == 2:
        cfg.SOLVER.IMS_PER_BATCH_TARGET = cfg.SOLVER.IMS_PER_BATCH_TARGET//2
        cfg.SOLVER.IMS_PER_BATCH = cfg.SOLVER.IMS_PER_BATCH//2
    if args.num_gpus == 1:
        cfg.SOLVER.IMS_PER_BATCH_TARGET = cfg.SOLVER.IMS_PER_BATCH_TARGET//4
        cfg.SOLVER.IMS_PER_BATCH = cfg.SOLVER.IMS_PER_BATCH//4
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False,
            #  find_unused_parameters=True
        )

    print(cfg)
    do_train(cfg, model, val_name, resume=True)
    do_test(cfg, model, "cityscapes_fine_detection_val_notsource")

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
