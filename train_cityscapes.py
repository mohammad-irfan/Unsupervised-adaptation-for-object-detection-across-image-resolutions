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
from detectron.evaluation import COCOEvaluator, inference_on_dataset
import detectron.utils.comm as comm
from detectron.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron.config import get_cfg
from detectron.data import (
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron.engine import default_writers, default_setup, default_argument_parser, launch
from detectron.evaluation import (
    COCOEvaluator,
    inference_on_dataset,
)
from detectron.modeling import build_model
from detectron.solver import build_lr_scheduler, build_optimizer
from detectron.utils.events import EventStorage
from prettytable import PrettyTable
import os

logger = logging.getLogger("detectron2")

para = {}
change = []
notrain_params = []
output_dir = "cityscape0.75_allfinetuned_bs16_3.125e-4lr/"


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
    do_test(cfg, model, val_name)
    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg, total_batch_size = cfg.SOLVER.IMS_PER_BATCH, is_source = True)
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

            if iteration % 500 == 0:
                print("Iteration:\t", iteration, " \t Loss:\t",losses_reduced)
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
    cfg.MODEL.DEVICE = "cuda"
    evaluator = COCOEvaluator(val_name, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, val_name)
    output = inference_on_dataset(model, val_loader, evaluator)
    f = open(cfg.OUTPUT_DIR+"results", "a")
    f.write(str(output))
    f.close()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = t


def main(args):
    global output_dir, notrain_params
    train_name, val_name = register_cityscapes(train_json= "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes_.75/annotations/instancesonly_filtered_gtFine_train_new.json",
                                            train_dir = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes_.75/",
                                            val_dir = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes_.75/",
                                            val_json = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes_.75/annotations/instancesonly_filtered_gtFine_val_new.json",
                                            train_name= "cityscapes_fine_detection_train",
                                            val_name="cityscapes_fine_detection_val")
    cfg = get_cfg()
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    file = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(file)
    cfg.MODEL.DEVICE = "cuda"
    cfg.merge_from_file("/misc/student/mirfan/config_files/cityscapes_0.75_R_50_FPN_withCocoBase.yaml")
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = "/misc/lmbraid19/mirfan/output/" + output_dir
    cfg.DATASETS.TRAIN = ("cityscapes_fine_detection_train",)
    cfg.DATASETS.TEST = ("cityscapes_fine_detection_val",)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    model = build_model(cfg)
    logger.info("Model:\n{}".format(model))
    distributed = comm.get_world_size() > 1
    print("Distributed = ")
    print(distributed)
    print("")
    os.environ[
        "TORCH_DISTRIBUTED_DEBUG"
    ] = "DETAIL"  # set to DETAIL for runtime logging.
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    print(cfg)
    do_train(cfg, model, val_name, resume=True)
    do_test(cfg, model, val_name)
    print("Testing on Full scale dataset now")
    train_name, val_name = register_cityscapes(train_json= "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_train_new.json",
                                        train_dir = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/",
                                        val_dir = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/",
                                        val_json = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_val_new.json",
                                        train_name= "cityscapes_fine_detection_trainfull",
                                        val_name="cityscapes_fine_detection_valfull")
    do_test(cfg, model, val_name)
if __name__ == "__main__":
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
