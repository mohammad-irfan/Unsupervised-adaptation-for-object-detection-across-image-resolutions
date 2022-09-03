import datetime
import logging
import time
from collections import abc
from contextlib import ExitStack    
from typing import List, Union
import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from detectron2.utils.comm import get_world_size, is_main_process
from detectron2.utils.logger import log_every_n_seconds
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2 import model_zoo
import os
from register_datasets import *
import torch
import numpy as np
import os
import logging
import os
from detectron2.layers import FrozenBatchNorm2d
import torch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluator, DatasetEvaluators
from detectron2.data import build_detection_test_loader
import detectron2.utils.comm as comm
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_setup, default_argument_parser, launch
from detectron2.evaluation import (
    COCOEvaluator
)
from detectron2.utils.events import EventStorage
from math import sqrt

logger = logging.getLogger("detectron2")


total = 0
old_mean = []
new_mean = []
old_var = []
new_var = []
def no_grad(module):
    module.requires_grad_(False)
    try:
        for child in module.children():
            child.requires_grad_(False)
    except:
        print()


#create function to calculate Manhattan distance 
def manhattan(a, b):
    return sum(abs(val1-val2) for val1, val2 in zip(a,b))

def reset_stats(module):
    if isinstance(module, nn.BatchNorm2d):
        module.reset_running_stats()
        # Use exponential moving average
        module.momentum = None
    for p in module.parameters():
        p.requires_grad_(False)

class SyncBatch(nn.SyncBatchNorm):
    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        global total, old_mean, old_var
        bn_module = nn.modules.batchnorm
        bn_module = ( FrozenBatchNorm2d)
        res = module
        if isinstance(module, bn_module):
            old_mean.append(module.running_mean)
            old_var.append(module.running_var)
            res = cls(module.num_features)
            res.load_state_dict(module.state_dict(), strict = False)
            res.reset_running_stats()
            res.momentum = None
            del module
            total += 1
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        for p in res.parameters():
            p.requires_grad_(False)
        return res


class SyncBatch1(nn.SyncBatchNorm):
    @classmethod
    def convert_frozen_batchnorm(cls, module):
        """
        Convert all BatchNorm/SyncBatchNorm in module into FrozenBatchNorm.

        Args:
            module (torch.nn.Module):

        Returns:
            If module is BatchNorm/SyncBatchNorm, returns a new module.
            Otherwise, in-place convert module and return it.

        Similar to convert_sync_batchnorm in
        https://github.com/pytorch/pytorch/blob/master/torch/nn/modules/batchnorm.py
        """
        global total, new_mean, new_var
        bn_module = nn.modules.batchnorm
        bn_module = ( SyncBatch)
        res = module
        if isinstance(module, bn_module):
            new_mean.append(module.running_mean)
            new_var.append(module.running_var)
            res = cls(module.num_features)
            res.load_state_dict(module.state_dict(), strict = False)
            # res.reset_running_stats()
            res.momentum = None
            del module
            total += 1
        else:
            for name, child in module.named_children():
                new_child = cls.convert_frozen_batchnorm(child)
                if new_child is not child:
                    res.add_module(name, new_child)
        for p in res.parameters():
            p.requires_grad_(False)
        return res

def inference_on_dataset(
    model, data_loader, evaluator: Union[DatasetEvaluator, List[DatasetEvaluator], None]
):
    """
    Run model on the data_loader and evaluate the metrics with evaluator.
    Also benchmark the inference speed of `model.__call__` accurately.
    The model will be used in eval mode.

    Args:
        model (callable): a callable which takes an object from
            `data_loader` and returns some outputs.

            If it's an nn.Module, it will be temporarily set to `eval` mode.
            If you wish to evaluate a model in `training` mode instead, you can
            wrap the given model and override its behavior of `.eval()` and `.train()`.
        data_loader: an iterable object with a length.
            The elements it generates will be the inputs to the model.
        evaluator: the evaluator(s) to run. Use `None` if you only want to benchmark,
            but don't want to do any evaluation.

    Returns:
        The return value of `evaluator.evaluate()`
    """
    model.eval()
    num_devices = get_world_size()
    logger = logging.getLogger(__name__)
    logger.info("Start inference on {} batches".format(len(data_loader)))

    total = len(data_loader)  # inference data loader must have a fixed length
    if evaluator is None:
        # create a no-op evaluator
        evaluator = DatasetEvaluators([])
    if isinstance(evaluator, abc.MutableSequence):
        evaluator = DatasetEvaluators(evaluator)
    evaluator.reset()

    num_warmup = min(5, total - 1)
    start_time = time.perf_counter()
    total_data_time = 0
    total_compute_time = 0
    total_eval_time = 0
    with ExitStack() as stack:
        stack.enter_context(torch.no_grad())
        start_data_time = time.perf_counter()
        for idx, inputs in enumerate(data_loader):
            # if idx == 100:
            #     break
            total_data_time += time.perf_counter() - start_data_time
            if idx == num_warmup:
                start_time = time.perf_counter()
                total_data_time = 0
                total_compute_time = 0
                total_eval_time = 0

            start_compute_time = time.perf_counter()
            outputs = model(inputs)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            total_compute_time += time.perf_counter() - start_compute_time

            start_eval_time = time.perf_counter()
            evaluator.process(inputs, outputs)
            total_eval_time += time.perf_counter() - start_eval_time

            iters_after_start = idx + 1 - num_warmup * int(idx >= num_warmup)
            data_seconds_per_iter = total_data_time / iters_after_start
            compute_seconds_per_iter = total_compute_time / iters_after_start
            eval_seconds_per_iter = total_eval_time / iters_after_start
            total_seconds_per_iter = (time.perf_counter() - start_time) / iters_after_start
            if idx >= num_warmup * 2 or compute_seconds_per_iter > 5:
                eta = datetime.timedelta(seconds=int(total_seconds_per_iter * (total - idx - 1)))
                log_every_n_seconds(
                    logging.INFO,
                    (
                        f"Inference done {idx + 1}/{total}. "
                        f"Dataloading: {data_seconds_per_iter:.4f} s/iter. "
                        f"Inference: {compute_seconds_per_iter:.4f} s/iter. "
                        f"Eval: {eval_seconds_per_iter:.4f} s/iter. "
                        f"Total: {total_seconds_per_iter:.4f} s/iter. "
                        f"ETA={eta}"
                    ),
                    n=5,
                )
            start_data_time = time.perf_counter()

    # Measure the time only for this worker (before the synchronization barrier)
    total_time = time.perf_counter() - start_time
    total_time_str = str(datetime.timedelta(seconds=total_time))
    # NOTE this format is parsed by grep
    logger.info(
        "Total inference time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_time_str, total_time / (total - num_warmup), num_devices
        )
    )
    total_compute_time_str = str(datetime.timedelta(seconds=int(total_compute_time)))
    logger.info(
        "Total inference pure compute time: {} ({:.6f} s / iter per device, on {} devices)".format(
            total_compute_time_str, total_compute_time / (total - num_warmup), num_devices
        )
    )

    results = evaluator.evaluate()
    # An evaluator may return None when not in main process.
    # Replace it by an empty dict instead to make it easier for downstream code to handle
    if results is None:
        results = {}
    return results

def do_train(cfg, model, val_name, resume=False):

    max_iter = 8

    # a = [a for a  in model.children()]
    # if comm.get_world_size() > 1:
    #     a = [_ for _  in a[0].children()]
    # c = [c for c in  a[0].children()]
    # c = [c for c in  c[-1].children()]
    # c = [c for c in  c[1].children()]
    # c = [c for c in  c[0].children()]
    # c = [c for c in  c[0].children()]
    # if comm.get_local_rank() == 0:
    #     print("Printing data before training")
    #     print(c[0].num_batches_tracked)
    #     print(c[0].running_mean.data)
    model.train()
    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement in a small training loop
    data_loader = build_detection_train_loader(cfg)
    with EventStorage() as storage:
        # storage.enter_context(torch.no_grad())
        with torch.no_grad():    
            for data, iteration in zip(data_loader, range(0, max_iter)):
                storage.iter = iteration
                loss_dict = model(data)
                if comm.get_local_rank() == 0 and iteration % 25 == 0:
                    print("Iteration ", iteration, " done")



def do_test(cfg, model, val_name):
    # a = [a for a  in model.children()]
    # if comm.get_world_size() > 1:
    #     a = [_ for _  in a[0].children()]
    # c = [c for c in  a[0].children()]
    # c = [c for c in  c[-1].children()]
    # c = [c for c in  c[1].children()]
    # c = [c for c in  c[0].children()]
    # c = [c for c in  c[0].children()]
    # if comm.get_local_rank() == 0:
    #     print("Printing data after training")
    #     print(c[0].num_batches_tracked)
    #     print(c[0].running_mean.data)
    t = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
    cfg.MODEL.DEVICE = "cuda"
    evaluator = COCOEvaluator(val_name, output_dir=cfg.OUTPUT_DIR)
    val_loader = build_detection_test_loader(cfg, val_name)
    model.eval()
    output = inference_on_dataset(model, val_loader, evaluator)
    # print("Printing data after test")
    # print(c[0].num_batches_tracked)
    # print(c[0].running_mean.data)
    f = open(cfg.OUTPUT_DIR+"results", "a")
    f.write(str(output))
    f.close()
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = t


def main(args):

    __, val_name = register_cityscapes(train_json= "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_val_new.json",
                                            val_json = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_val_new.json")
    cfg = get_cfg()
    cfg.merge_from_list(args.opts)
    default_setup(cfg, args)
    file = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(file)
    cfg.MODEL.DEVICE = "cuda"
    # cfg.merge_from_file("/misc/student/mirfan/config_files/cityscapes_R_50_FPN.yaml")
    cfg.MODEL.DEVICE = "cuda"
    cfg.OUTPUT_DIR = "/misc/lmbraid19/mirfan/output/cityscapes_7_coco_cityscapes_adptbn/"
    cfg.DATASETS.TRAIN = ("cityscapes_fine_detection_train",)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    from detectron2.engine import DefaultPredictor
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.INPUT.MIN_SIZE_TEST =  1024
    # cfg.INPUT.MAX_SIZE_TEST =  0
    cfg.INPUT.MIN_SIZE_TRAIN = 0
    cfg.INPUT.MAX_SIZE_TRAIN = 0
    predictor = DefaultPredictor(cfg)
    model = predictor.model 
    model = model.cuda()
    logger.info("Model:\n{}".format(model))

    distributed = comm.get_world_size() > 1
    # print("Distributed = ")
    # print(distributed)
    # print("")
    # model = SyncBatch.convert_frozen_batchnorm(model)
    # a = [a for a  in model.children()]
    # a = [_ for _  in a[-1].children()]
    # for p in a[-1].parameters():
    #     p.requires_grad_(True)
    #     break
    # model.train()
    # a = [a for a  in model.children()]
    # a = [_ for _  in a[0].children()]
    # a[-1].eval()
    # model.backbone._out_features = ["res2", "res3", "res4"]
    # model.backbone.stem = SyncBatch.convert_frozen_batchnorm(model.backbone.stem)
    # model.backbone.res2 = SyncBatch.convert_frozen_batchnorm(model.backbone.res2)
    # model.backbone.res3 = SyncBatch.convert_frozen_batchnorm(model.backbone.res4)
    # model.backbone.res4 = SyncBatch.convert_frozen_batchnorm(model.backbone.res4)
    model = model.cuda()

    # a = [a for a  in model.children()]
    # a = [_ for _  in a[-1].children()]
    # for p in a[-1].parameters():
    #     p.requires_grad_(True)
    #     break
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )
    

    # print(cfg)
    do_train(cfg, model, val_name, resume=True)
    do_test(cfg, model, val_name)
    model1 = SyncBatch1.convert_frozen_batchnorm(model)
    var = [float(manhattan(new_var[i], old_var[i])/len(new_var[i])) for i in range(53)]
    mean = [float(manhattan(new_mean[i], old_mean[i])/len(new_mean[i])) for i in range(53)]
    1

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    # args.num_gpus = 1
    # print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
