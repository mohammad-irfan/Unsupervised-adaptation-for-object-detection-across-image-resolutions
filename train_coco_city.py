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
from detectron.modeling import build_model
from detectron.solver import build_lr_scheduler, build_optimizer
from detectron.utils.events import EventStorage


output_dir = "cityscape_coco_all_mean_conv2d/"

logger = logging.getLogger("detectron2")

from prettytable import PrettyTable
import torch.nn.functional as F
import torch.nn as nn
class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07, contrast_mode='all',
                 base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, head, up_sample, labels=None, mask=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        labels = torch.tensor( np.array(([[i]*2 for i in range(features.shape[0]//2)])).flatten())
        features = up_sample(features)
        features = head(features)
        features = features.view(int(features.shape[0]/2), 2, -1)
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # anchor_dot_contrast = torch.matmul(anchor_feature, contrast_feature.T)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        # logits = anchor_dot_contrast
        # tile mask
        # mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss =  1e7 * torch.abs(mean_log_prob_pos)
        loss = loss.view(anchor_count, batch_size).mean()
        del features, anchor_dot_contrast
        return loss



def consistency_loss(model, target_data):
    images = model.module.preprocess_image(target_data)
    pom = model.module.backbone(images.tensor)
    # labels = torch.cat([torch.arange(len(target_data)) for i in range(256)], dim=0)
    features = torch.mean(pom["p6"].reshape(len(target_data), -1), 0)
    labels = torch.cat([torch.arange(1) for i in range(len(target_data))], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    # features = F.normalize(pom["p6"], dim=1)
    similarity_matrix = torch.matmul(features.view(len(target_data),-1), features.view(len(target_data),-1).T)
    mask = torch.eye(labels.shape[0], dtype=torch.bool)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    # select only the negatives the negatives
    negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

    logits = torch.cat([positives, negatives], dim=1)
    labels = torch.zeros(logits.shape[0], dtype=torch.long)
    # labels = torch.tensor([0,1])
    loss = torch.nn.CrossEntropyLoss()(logits,labels.to(model.module.device) )
    del logits, labels, positives, negatives, 
    return  loss

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
    cfg.DATALOADER.NUM_WORKERS = 24
    data_loader = build_detection_train_loader(cfg, 
        total_batch_size = cfg.SOLVER.IMS_PER_BATCH , is_source = True)
    train_name, val_name = register_cityscapes(train_json= "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_train_new.json",
                                            val_json = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_val_new.json")
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATALOADER.NUM_WORKERS = 12
    target_data_loader = build_detection_train_loader(cfg, 
        total_batch_size = cfg.SOLVER.IMS_PER_BATCH_TARGET , is_source = False)
    logger.info("Starting training from iteration {}".format(start_iter))
    # do_test(cfg, model, "cityscapes_fine_detection_val")
    consistency_criterion = SupConLoss()
    with EventStorage(start_iter) as storage:
        for data,target_data, iteration in zip(data_loader, target_data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            loss_dict = model(data, target_data)
            losses = sum(loss_dict.values())
            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            if iteration % 500 == 0 and comm.get_local_rank() == 0:
                print("Iteration:\t", iteration, " \t Loss:\t",losses_reduced)
            optimizer.zero_grad()
            losses.backward()
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
            del loss_dict, target_data, data
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
    train_name, val_name = register_coco(val_json = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/coco/annotations/instances_val2017.json",
    train_json = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/coco/annotations/instances_train2017.json",
    train_name="coco_2017_train1", val_name="coco_2017_val1" )
    cfg = get_cfg()
    cfg.merge_from_list(args.opts)
    file = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
    cfg.merge_from_file(model_zoo.get_config_file(file))
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(file)
    cfg.MODEL.DEVICE = "cuda"
    cfg.set_new_allowed(True) 
    cfg.merge_from_file("/misc/student/mirfan/config_files/coco_cityscapes_R_50_FPN.yaml")
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
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    print(cfg)
    do_train(cfg, model, val_name, resume=True)
    do_test(cfg, model, "cityscapes_fine_detection_val")

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