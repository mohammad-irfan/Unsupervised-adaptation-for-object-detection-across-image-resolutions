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


output_dir = "cityscape0.75_4bs_cityscape_all_24kfinetuning_sum/"

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
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        # if len(features.shape) > 3:
        #     features = features.view(features.shape[0], features.shape[1], -1)
        labels = torch.tensor( np.array(([[i]*2 for i in range(features.shape[0]//2)])).flatten())
        features = up_sample(features)
        # features = torch.mean(features,dim=1)
        # features = head(features.view(features.shape[0], -1))
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
    cfg.DATALOADER.NUM_WORKERS = 16
    data_loader = build_detection_train_loader(cfg, 
        total_batch_size = cfg.SOLVER.IMS_PER_BATCH , is_source = True)
    train_name, val_name = register_cityscapes(train_json= "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_train_new.json",
                                            train_dir = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/",
                                            val_dir = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/",
                                            val_json = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_val_new.json",
                                            train_name= "cityscapes_fine_detection_train_notsource",
                                            val_name="cityscapes_fine_detection_val_notsource")
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATALOADER.NUM_WORKERS = 8
    target_data_loader = build_detection_train_loader(cfg, 
        total_batch_size = cfg.SOLVER.IMS_PER_BATCH_TARGET , is_source = False)
    
    head = []
    up_sample = []
    feature_length = [512,256,128,64,32,16]
    start = 3
    for i in range(start,5):
        head.append(nn.Sequential(
            nn.Conv2d(256, 1, kernel_size = 1), nn.ReLU(inplace = True), 
                    nn.Flatten(1, -1),
                    nn.Linear(feature_length[i]*feature_length[i+1], feature_length[i]*feature_length[i+1]),
                    nn.ReLU(inplace=True),
                    nn.Linear(feature_length[i]*feature_length[i+1], 256)
                ))
        up_sample.append(nn.Upsample(size=(feature_length[i], feature_length[i+1]), mode='bilinear', align_corners=True))

    consistency_criterion = SupConLoss()
    logger.info("Starting training from iteration {}".format(start_iter))
    loss_dict = None
    with EventStorage(start_iter) as storage:
        for data, target_data, iteration in zip(data_loader, target_data_loader, range(start_iter, max_iter)):
            storage.iter = iteration
            data.requires_grad = True
            target_data.requires_grad = True
            loss_dict = model(data)
            del data
            images = model.preprocess_image(target_data)
            pom = model.backbone(images.tensor)
            # loss_dict = {}
            loss_dict["consistency_loss"] = []
            del target_data
            for i in range(5 - start):
                head[i] = head[i].to(model.device)
                c_loss = consistency_criterion(pom["p"+str(i+2+start)], head[i], up_sample[i])
                loss_dict["consistency_loss"].append(abs(c_loss))
                # head[i] = head[i].to("cpu")
            # print(loss_dict["consistency_loss"])
            loss_dict["consistency_loss"] = torch.sum(torch.stack(loss_dict["consistency_loss"]))
            losses = sum(loss_dict.values())
            # losses += consistency_loss(model, target_data)
            # losses = loss_dict["loss_cls"]
            # assert torch.isfinite(losses).all(), loss_dict
            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            if iteration % 500 == 0 and comm.get_local_rank() == 0:
                print("Iteration:\t", iteration, " \t Loss:\t",loss_dict)
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
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
            del loss_dict, c_loss, pom, images
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
    from detectron2.checkpoint import DetectionCheckpointer
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(file)
    cfg.MODEL.DEVICE = "cuda"
    cfg.set_new_allowed(True) 
    
    cfg.merge_from_file("/misc/student/mirfan/config_files/cityscapes.75_cityscapes_R_50_FPN.yaml")
    default_setup(cfg, args)
    cfg.OUTPUT_DIR = "/misc/lmbraid19/mirfan/output/" + output_dir
    cfg.DATASETS.TRAIN = (train_name,)
    cfg.DATASETS.TEST = ("cityscapes_fine_detection_val_notsource",)
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    model = build_model(cfg)
    DetectionCheckpointer(model).load("/misc/lmbraid19/mirfan/output/cityscape0.75_allfinetuned_bs8_24kiter/model_final.pth")  # load a file, usually from cfg.MODEL.WEIGHTS
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
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
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
