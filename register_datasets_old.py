from detectron2.data.datasets import register_coco_instances

def register_coco(train_name = "coco_2017_train", 
                    train_json = "/misc/lmbssd/saikiat/datasets/coco/annotations/instances_train2017.json", 
                    train_dir = "/misc/lmbssd/saikiat/datasets/coco/train2017", 
                    val_name = "coco_2017_val",
                    val_json =  "/misc/lmbssd/saikiat/datasets/coco/annotations/instances_val2017.json",
                    val_dir =  "/misc/lmbssd/saikiat/datasets/coco/val2017"):

    register_coco_instances(train_name, {}, train_json, train_dir)
    register_coco_instances(val_name, {}, val_json, val_dir)
    
    return train_name, val_name


def register_cityscapes(train_name = "cityscapes_fine_detection_train", 
                    train_json = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json", 
                    train_dir = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/", 
                    val_name = "cityscapes_fine_detection_val",
                    val_json =  "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_val.json",
                    val_dir =  "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/"):

    register_coco_instances(train_name, {}, train_json, train_dir)
    register_coco_instances(val_name, {}, val_json, val_dir)
    
    return train_name, val_name
