import json
import copy

file_coco = "/misc/lmbssd/saikiat/datasets/coco/annotations/instances_val2017.json"
file_cityscapes = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_train.json"
a = open(file_coco, "r")
coco = json.load(a)

# ann = []
# for c in coco["annotations"]:
#     if c["category_id"] in [1,2,3,4,6,7,8]:
#         ann.append(c)
# coco["annotations"] = ann
# with open('/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/coco/annotations/instances_val2017.json', 'w') as f:
#     json.dump(coco, f)

a = open(file_cityscapes, "r")
cityscapes = json.load(a)
city_dict = {}
coco_dict = {}


for cat in coco["categories"]:
    coco_dict[cat["name"]] = cat["id"]
for cat in cityscapes["categories"]:
    city_dict[cat["id"]] = cat["name"]

cityscapes_new = copy.copy(cityscapes)
cityscapes_new ["annotations"] = []
for image in cityscapes["annotations"]:
    old_id = image["category_id"]  
    if city_dict[old_id] == "rider" or city_dict[old_id] == "caravan" or city_dict[old_id] == "trailer":
        continue
    new_id = coco_dict[city_dict[old_id]]
    image["category_id"] = new_id 
    cityscapes_new["annotations"].append(image)

cityscapes_new["categories"] = []
for name, id in coco_dict.items():
    cityscapes_new["categories"].append({"id" : id , "name" : name})

with open('/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_train_new.json', 'w') as f:
    json.dump(cityscapes_new, f)
