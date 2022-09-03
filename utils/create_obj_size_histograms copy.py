import matplotlib.pyplot as plt
import json
import seaborn as sns
import pandas as pd

file_coco = "/misc/lmbssd/saikiat/datasets/coco/annotations/instances_val2017.json"
file_cityscapes = "/misc/lmbraid19/mirfan/cityscapes-to-coco-conversion/data/cityscapes/annotations/instancesonly_filtered_gtFine_val_new.json"
a = open(file_coco, "r")
coco = json.load(a)

a = open(file_cityscapes, "r")
cityscapes = json.load(a)
city_dict = {}

def plot_histogram(dataset, savefile, include = None):
    city_dict = {}
    for cat in dataset["categories"]:
        city_dict[cat["id"]] = cat["name"]

    d = {}
    d["Category"] = []
    d["Area"] = []
    for ann in dataset["annotations"]:
        category_name = city_dict[ann["category_id"]]
        if include is not None and category_name not in include:
            continue
        d["Category"].append(category_name)
        # try:
        #     d["Area"].append(int(len(ann["segmentation"]["counts"])/2))
        # except:
        #     d["Area"].append(int(len(ann["segmentation"][0])/2))
        d["Area"].append(ann["bbox"][2] * ann["bbox"][3])

    df2 = pd.DataFrame(d)
    g = sns.FacetGrid(df2, col="Category", col_wrap=4, sharey=False)
    g.map(sns.histplot, "Area", log_scale=True)
    # sns.histplot(data=df2, hue = "Category", y = "Area", multiple = "dodge",log_scale=True)
    plt.savefig(savefile)


def plot_histogram_together(cityscapes, coco, savefile, include = None):
    city_dict = {}
    coco_dict = {}
    for cat in cityscapes["categories"]:
        city_dict[cat["id"]] = cat["name"]
    for cat in coco["categories"]:
        coco_dict[cat["id"]] = cat["name"]

    d = {}
    d["Dataset"] = []
    d["Category"] = []
    d["Area"] = []
    def get_data(data, dic, name):
        for ann in data["annotations"]:
            category_name = dic[ann["category_id"]]
            if include is not None and category_name not in include:
                continue
            d["Category"].append(category_name)
            try:
                d["Area"].append(int(len(ann["segmentation"]["counts"])/2))
            except:
                d["Area"].append(int(len(ann["segmentation"][0])/2))
            # d["Area"].append(ann["bbox"][2] * ann["bbox"][3])
            d["Dataset"] .append(name)
    
    get_data(cityscapes, city_dict, "Cityscapes")
    get_data(coco, coco_dict, "Coco")


    df2 = pd.DataFrame(d)
    g = sns.FacetGrid(df2, col="Category", col_wrap=4, sharey=False, sharex= False, hue = "Dataset", xlim=(0,1000))
    g.map(sns.histplot, "Area" , log_scale=False)
    g.add_legend()
    # sns.histplot(data=df2, hue = "Category", y = "Area", multiple = "dodge",log_scale=True)
    plt.savefig(savefile)

# plot_histogram(cityscapes, savefile = "/misc/student/mirfan/histograms/Area/cityscapes_bbox2*3")
# , include = ['rider', 'car', 'person', 'bicycle', 'bus', 'motorcycle', 'truck', 'train'])

plot_histogram_together(cityscapes, coco, savefile = "/misc/student/mirfan/histograms/Area/together_len_seg", include = ['rider', 'car', 'person', 'bicycle', 'bus', 'motorcycle', 'truck', 'train'])
