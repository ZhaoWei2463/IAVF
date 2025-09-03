# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     LoadModel
   Description :
   Author :       zhaowei
   date：          2025/9/3
-------------------------------------------------
   Change Activity:
                   2025/9/3:
-------------------------------------------------
"""
__author__ = 'zhaowei'
from model.IAVF import IAVF
import torch
import os
from utils import tools
from utils.datasets import get_dataset
from train_eval import k_fold,evaluate_network
from torch_geometric.data import DataLoader
import numpy as np
from pprint import pprint
#  NCI1,NCI109,Mutagenicity,REDDIT-MULTI-12K
ds_name = "PROTEINS"
gpu_id = 0

ds_config = tools.load_json(f"config/{ds_name}.json")
config = tools.load_json("config/IAVFCommon.json")
# updata config
net_config = config["net_params"]
train_config = config["train_config"]
for key, value in ds_config.items():
    if key in net_config:
        net_config[key] = value
    else:
        train_config[key] = value
if torch.cuda.is_available():
    device = f"cuda:{gpu_id}"
    map_location=f'cuda:{gpu_id}'
else:
    device = "cpu"
    map_location = "cpu"


seed = 8971
tools.set_seed(seed)
dataset = get_dataset(ds_name, config["data_dir"])
num_feature, num_classes = dataset.num_features, dataset.num_classes
net_config["device"] = device
net_config["in_channels"] = num_feature
net_config["out_channels"] = num_classes
pprint(net_config)
pprint(train_config)
model = IAVF(**net_config)
model.to(device)
print(model)
all_accs = []

for fold, (train_idx, test_idx,
           val_idx) in enumerate(zip(*k_fold(dataset, 10, seed))):

    train_dataset = dataset[train_idx]
    test_dataset = dataset[test_idx]
    val_dataset = dataset[val_idx]
    train_loader = DataLoader(train_dataset, train_config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, train_config["batch_size"], shuffle=False)
    test_loader = DataLoader(test_dataset, train_config["batch_size"], shuffle=False)

    time_str = "15h26m40s_on_Sep_03_2025_84"  #Manually replace <time_str> with the timestamp folder name generated during training.

    model.load_state_dict(torch.load(os.path.join("out","result",ds_name,time_str,"models",
                                                  f"{fold+1}.pth"), map_location))
    _,acc = evaluate_network(model,device,test_loader)
    all_accs.append(acc)
    print(f"fold {fold + 1}, test acc is {acc}")
acc_mean = np.mean(np.array(all_accs))
acc_std = np.std(np.array(all_accs))
print("Test Accuracy: {:.4f} ± {:.4f}".format(acc_mean * 100 ,acc_std * 100))


