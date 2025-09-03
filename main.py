# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     main.py
   Description :   main
   Author :       zhaowei
   date：          2025/9/3
-------------------------------------------------
   Change Activity:
                   2025/9/3:
-------------------------------------------------
"""
__author__ = 'zhaowei'

import argparse
from utils.datasets import get_dataset
import train_eval
from utils import tools
import time
from pprint import pprint
import random
from model.IAVF import IAVF
import os
import torch
from pathlib import Path

def get_parser():
    parser = argparse.ArgumentParser(description="Demo of argparse")
    parser.add_argument('--ds_name', default='PROTEINS',choices=["PROTEINS","DD","NCI1","NCI109","Mutagenicity","REDDIT-MULTI-12K"])
    parser.add_argument('--gpu_id', default='0')
    return parser

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    ds_name = args.ds_name
    gpu_id = args.gpu_id
    base_dir = Path(__file__).resolve().parent
    base_dir = base_dir.as_posix() 
    common_config_path = os.path.join(base_dir, "config", "IAVFCommon.json")
    config = tools.load_json(common_config_path)
    ds_config_loc = f"config/{ds_name}.json"
    if not os.path.exists(ds_config_loc):
        ds_config = {}
    else:
        ds_config = tools.load_json(f"config/{ds_name}.json")
    # updata config
    net_config = config["net_params"]
    train_config = config["train_config"]
    for key,value in ds_config.items():
        if key in net_config:
            net_config[key] = value
        else:
            train_config[key] = value
    if torch.cuda.is_available():
        device = f"cuda:{gpu_id}"
    else:
        device = "cpu"
    seed = 8971
    tools.set_seed(seed)
    dataset = get_dataset(ds_name, base_dir, config["data_dir"])
    num_feature, num_classes = dataset.num_features, dataset.num_classes
    net_config["device"] = device
    net_config["in_channels"] = num_feature
    net_config["out_channels"] = num_classes
    config["dataset"] = ds_name
    pprint(config)
    model = IAVF
    time_str = time.strftime('%Hh%Mm%Ss_on_%b_%d_%Y_') + str(random.randint(0, 100))
    acc, std, duration_mean = train_eval.cross_validation_with_acc_val_set(ds_name,"IAVF",dataset,seed,model,device,config,time_str,str(base_dir))
    print(f"test acc is {acc}, test std is {std}, duration mean is {duration_mean}")
