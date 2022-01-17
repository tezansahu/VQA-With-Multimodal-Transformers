import os
import yaml
import argparse
import json
from typing import Text

import torch

from load_data import loadDaquarDataset
from data_collator import createMultimodalDataCollator
from model import createMultimodalModelForVQA
from train import trainMultimodalModelForVQA
from evaluate import WuPalmerScoreCalculator

def main(config_path: Text):
    with open(config_path) as conf_file:
        config = yaml.safe_load(conf_file)
    
    if config["base"]["use_cuda"]:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        os.environ['CUDA_VISIBLE_DEVICES'] = '0' # SET ONLY 1 GPU DEVICE
    else:
        device =  torch.device('cpu')
    
    data = loadDaquarDataset(config)
    multimodal_collator = createMultimodalDataCollator(config)
    multimodal_model = createMultimodalModelForVQA(config, data["answer_space"]).to(device)

    training_metrics, eval_multi_metrics = trainMultimodalModelForVQA(
        config, device, data["dataset"], 
        multimodal_collator, multimodal_model,
        WuPalmerScoreCalculator.compute_metrics
    )

    
    os.makedirs(config["metrics"]["metrics_folder"], exist_ok=True)

    training_metrics_path = os.path.join(config["metrics"]["metrics_folder"], config["metrics"]["training_metrics_file"])
    json.dump(
        obj=training_metrics,
        fp=open(training_metrics_path, 'w'),
        indent=4
    )

    eval_metrics_path = os.path.join(config["metrics"]["metrics_folder"], config["metrics"]["eval_metrics_file"])
    json.dump(
        obj=eval_multi_metrics,
        fp=open(eval_metrics_path, 'w'),
        indent=4
    )




if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    
    main(args.config)