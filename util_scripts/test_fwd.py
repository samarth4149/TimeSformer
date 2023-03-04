# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import os
import sys
sys.path.append(os.path.abspath('.'))

"""Wrapper to train and test a video classification model."""
from timesformer.utils.misc import launch_job
from timesformer.utils.parser import load_config, parse_args

from tools.test_net import test
from tools.train_net import train

from timesformer.models.build import build_model
from timesformer.utils.checkpoint import load_test_checkpoint
import torch


def main():
    """
    Main function to spawn the train and test process.
    """
    args = parse_args()
    if args.num_shards > 1:
       args.output_dir = str(args.job_dir)
    args.num_shards = 1
    cfg = load_config(args)
    cfg.defrost()
    cfg.NUM_GPUS = 1
    cfg.MODEL.MODEL_NAME = 'vit_base_patch16_224'
    cfg.DATA.NUM_FRAMES = 16
    cfg.MODEL.PRETRAINED = True
    cfg.TIMESFORMER.PRETRAINED_MODEL = '/path/to/pretrained_model.pth'
    model = build_model(cfg)
    load_test_checkpoint(cfg, model)
    
    x = torch.ones(1, 3, 16, 224, 224).cuda()
    print('Model out :', model(x))
    

if __name__ == "__main__":
    main()
