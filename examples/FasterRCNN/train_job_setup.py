#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

import argparse

from data_prepare.coco_format import COCOFormatDataLoader
from tensorpack import *
from tensorpack.tfutils import collect_env_info
from tensorpack.utils import logger
from tensorpack.tfutils.common import get_tf_version_tuple
import os

from dataset import register_coco, register_balloon, register_coco_format
from config import config as cfg
from config import finalize_configs
from data import get_train_dataflow
from eval import EvalCallback
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel

import multiprocessing as mp

try:
    import horovod.tensorflow as hvd
except ImportError:
    pass

"""
python3 train.py --config DATA.BASEDIR=~/dentalpoc/data/balloon MODE_FPN=True \
	"DATA.VAL=('balloon_val',)"  "DATA.TRAIN=('balloon_train',)" \
	TRAIN.BASE_LR=1e-3 TRAIN.EVAL_PERIOD=0 "TRAIN.LR_SCHEDULE=[1000]" \
	"PREPROC.TRAIN_SHORT_EDGE_SIZE=[600,1200]" TRAIN.CHECKPOINT_PERIOD=1 DATA.NUM_WORKERS=1 \
	--load ../../../pretrained-models/COCO-MaskRCNN-R50FPN2x.npz --logdir ~/logs/balloon-test
"""


def set_config_v1():
    cfg.freeze(False)
    cfg.MODE_FPN = True
    cfg.DATA.BASEDIR = os.path.abspath('../../../data/toooth')
    cfg.DATA.VAL = ('coco_formatted_eval',)
    cfg.DATA.TRAIN = ('coco_formatted_train',)
    cfg.TRAIN.BASE_LR = 1e-3
    cfg.TRAIN.EVAL_PERIOD = 1
    cfg.TRAIN.LR_SCHEDULE = [1000]
    cfg.PREPROC.TRAIN_SHORT_EDGE_SIZE = [600, 1200]
    cfg.TRAIN.CHECKPOINT_PERIOD = 1
    cfg.DATA.NUM_WORKERS = 1
    cfg.TRAIN.CHECKPOINT_PERIOD = 1
    cfg.freeze(True)


def add_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='Load a model to start training from. It overwrites BACKBONE.WEIGHTS')
    parser.add_argument('--logdir', help='Log directory. Will remove the old one if already exists.',
                        default='train_log/maskrcnn')
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py", nargs='+')

    args = parser.parse_args()
    return args


def arrange_multiprocess():
    # "spawn/forkserver" is safer than the default "fork" method and
    # produce more deterministic behavior & memory saving
    # However its limitation is you cannot pass a lambda function to subprocesses.
    mp.set_start_method('spawn')


def maybe_overwrite_config(train_args):
    # TODO: add interactive sanity check:
    #   - logdir is not empty yet still load from COCO pretrained
    #   - logdir is not empty should automatically pickup the training
    #   - Should automatically load category name
    # args.load = "~/logs/balloon-test/checkpoint"
    train_args.logdir = "/root/dentalpoc/logs/tootth3"
    config_yaml_path_copy_dump = os.path.join(train_args.logdir, 'train_config.yaml')
    if train_args.config:
        cfg.update_config_from_args(train_args.config)
    return train_args


def _setup_logging(logdir, is_horovod):
    # Setup logging ...

    if is_horovod:
        hvd.init()
    if not is_horovod or hvd.rank() == 0:
        logger.set_logger_dir(logdir, 'd')

    logger.info("Environment Information:\n" + collect_env_info())


def register_data_pipeline():
    # register_coco(cfg.DATA.BASEDIR)  # add COCO datasets to the registry
    # register_balloon(cfg.DATA.BASEDIR)  # add the demo balloon datasets to the registry
    annotations_dir = os.path.join(cfg.DATA.BASEDIR, 'annotations')
    data_load = COCOFormatDataLoader(project_root_dir='', coco_dir=annotations_dir)
    _latest = data_load.latest()
    latest_coco = next(_latest)
    register_coco_format(annotations_dir, data_meta_info=latest_coco)


def register_data_pipeline_v2(class_names=['decay']):
    # register_coco(cfg.DATA.BASEDIR)  # add COCO datasets to the registry
    # register_balloon(cfg.DATA.BASEDIR)  # add the demo balloon datasets to the registry
    annotations_dir = os.path.join(cfg.DATA.BASEDIR, 'annotations')
    data_load = COCOFormatDataLoader(project_root_dir='', coco_dir=annotations_dir)
    _latest = data_load.latest()
    latest_coco = next(_latest)
    register_coco_format(annotations_dir, data_meta_info=latest_coco)


def config_setup():
    # config_yaml_path = os.path.join(os.path.abspath(cfg.PROJECT_ROOT), 'train_config/default.yaml')
    # cfg.to_yaml(output_path=config_yaml_path)

    set_config_v1()

    arrange_multiprocess()

    train_args = add_args()
    train_args = maybe_overwrite_config(train_args)

    register_data_pipeline_v2()
    is_horovod_ = cfg.TRAINER == 'horovod'

    _setup_logging(train_args.logdir, is_horovod_)

    # TODO: what does freeze do?
    finalize_configs(is_training=True)

    return train_args, is_horovod_

