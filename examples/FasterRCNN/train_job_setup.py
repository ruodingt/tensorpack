#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

import argparse
import datetime
import multiprocessing as mp

from config import config as cfg
from config import finalize_configs
from dataset import register_coco_format
from dataset.data_config import DataConfig
from tensorpack.tfutils import collect_env_info
from tensorpack.utils import logger
from utils.logconfig_checker import LogConfigChecker

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


def set_config_v1(data_config: DataConfig):
    cfg.freeze(False)
    cfg.MODE_FPN = True
    cfg.DATA.IMAGE_DATA_BASEDIR = data_config.image_data_basedir

    # TODO: this one change to path later
    cfg.DATA.TRAIN = data_config.get_nickname_list(DataConfig.TRAIN)
    cfg.DATA.VAL = data_config.get_nickname_list(DataConfig.EVAL)
    cfg.TRAIN.BASE_LR = 1e-2
    cfg.TRAIN.EVAL_PERIOD = 1
    cfg.TRAIN.LR_SCHEDULE = [1000, 2000]
    cfg.TRAIN.STEPS_PER_EPOCH = 500
    cfg.PREPROC.TRAIN_SHORT_EDGE_SIZE = [300, 800]  ## [400, 700] ,[300, 400]  # [300, 500]
    cfg.PREPROC.TEST_SHORT_EDGE_SIZE = 800  # 1000
    cfg.PREPROC.MAX_SIZE = 2000  # 1100, 1216
    cfg.TRAIN.CHECKPOINT_PERIOD = 1
    cfg.DATA.NUM_WORKERS = 1
    cfg.TRAIN.CHECKPOINT_PERIOD = 1

    cfg.BACKBONE.FREEZE_AT = 3  # 3
    cfg.freeze(True)


class TimeStamp:
    PATTERN = "@ %Y-%m-%d %H.%M.%S UTC"

    @classmethod
    def stamp(cls):
        x = datetime.datetime.now()
        return x.strftime(cls.PATTERN)

    @classmethod
    def stamp_to_datetime(cls, stamp: str):
        _dt = datetime.datetime.strptime(stamp, cls.PATTERN)
        return _dt

    def latest_item(self, ls: [str]):
        """
        TODO: given a list of items with time stamps, identify the latest item
        :param ls:
        :return:
        """
        raise NotImplementedError("Fill me in!")


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


def maybe_overwrite_config(train_args, log_subdir=None, log_root="/root/dentalpoc/logs"):
    """
    Handle following scenarios interactively:
       - logdir is not empty yet still want load from COCO pre-trained: create new dir
       - logdir is not empty: pickup the training
       - logdir is not exist yet: create new one and train from args.load
    :param train_args:
    :param log_subdir:
    :param log_root:
    :return:
    """

    lcc = LogConfigChecker(load_weight=train_args.load, default_log_subdir=log_subdir, log_root=log_root)
    logdir, load_weight = lcc.execute()

    train_args.logdir = logdir
    train_args.load = load_weight

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


# def register_data_pipeline_v2(data_config: DataConfig):
#     # assert 'train' in data_split_meta_info
#     # assert 'eval' in data_split_meta_info
#     register_coco_format(data_config=data_config)


def config_setup(data_config: DataConfig):
    # config_yaml_path = os.path.join(os.path.abspath(cfg.PROJECT_ROOT), 'train_config/default.yaml')
    # cfg.to_yaml(output_path=config_yaml_path)

    if data_config is None:
        data_config = DataConfig(image_data_basedir=None)
        data_config.pop_with_default()

    set_config_v1(data_config=data_config)

    arrange_multiprocess()

    train_args = add_args()
    train_args = maybe_overwrite_config(train_args)

    register_coco_format(data_config=data_config)
    is_horovod_ = cfg.TRAINER == 'horovod'

    _setup_logging(train_args.logdir, is_horovod_)

    # TODO: what does freeze do?
    finalize_configs(is_training=True)

    return train_args, is_horovod_


#  https://github.com/tensorflow/tensorflow/pull/32567/files

