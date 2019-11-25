#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

import argparse

from data_prepare.coco_format import COCOFormatDataLoader
from tensorpack import *
from tensorpack.tfutils import collect_env_info
from tensorpack.tfutils.common import get_tf_version_tuple

from dataset import register_coco, register_balloon, register_coco_format
from config import config as cfg
from config import finalize_configs
from data import get_train_dataflow
from eval import EvalCallback
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel


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

import os
def set_config_A():
    cfg.DATA.BASEDIR = os.path.abspath('../../../data/toooth')
    cfg.MODE_FPN = True
    cfg.DATA.VAL = ('coco_formated_eval',)
    cfg.DATA.TRAIN = ('coco_formated_train',)
    cfg.TRAIN.BASE_LR = 1e-3
    cfg.TRAIN.EVAL_PERIOD = 1
    cfg.TRAIN.LR_SCHEDULE = [1000]
    cfg.PREPROC.TRAIN_SHORT_EDGE_SIZE = [600,1200]
    cfg.TRAIN.CHECKPOINT_PERIOD = 1
    cfg.DATA.NUM_WORKERS = 1
    cfg.TRAIN.CHECKPOINT_PERIOD = 1

if __name__ == '__main__':
    set_config_A()
    # "spawn/forkserver" is safer than the default "fork" method and
    # produce more deterministic behavior & memory saving
    # However its limitation is you cannot pass a lambda function to subprocesses.
    import multiprocessing as mp
    mp.set_start_method('spawn')
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='Load a model to start training from. It overwrites BACKBONE.WEIGHTS')
    parser.add_argument('--logdir', help='Log directory. Will remove the old one if already exists.',
                        default='train_log/maskrcnn')
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py", nargs='+')

    if get_tf_version_tuple() < (1, 6):
        # https://github.com/tensorflow/tensorflow/issues/14657
        logger.warn("TF<1.6 has a bug which may lead to crash in FasterRCNN if you're unlucky.")

    args = parser.parse_args()

    # args.load = "~/logs/balloon-test/checkpoint"
    args.logdir = "/root/dentalpoc/logs/tootth2"

    if args.config:
        cfg.update_args(args.config)

    # register_coco(cfg.DATA.BASEDIR)  # add COCO datasets to the registry
    # register_balloon(cfg.DATA.BASEDIR)  # add the demo balloon datasets to the registry
    import os
    annotations_dir = os.path.join(cfg.DATA.BASEDIR, 'annotations')
    data_load = COCOFormatDataLoader(project_root_dir='', coco_dir=annotations_dir)
    _latest = data_load.latest()
    latest_coco = next(_latest)
    register_coco_format(annotations_dir, splits_dic=latest_coco, class_names=['decay'])

    # Setup logging ...
    is_horovod = cfg.TRAINER == 'horovod'
    if is_horovod:
        hvd.init()
    if not is_horovod or hvd.rank() == 0:
        logger.set_logger_dir(args.logdir, 'd')
    logger.info("Environment Information:\n" + collect_env_info())

    finalize_configs(is_training=True)

    # Create model
    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

    # Compute the training schedule from the number of GPUs ...
    stepnum = cfg.TRAIN.STEPS_PER_EPOCH
    # warmup is step based, lr is epoch based
    init_lr = cfg.TRAIN.WARMUP_INIT_LR * min(8. / cfg.TRAIN.NUM_GPUS, 1.)
    warmup_schedule = [(0, init_lr), (cfg.TRAIN.WARMUP, cfg.TRAIN.BASE_LR)]
    warmup_end_epoch = cfg.TRAIN.WARMUP * 1. / stepnum
    lr_schedule = [(int(warmup_end_epoch + 0.5), cfg.TRAIN.BASE_LR)]

    factor = 8. / cfg.TRAIN.NUM_GPUS
    for idx, steps in enumerate(cfg.TRAIN.LR_SCHEDULE[:-1]):
        mult = 0.1 ** (idx + 1)
        lr_schedule.append(
            (steps * factor // stepnum, cfg.TRAIN.BASE_LR * mult))
    logger.info("Warm Up Schedule (steps, value): " + str(warmup_schedule))
    logger.info("LR Schedule (epochs, value): " + str(lr_schedule))
    train_dataflow = get_train_dataflow()
    # This is what's commonly referred to as "epochs"
    total_passes = cfg.TRAIN.LR_SCHEDULE[-1] * 8 / train_dataflow.size()
    logger.info("Total passes of the training set is: {:.5g}".format(total_passes))

    # Create callbacks ...
    callbacks = [
        PeriodicCallback(
            ModelSaver(max_to_keep=10, keep_checkpoint_every_n_hours=1),
            every_k_epochs=cfg.TRAIN.CHECKPOINT_PERIOD),
        # linear warmup
        ScheduledHyperParamSetter(
            'learning_rate', warmup_schedule, interp='linear', step_based=True),
        ScheduledHyperParamSetter('learning_rate', lr_schedule),
        GPUMemoryTracker(),
        HostMemoryTracker(),
        ThroughputTracker(samples_per_step=cfg.TRAIN.NUM_GPUS),
        EstimatedTimeLeft(median=True),
        SessionRunTimeout(60000),   # 1 minute timeout
        GPUUtilizationTracker()
    ]
    if cfg.TRAIN.EVAL_PERIOD > 0:
        callbacks.extend([
            EvalCallback(dataset, *MODEL.get_inference_tensor_names(), args.logdir)
            for dataset in cfg.DATA.VAL
        ])

    if is_horovod and hvd.rank() > 0:
        session_init = None
    else:
        if args.load:
            # ignore mismatched values, so you can `--load` a model for fine-tuning
            session_init = SmartInit(args.load, ignore_mismatch=True)
        else:
            session_init = SmartInit(cfg.BACKBONE.WEIGHTS)

    traincfg = TrainConfig(
        model=MODEL,
        data=QueueInput(train_dataflow),
        callbacks=callbacks,
        steps_per_epoch=stepnum,
        max_epoch=cfg.TRAIN.LR_SCHEDULE[-1] * factor // stepnum,
        session_init=session_init,
        starting_epoch=cfg.TRAIN.STARTING_EPOCH
    )

    if is_horovod:
        trainer = HorovodTrainer(average=False)
    else:
        # nccl mode appears faster than cpu mode
        trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, average=False, mode='nccl')
    launch_train_with_config(traincfg, trainer)
