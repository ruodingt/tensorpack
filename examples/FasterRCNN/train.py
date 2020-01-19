#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py

from config import config as cfg
from data import get_train_dataflow
from dataset.data_config import DataConfig
from dataset.data_configs import data_conf_tooth_only, data_conf_lesion_only
from eval import EvalCallback
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel
from tensorpack import *
from tensorpack.utils import logger
from train_job_setup import config_setup
import os

try:
    import horovod.tensorflow as hvd
except ImportError:
    pass


def setup_training_schedule(train_dataflow):
    # Compute the training schedule from the number of GPUs ...
    step_num_ = cfg.TRAIN.STEPS_PER_EPOCH
    # warmup is step based, lr is epoch based (huh...?)
    init_lr = cfg.TRAIN.WARMUP_INIT_LR * min(8. / cfg.TRAIN.NUM_GPUS, 1.)
    warmup_schedule_ = [(0, init_lr), (cfg.TRAIN.WARMUP, cfg.TRAIN.BASE_LR)]
    warmup_end_epoch_ = cfg.TRAIN.WARMUP * 1. / step_num_
    lr_schedule_ = [(int(warmup_end_epoch_ + 0.5), cfg.TRAIN.BASE_LR)]

    factor = 8. / cfg.TRAIN.NUM_GPUS
    for idx, steps in enumerate(cfg.TRAIN.LR_SCHEDULE[:-1]):
        mult = 0.1 ** (idx + 1)
        lr_schedule_.append(
            (steps * factor // step_num_, cfg.TRAIN.BASE_LR * mult))
    logger.info("Warm Up Schedule (steps, value): " + str(warmup_schedule_))
    logger.info("LR Schedule (epochs, value): " + str(lr_schedule_))

    # This is what's commonly referred to as "epochs"
    total_passes = cfg.TRAIN.LR_SCHEDULE[-1] * 8 / train_dataflow.size()
    logger.info("Total passes of the training set is: {:.5g}".format(total_passes))
    max_epoch_ = cfg.TRAIN.LR_SCHEDULE[-1] * factor // step_num_
    return warmup_schedule_, lr_schedule_, step_num_, max_epoch_


def session_initialization(is_horovod, load):
    if is_horovod and hvd.rank() > 0:
        session_init_ = None
    else:
        if load:
            print('load {}'.format(load))
            # ignore mismatched values, so you can `--load` a model for fine-tuning
            session_init_ = SmartInit(load, ignore_mismatch=True)
        else:
            session_init_ = SmartInit(cfg.BACKBONE.WEIGHTS)

    return session_init_


def create_callbacks(warmup_schedule, lr_schedule, model, logdir):
    # Create callbacks ...
    callbacks_ = [
        PeriodicCallback(
            ModelSaver(max_to_keep=10,
                       keep_checkpoint_every_n_hours=1),
            every_k_epochs=cfg.TRAIN.CHECKPOINT_PERIOD),
        # linear warmup
        ScheduledHyperParamSetter(
            'learning_rate', warmup_schedule, interp='linear', step_based=True),
        ScheduledHyperParamSetter('learning_rate', lr_schedule),
        GPUMemoryTracker(),
        HostMemoryTracker(),
        ThroughputTracker(samples_per_step=cfg.TRAIN.NUM_GPUS),
        EstimatedTimeLeft(median=True),
        SessionRunTimeout(60000),  # 60000 = 1 minute timeout
        GPUUtilizationTracker()
    ]

    if cfg.TRAIN.EVAL_PERIOD > 0:
        callbacks_.extend([
            EvalCallback(dataset, *model.get_inference_tensor_names(), logdir)
            for dataset in cfg.DATA.VAL + cfg.DATA.TRAIN
        ])
    return callbacks_


if __name__ == '__main__':

    data_config = DataConfig(image_data_basedir=None)
    data_config.pop_from_dict(data_conf_lesion_only)

    args, is_horovod = config_setup(data_config=data_config)

    # Create model
    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

    # get dataflow
    train_dataflow = get_train_dataflow()

    # setup training schedule
    warmup_schedule, lr_schedule, step_num, max_epoch = setup_training_schedule(train_dataflow)

    callbacks = create_callbacks(warmup_schedule, lr_schedule, model=MODEL, logdir=args.logdir)

    train_config = TrainConfig(
        model=MODEL,
        data=QueueInput(train_dataflow),
        callbacks=callbacks,
        steps_per_epoch=step_num,
        max_epoch=max_epoch,
        session_init=session_initialization(is_horovod=is_horovod, load=args.load),
        starting_epoch=cfg.TRAIN.STARTING_EPOCH
    )

    if is_horovod:
        trainer = HorovodTrainer(average=False)
    else:
        # nccl mode appears faster than cpu mode
        trainer = SyncMultiGPUTrainerReplicated(cfg.TRAIN.NUM_GPUS, average=False, mode='nccl')

    # exit()
    launch_train_with_config(train_config, trainer)
