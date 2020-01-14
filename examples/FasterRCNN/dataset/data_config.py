import os
from collections import namedtuple

DataSubsetSplit = namedtuple("DataSubsetSplit", "nickname, ann_path")


class DataConfig:
    TRAIN = 'TRAIN'
    EVAL = 'EVAL'
    IMAGE_BASEDIR = 'IMAGE_BASEDIR'
    NICKNAME = 'nickname'
    ANN_PATH = 'ann_path'

    def __init__(self, image_data_basedir):
        self.train_splits = []
        self.eval_splits = []
        self.image_data_basedir = image_data_basedir

    def add_subset_spilt(self, subset_split: DataSubsetSplit, append_to_group):
        if append_to_group == self.TRAIN:
            self.train_splits.append(subset_split)
        elif append_to_group == self.EVAL:
            self.eval_splits.append(subset_split)
        else:
            raise Exception("append_to_group can only be TRAIN or EVAL")

    def get_nickname_list(self, sp_group):
        if sp_group == self.TRAIN:
            return tuple(map(lambda a: a.nickname, self.train_splits))
        elif sp_group == self.EVAL:
            return tuple(map(lambda a: a.nickname, self.eval_splits))
        else:
            raise Exception("sp_group can only be TRAIN or EVAL")

    def reset(self):
        self.train_splits = []
        self.eval_splits = []

    def pop_with_default(self):
        self.reset()
        self.image_data_basedir = os.path.abspath('../../../data/datasets_coco/')
        # train_ann_path = os.path.join(os.path.abspath('../../../data/'), 'coco_stack_out/web_decay_601.json')
        # eval_ann_path = os.path.join(os.path.abspath('../../../data/'), 'coco_stack_out/web_decay_601.json')
        self.add_subset_spilt(DataSubsetSplit(
            nickname='decay_train',
            ann_path=os.path.join(os.path.abspath('../../../data/'), 'coco_stack_out/web_decay_600-5.json')),
            append_to_group=self.TRAIN)

        self.add_subset_spilt(DataSubsetSplit(
            nickname='decay_eval',
            ann_path=os.path.join(os.path.abspath('../../../data/'), 'coco_stack_out/legacy_decay-3.json')),
            append_to_group=self.EVAL)

    def pop_from_dict(self, d):
        self.reset()
        self.image_data_basedir = d[self.IMAGE_BASEDIR]

        for t in d[self.TRAIN]:
            self.add_subset_spilt(DataSubsetSplit(
                nickname=t[self.NICKNAME],
                ann_path=t[self.ANN_PATH]),
                append_to_group=self.TRAIN)

        for t in d[self.EVAL]:
            self.add_subset_spilt(DataSubsetSplit(
                nickname=t[self.NICKNAME],
                ann_path=t[self.ANN_PATH]),
                append_to_group=self.EVAL)

