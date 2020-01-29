import random
from collections import Counter

from dataset.coco_format_dc import COCOFormatDetectionSubset
from dataset.data_config import DataSubsetSplit, DataConfig
from dataset.data_configs_dict import data_conf_tooth_only
import numpy as np


def load_dataset(data_config_dict=data_conf_tooth_only):
    data_config = DataConfig(image_data_basedir=None)
    data_config.pop_from_dict(data_config_dict)
    datasets = {}

    for _split in data_config.train_splits + data_config.eval_splits:  # type: DataSubsetSplit
        _name = _split.nickname
        coco_d = COCOFormatDetectionSubset(_split.ann_path, image_data_basedir=data_config.image_data_basedir)
        datasets[_name] = coco_d.to_image_annos(box_size_type=None)
    return datasets


class Distribution:
    def __init__(self):
        self.slot = 50
        self.distribution = Counter()
        self.accumulation = 0

    def add_to_distribution(self, ls):
        ls_int_group = list(map(lambda a: int(a/self.slot), ls))
        self.accumulation += len(ls_int_group)
        self.distribution += Counter(ls_int_group)

    @staticmethod
    def normalise(dist, sumx):
        nd = Counter({a: b/sumx for a, b in dist.items()})
        return nd

    def kullback_leibler_divergence(self, dist):
        assert self.slot == dist.slot

        p = Counter() + self.distribution
        q = Counter() + dist.distribution

        x_all = p+q
        for k in x_all:
            p[k] += 1
            q[k] += 1

        p = self.normalise(dist=p, sumx=self.accumulation+len(x_all))
        q = self.normalise(dist=q, sumx=self.accumulation+len(x_all))

        kld_acc = 0
        for k in x_all:
            kld_acc += p[k] * np.log((p[k]) / (q[k]))
        print("the KL divergence is:", kld_acc)



class MockAugment:
    def __init__(self, train_or_test_short_edge_length, max_size, data, sample_size=1):
        if isinstance(train_or_test_short_edge_length, int):
            self.short_edge_length = (train_or_test_short_edge_length, train_or_test_short_edge_length)
        else:
            self.short_edge_length = train_or_test_short_edge_length
        self.max_size = max_size
        self.data = data
        self.sample_size = sample_size

    def sample_strategy_uniform(self, img_ann):
        h, w = img_ann.img['height'], img_ann.img['width']

        size = random.randint(
            self.short_edge_length[0], self.short_edge_length[1] + 1)
        scale = size * 1.0 / min(h, w)
        if h < w:
            new_h, new_w = size, scale * w
        else:
            new_h, new_w = scale * h, size

        if max(new_h, new_w) > self.max_size:
            scale = self.max_size * 1.0 / max(new_h, new_w)
        return scale

    def apply(self) -> Distribution:
        dataset = self.data
        bbox_distribution = Distribution()
        for img_ann in dataset:
            for _ in range(self.sample_size):
                s = self.sample_strategy_uniform(img_ann=img_ann)
                new_box_sizes = (s * np.array(img_ann.bbox_sizes)).tolist()
                bbox_distribution.add_to_distribution(new_box_sizes)
        return bbox_distribution

dss = load_dataset()

MAX_SIZE = 1500
TRAIN_SHORT_EDGE = (200, 300)
TEST_SHORT_EDGE = 800

train_augmentation = MockAugment(train_or_test_short_edge_length=TRAIN_SHORT_EDGE,
                                 max_size=MAX_SIZE, data=dss['decay_train'], sample_size=10)

test_augmentation = MockAugment(train_or_test_short_edge_length=TEST_SHORT_EDGE,
                                max_size=MAX_SIZE, data=dss['decay_eval'], sample_size=1)

train_box_dist = train_augmentation.apply()
test_box_dist = test_augmentation.apply()

train_box_dist.kullback_leibler_divergence(test_box_dist)

xx = 0

