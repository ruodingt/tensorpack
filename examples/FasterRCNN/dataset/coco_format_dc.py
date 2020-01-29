# -*- coding: utf-8 -*-
import copy
import json
import os
import time
from collections import defaultdict, namedtuple

import numpy as np
import tqdm
from pycocotools.coco import COCO

from config import config as cfg
from dataset import DatasetRegistry, DatasetSplit
from dataset.data_config import DataConfig, DataSubsetSplit
from tensorpack.utils import logger
from tensorpack.utils.timer import timed_operation

__all__ = ['register_coco_format']


ReplaceItem = namedtuple("ReplaceItem", 'new_id, item')

class COCO2(COCO):
    def __init__(self, annotation_file=None, use_ext=False):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.
        :param annotation_file (str): location of annotation file
        :param image_folder (str): location to the folder that hosts images.
        :return:
        """
        self.use_ext = use_ext
        # load dataset
        self.dataset,self.anns,self.cats,self.imgs = dict(),dict(),dict(),dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            dataset = self.remove_empty_annotations(dataset)
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = self.replace_category_id(dataset)
            self.createIndex(use_ext)

    def replace_category_id(self, dataset):
        _categories_sorted = sorted(dataset['categories'], key=lambda k: k['id'], reverse=False)
        coco_new_id_2_original_id = dict(map(
            lambda x: (x[1]['id'], ReplaceItem(new_id=x[0]+1, item=x[1])), enumerate(_categories_sorted)))

        for c in dataset['categories']:
            c['old_id'] = c['id']
            c['id'] = coco_new_id_2_original_id[c['id']].new_id

        for a in dataset['annotations']:
            a['old_category_id'] = a['category_id']
            a['category_id'] = coco_new_id_2_original_id[a['category_id']].new_id
        return dataset

    def remove_empty_annotations(self, dataset):
        for i in range(len(dataset['annotations'])):
            if not dataset['annotations'][i]['segmentation']:
                dataset['annotations'].pop(i)
                print("drop annotation #{}".format(i))
        return dataset


class ImageAnno:
    def __init__(self, img, annos, size_type=None):
        self.img = img
        self.anno_ls = annos
        self.size_type = size_type

        self.bbox_sizes = self._extract_box_size()

    def _extract_box_size(self):
        if self.size_type is None:
            box_sizes_hw = [(a['bbox'][2], a['bbox'][3]) for a in self.anno_ls]
            box_sizes = list(map(lambda x: np.sqrt(x[0] ** 2 + x[1] ** 2), box_sizes_hw))
            return box_sizes
        else:
            raise NotImplementedError()


class COCOFormatDetectionSubset(DatasetSplit):
    # handle a few special splits whose names do not match the directory names
    # _INSTANCE_TO_BASEDIR = {
    #     'valminusminival2014': 'val2014',
    #     'minival2014': 'val2014',
    #     'val2017_100': 'val2017',
    # }

    """
    Mapping from the incontinuous COCO category id to an id in [1, #category]
    For your own coco-format, dataset, change this to an **empty dict**.
    """

    # COCO_id_to_category_id = {13: 12, 14: 13, 15: 14, 16: 15, 17: 16, 18: 17, 19: 18, 20: 19, 21: 20, 22: 21, 23: 22, 24: 23, 25: 24, 27: 25, 28: 26, 31: 27, 32: 28, 33: 29, 34: 30, 35: 31, 36: 32, 37: 33, 38: 34, 39: 35, 40: 36, 41: 37, 42: 38, 43: 39, 44: 40, 46: 41, 47: 42, 48: 43, 49: 44, 50: 45, 51: 46, 52: 47, 53: 48, 54: 49, 55: 50, 56: 51, 57: 52, 58: 53, 59: 54, 60: 55, 61: 56, 62: 57, 63: 58, 64: 59, 65: 60, 67: 61, 70: 62, 72: 63, 73: 64, 74: 65, 75: 66, 76: 67, 77: 68, 78: 69, 79: 70, 80: 71, 81: 72, 82: 73, 84: 74, 85: 75, 86: 76, 87: 77, 88: 78, 89: 79, 90: 80}  # noqa

    def __init__(self, annotation_fp, image_data_basedir=None):
        print("load annotation file from: ", annotation_fp)
        """
        Args:
            image_data_basedir (str): root of the dataset which contains the subdirectories for each split and annotations
            annotation_fp (str): the name of the split, e.g. "train2017".
                The split has to match an annotation file in "annotations/" and a directory of images.

        Examples:
            For a directory of this structure:

            DIR/
              annotations/
                instances_XX.json
                instances_YY.json
              XX/
              YY/

            use `COCODetection(DIR, 'XX')` and `COCODetection(DIR, 'YY')`
        """

        self.base_dir = image_data_basedir
        assert os.path.isfile(annotation_fp), annotation_fp

        self.annotation_fp = annotation_fp

        self.coco = COCO2(annotation_fp)
        self.annotation_file_path = annotation_fp

        if self.base_dir is not None:
            for _id in self.coco.imgs:
                _p = os.path.join(self.base_dir, self.coco.imgs[_id]['path'][1:])
                self.coco.imgs[_id]['path'] = _p

        _categories = list(map(lambda x: (x['id'], x['name']), self.coco.cats.values()))
        _categories_sorted = sorted(_categories, key=lambda k: k[0], reverse=False)
        if len(_categories_sorted) > 1:
            assert _categories_sorted[0][0] == 1
        _, self.categories_name = list(zip(*_categories_sorted))
        logger.info("Instances loaded from {}.".format(annotation_fp))
        self.box_sizes = []

    def to_image_annos(self, box_size_type=None):
        ls = []
        for k, v in self.coco.imgToAnns.items():
            ls.append(ImageAnno(img=self.coco.imgs[k], annos=v, size_type=box_size_type))
        return ls

    def f_score(self,p,r,k):
        """
         recall is considered k times as important as precision
        :param p:
        :param r:
        :param k:
        :return:
        """
        return (1 + k**2)*p*r/(k**2*p + r)

    def print_coco_metrics(self, results):
        """
        # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
        Args:
            results(list[dict]): results in coco format
        Returns:
            dict: the evaluation metrics
        """
        from pycocotools.cocoeval import COCOeval
        ret = {}
        has_mask = "segmentation" in results[0]  # results will be modified by loadRes

        cocoDt = self.coco.loadRes(results)
        cocoEval = COCOeval(self.coco, cocoDt, 'bbox')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
        fields_ap = ['IoU=0.5:0.95', 'IoU=0.5', 'IoU=0.75', 'small', 'medium', 'large']
        # fields_ar = ['IoU=0.50:0.95', ]
        for k in range(6):
            ret['mAP(bbox)/' + fields_ap[k]] = cocoEval.stats[k]
        for k in range(6, 12):
            ret['mAR(bbox)/' + 'IoU=0.5:0.95'] = cocoEval.stats[k]

        ret['mf_score(bbox)/' + 'f1_AP[IoU=0.5]_vs_AR[IoU=0.5:0.95]'] = \
            self.f_score(p=cocoEval.stats[1], r=cocoEval.stats[7], k=1)
        ret['mf_score(bbox)/' + 'f2_AP[IoU=0.5]_vs_AR[IoU=0.5:0.95]'] = \
            self.f_score(p=cocoEval.stats[1], r=cocoEval.stats[7], k=2)

        if len(results) > 0 and has_mask:
            cocoEval = COCOeval(self.coco, cocoDt, 'segm')
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            for k in range(6):
                ret['mAP(segm)/' + fields_ap[k]] = cocoEval.stats[k]
            for k in range(6, 12):
                ret['mAR(segm)/' + 'IoU=0.5:0.95'] = cocoEval.stats[k]

        ret['mf_score(segm)/' + 'f1_AP[IoU=0.5]_vs_AR[IoU=0.5:0.95]'] = \
            self.f_score(p=cocoEval.stats[1], r=cocoEval.stats[7], k=1)
        ret['mf_score(segm)/' + 'f2_AP[IoU=0.5]_vs_AR[IoU=0.5:0.95]'] = \
            self.f_score(p=cocoEval.stats[1], r=cocoEval.stats[7], k=2)

        print(">>>>>>>>>>>>>>ret", ret)
        return ret

    def load(self, add_gt=True, add_mask=False):
        """
        Args:
            add_gt: whether to add ground truth bounding box annotations to the dicts
            add_mask: whether to also add ground truth mask

        Returns:
            a list of dict, each has keys including:
                'image_id', 'file_name',
                and (if add_gt is True) 'boxes', 'class', 'is_crowd', and optionally
                'segmentation'.
        """
        # self.reload_coco()
        with timed_operation('Load annotations for {}'.format(
                os.path.basename(self.annotation_file_path))):
            img_ids_r = self.coco.getImgIds()

            if add_gt:
                img_ids = list(filter(lambda x: x in self.coco.imgToAnns, img_ids_r))
            else:
                img_ids = img_ids_r

            img_ids.sort()
            # list of dict, each has keys: height,width,id,file_name
            imgs = copy.deepcopy(self.coco.loadImgs(img_ids))

            # print("drop images without annotations")
            # imgs = list(filter(lambda x: x in self.coco.imgToAnns, imgs_r))

            for idx, img in enumerate(tqdm.tqdm(imgs)):
                # img['image_id'] = img.pop('id')
                #  FIXME: why pop here, if use func to load COCOFormatDetectionSubset, then we can use pop
                #       when use func, it can prevent data
                img['image_id'] = img['id']
                img['file_name'] = img['path']
                if idx == 0:
                    # make sure the directories are correctly set
                    assert os.path.isfile(img["file_name"]), img["file_name"]
                if add_gt:
                    self._add_detection_gt(img, add_mask)
            return imgs

    def _add_detection_gt(self, img, add_mask):
        """
        Add 'boxes', 'class', 'is_crowd' of this image to the dict, used by detection.
        If add_mask is True, also add 'segmentation' in coco poly format.
        """
        # ann_ids = self.coco.getAnnIds(imgIds=img['image_id'])
        # objs = self.coco.loadAnns(ann_ids)
        objs = self.coco.imgToAnns[img['image_id']]  # equivalent but faster than the above two lines
        if 'minival' not in self.annotation_file_path:
            # TODO better to check across the entire json, rather than per-image
            ann_ids = [ann["id"] for ann in objs]
            assert len(set(ann_ids)) == len(ann_ids), \
                "Annotation ids in '{}' are not unique!".format(self.annotation_file_path)

        # clean-up boxes
        width = img.pop('width')
        height = img.pop('height')

        all_boxes = []
        all_segm = []
        all_cls = []
        all_iscrowd = []
        for objid, obj in enumerate(objs):
            if obj.get('ignore', 0) == 1:
                continue
            x1, y1, w, h = list(map(float, obj['bbox']))
            # bbox is originally in float
            # x1/y1 means upper-left corner and w/h means true w/h. This can be verified by segmentation pixels.
            # But we do make an assumption here that (0.0, 0.0) is upper-left corner of the first pixel
            x2, y2 = x1 + w, y1 + h

            # np.clip would be quite slow here
            x1 = min(max(x1, 0), width)
            x2 = min(max(x2, 0), width)
            y1 = min(max(y1, 0), height)
            y2 = min(max(y2, 0), height)
            w, h = x2 - x1, y2 - y1
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 1 and w > 0 and h > 0:
                all_boxes.append([x1, y1, x2, y2])
                all_cls.append(obj['category_id'])
                iscrowd = obj.get("iscrowd", 0)
                all_iscrowd.append(iscrowd)

                if add_mask:
                    segs = obj['segmentation']
                    if not isinstance(segs, list):
                        assert iscrowd == 1
                        all_segm.append(None)
                    else:
                        valid_segs = [np.asarray(p).reshape(-1, 2).astype('float32') for p in segs if len(p) >= 6]
                        if len(valid_segs) == 0:
                            logger.error("Object {} in image {} has no valid polygons!".format(objid, img['file_name']))
                        elif len(valid_segs) < len(segs):
                            logger.warn("Object {} in image {} has invalid polygons!".format(objid, img['file_name']))
                        all_segm.append(valid_segs)
            else:
                print('seg rejected - {}'.format('xx'))

        # all geometrically-valid boxes are returned
        if len(all_boxes):
            img['boxes'] = np.asarray(all_boxes, dtype='float32')  # (n, 4)
        else:
            img['boxes'] = np.zeros((0, 4), dtype='float32')
        cls = np.asarray(all_cls, dtype='int32')  # (n,)
        if len(cls):
            assert cls.min() > 0, "Category id in COCO format must > 0!"
        img['class'] = cls  # n, always >0
        img['is_crowd'] = np.asarray(all_iscrowd, dtype='int8')  # n,
        if add_mask:
            # also required to be float32
            img['segmentation'] = all_segm

    def training_roidbs(self):
        return self.load(add_gt=True, add_mask=cfg.MODE_MASK)

    def inference_roidbs(self):
        return self.load(add_gt=False)

    def eval_inference_results(self, results, output=None):
        # continuous_id_to_COCO_id = {v: k for k, v in self.COCO_id_to_category_id.items()}
        for res in results:
            # convert to COCO's incontinuous category id
            # if res['category_id'] in continuous_id_to_COCO_id:
            #     res['category_id'] = continuous_id_to_COCO_id[res['category_id']]
            # COCO expects results in xywh format
            box = res['bbox']
            box[2] -= box[0]
            box[3] -= box[1]
            res['bbox'] = [round(float(x), 3) for x in box]

        if output is not None:
            with open(output, 'w') as f:
                json.dump(results, f)
        if len(results):
            # sometimes may crash if the results are empty?
            return self.print_coco_metrics(results)
        else:
            return {}


def register_coco_format(data_config: DataConfig):
    """
    Add COCO datasets like "coco_train201x" to the registry,
    so you can refer to them with names in `cfg.DATA.TRAIN/VAL`.

    Note that train2017==trainval35k==train2014+val2014-minival2014, and val2017==minival2014.
    """

    # split_names = ['train', 'eval']

    class_names_ls = {}
    class_names = []

    for _split in data_config.train_splits + data_config.eval_splits:  # type: DataSubsetSplit
        _name = _split.nickname
        print("register coco:", _split)
        class_names = DatasetRegistry.register(
            dataset_name=_name,
            func=lambda sp=_split:
            COCOFormatDetectionSubset(_split.ann_path,
                                      image_data_basedir=data_config.image_data_basedir), logx=_split)
        class_names_ls[_name] = class_names

    # consistency check
    for nm, cls_n in class_names_ls.items():
        assert class_names == cls_n, "Train and Val category sets are not consistent"

    class_names_include_bg = ["BG"] + list(class_names)
    for subset_name, _ in class_names_ls.items():
        DatasetRegistry.register_metadata(subset_name, 'class_names', class_names_include_bg)

    # TODO: check dataset here
    return


