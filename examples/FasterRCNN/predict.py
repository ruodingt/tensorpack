#!/usr/bin/env python
# -*- coding: utf-8 -*-
import argparse
import itertools
import re

import numpy as np
import os
import shutil
import tensorflow as tf
import cv2
import tqdm

import tensorpack.utils.viz as tpviz
from tensorpack.predict import MultiTowerOfflinePredictor, OfflinePredictor, PredictConfig
from tensorpack.tfutils import SmartInit, get_tf_version_tuple
from tensorpack.tfutils.export import ModelExporter
from tensorpack.utils import fs, logger

from dataset import DatasetRegistry, register_coco, register_balloon
from config import config as cfg
from config import finalize_configs
from data import get_eval_dataflow, get_train_dataflow
from eval import DetectionResult, multithread_predict_dataflow, predict_image, run_resize_image, predict_resized_image
from modeling.generalized_rcnn import ResNetC4Model, ResNetFPNModel
from viz import (
    draw_annotation, draw_final_outputs, draw_predictions,
    draw_proposal_recall, draw_final_outputs_blackwhite)

from data_prepare.coco_format import COCOFormatDataLoader
from dataset import register_coco, register_balloon, register_coco_format


def do_visualize(model, model_path, nr_visualize=100, output_dir='output'):
    """
    Visualize some intermediate results (proposals, raw predictions) inside the pipeline.
    """
    df = get_train_dataflow()
    df.reset_state()

    pred = OfflinePredictor(PredictConfig(
        model=model,
        session_init=SmartInit(model_path),
        input_names=['image', 'gt_boxes', 'gt_labels'],
        output_names=[
            'generate_{}_proposals/boxes'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'generate_{}_proposals/scores'.format('fpn' if cfg.MODE_FPN else 'rpn'),
            'fastrcnn_all_scores',
            'output/boxes',
            'output/scores',
            'output/labels',
        ]))

    if os.path.isdir(output_dir):
        shutil.rmtree(output_dir)
    fs.mkdir_p(output_dir)
    with tqdm.tqdm(total=nr_visualize) as pbar:
        for idx, dp in itertools.islice(enumerate(df), nr_visualize):
            img, gt_boxes, gt_labels = dp['image'], dp['gt_boxes'], dp['gt_labels']

            rpn_boxes, rpn_scores, all_scores, \
            final_boxes, final_scores, final_labels = pred(img, gt_boxes, gt_labels)

            # draw groundtruth boxes
            gt_viz = draw_annotation(img, gt_boxes, gt_labels)
            # draw best proposals for each groundtruth, to show recall
            proposal_viz, good_proposals_ind = draw_proposal_recall(img, rpn_boxes, rpn_scores, gt_boxes)
            # draw the scores for the above proposals
            score_viz = draw_predictions(img, rpn_boxes[good_proposals_ind], all_scores[good_proposals_ind])

            results = [DetectionResult(*args) for args in
                       zip(final_boxes, final_scores, final_labels,
                           [None] * len(final_labels))]
            final_viz = draw_final_outputs(img, results)

            viz = tpviz.stack_patches([
                gt_viz, proposal_viz,
                score_viz, final_viz], 2, 2)

            if os.environ.get('DISPLAY', None):
                tpviz.interactive_imshow(viz)
            cv2.imwrite("{}/{:03d}.png".format(output_dir, idx), viz)
            pbar.update()


def do_evaluate(pred_config, output_file):
    num_tower = max(cfg.TRAIN.NUM_GPUS, 1)
    graph_funcs = MultiTowerOfflinePredictor(
        pred_config, list(range(num_tower))).get_predictors()

    for dataset in cfg.DATA.VAL:
        logger.info("Evaluating {} ...".format(dataset))
        dataflows = [
            get_eval_dataflow(dataset, shard=k, num_shards=num_tower)
            for k in range(num_tower)]
        all_results = multithread_predict_dataflow(dataflows, graph_funcs)
        output = output_file + '-' + dataset
        DatasetRegistry.get(dataset).eval_inference_results(all_results, output)


# from pycocotools.coco import annToMask
def convert_box_mode_xywh_2_xyxy(box):
    return box[0], box[1], box[0] + box[2], box[1] + box[3]


def do_sanity_check(pred_func, output_dir='/root/dentalpoc/logs/xxxxx'):
    # num_tower = max(cfg.TRAIN.NUM_GPUS, 1)
    # graph_funcs = MultiTowerOfflinePredictor(
    #     pred_config, list(range(num_tower))).get_predictors()
    os.makedirs(output_dir, exist_ok=True)

    for dataset in cfg.DATA.VAL:
        logger.info("sanity checking {} ...".format(dataset))
        # dataflows = [
        #     get_eval_dataflow(dataset, shard=k, num_shards=num_tower, add_gt=True)
        #     for k in range(num_tower)]
        # all_results = multithread_predict_dataflow(dataflows, graph_funcs)
        coco_format_detetcion = DatasetRegistry.get(dataset)
        coco_object = coco_format_detetcion.coco
        for _im_id, _img_dic in list(coco_object.imgs.items())[1:]:
            _img_path = _img_dic['path']
            _img_seg_polygons = coco_object.imgToAnns[_im_id]
            detection_ground_truths = list(
                map(lambda x: DetectionResult(box=convert_box_mode_xywh_2_xyxy(x['bbox']),
                                              score=1.0,
                                              class_id=x['category_id'],
                                              mask=coco_object.annToMask(x)),
                    _img_seg_polygons))

            print("S======check")
            _predict_with_gt(pred_func=pred_func,
                             input_file=_img_path,
                             ground_truths=detection_ground_truths,
                             output_dir=output_dir)

        xxx = 0
        # output = output_file + '-' + dataset
        # DatasetRegistry.get(dataset).eval_inference_results(all_results, output)


def _predict_with_gt(pred_func, input_file, ground_truths, output_dir=None):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)

    resized_img, orig_shape, scale = run_resize_image(img)
    results = predict_image(img, pred_func)
    if cfg.MODE_MASK:
        final = draw_final_outputs_blackwhite(img, results)
    else:
        final = draw_final_outputs(img, results)

    image_with_gt = draw_final_outputs(img, ground_truths)
    viz = np.concatenate((image_with_gt, final), axis=1)
    out_path = os.path.join(output_dir, re.sub('/', '-', input_file) + '.out.png')
    cv2.imwrite(out_path, viz)
    logger.info("Inference output for {} written to\n {}".format(input_file, out_path))
    # tpviz.interactive_imshow(viz)


def do_predict(pred_func, input_file):
    img = cv2.imread(input_file, cv2.IMREAD_COLOR)
    results = predict_image(img, pred_func)
    if cfg.MODE_MASK:
        final = draw_final_outputs_blackwhite(img, results)
    else:
        final = draw_final_outputs(img, results)
    viz = np.concatenate((img, final), axis=1)
    cv2.imwrite("output.png", viz)
    logger.info("Inference output for {} written to output.png".format(input_file))
    tpviz.interactive_imshow(viz)


def set_config_A():
    cfg.DATA.BASEDIR = os.path.abspath('../../../data/toooth')
    cfg.MODE_FPN = True
    cfg.DATA.VAL = ('coco_formated_eval',)
    cfg.DATA.TRAIN = ('coco_formated_train',)
    cfg.TRAIN.BASE_LR = 1e-3
    cfg.TRAIN.EVAL_PERIOD = 1
    cfg.TRAIN.LR_SCHEDULE = [1000]
    cfg.PREPROC.TRAIN_SHORT_EDGE_SIZE = [600, 1200]
    cfg.TRAIN.CHECKPOINT_PERIOD = 1
    cfg.DATA.NUM_WORKERS = 1
    cfg.TRAIN.CHECKPOINT_PERIOD = 1


"""
./predict.py --predict input1.jpg input2.jpg --load /path/to/Trained-Model-Checkpoint --config SAME-AS-TRAINING
"""

if __name__ == '__main__':
    set_config_A()
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', help='load a model for evaluation.', required=False)
    parser.add_argument('--visualize', action='store_true', help='visualize intermediate results')
    parser.add_argument('--evaluate', help="Run evaluation. "
                                           "This argument is the path to the output json evaluation file")
    parser.add_argument('--predict', help="Run prediction on a given image. "
                                          "This argument is the path to the input image file", nargs='+')

    parser.add_argument('--sanity_check', help="Run prediction on a given image vs ground truth", action='store_true')
    parser.add_argument('--benchmark', action='store_true', help="Benchmark the speed of the model + postprocessing")
    parser.add_argument('--config', help="A list of KEY=VALUE to overwrite those defined in config.py",
                        nargs='+')
    parser.add_argument('--output-pb', help='Save a model to .pb')
    parser.add_argument('--output-serving', help='Save a model to serving file')

    args = parser.parse_args()

    args.load = "/root/dentalpoc/logs/tootth2/checkpoint"

    if args.config:
        cfg.update_config_from_args(args.config)
    # register_coco(cfg.DATA.BASEDIR)  # add COCO datasets to the registry
    # register_balloon(cfg.DATA.BASEDIR)

    import os

    annotations_dir = os.path.join(cfg.DATA.BASEDIR, 'annotations')
    data_load = COCOFormatDataLoader(project_root_dir='', coco_dir=annotations_dir)
    _latest = data_load.latest()
    latest_coco = next(_latest)
    register_coco_format(None, data_meta_info=latest_coco)

    MODEL = ResNetFPNModel() if cfg.MODE_FPN else ResNetC4Model()

    if not tf.test.is_gpu_available():
        from tensorflow.python.framework import test_util

        assert get_tf_version_tuple() >= (1, 7) and test_util.IsMklEnabled(), \
            "Inference requires either GPU support or MKL support!"
    assert args.load
    finalize_configs(is_training=False)

    if args.predict or args.visualize:
        cc = list(args.predict)
        cfg.TEST.RESULT_SCORE_THRESH = cfg.TEST.RESULT_SCORE_THRESH_VIS

    if args.visualize:
        do_visualize(MODEL, args.load)
    else:
        predcfg = PredictConfig(
            model=MODEL,
            session_init=SmartInit(args.load),
            input_names=MODEL.get_inference_tensor_names()[0],
            output_names=MODEL.get_inference_tensor_names()[1])

        if args.output_pb:
            ModelExporter(predcfg).export_compact(args.output_pb, optimize=False)
        elif args.output_serving:
            ModelExporter(predcfg).export_serving(args.output_serving, optimize=False)

        if args.predict:
            predictor = OfflinePredictor(predcfg)
            for image_file in args.predict:
                do_predict(predictor, image_file)
        elif args.sanity_check:
            predictor = OfflinePredictor(predcfg)
            do_sanity_check(pred_func=predictor, output_dir='/root/dentalpoc/out')
        elif args.evaluate:
            assert args.evaluate.endswith('.json'), args.evaluate
            do_evaluate(predcfg, args.evaluate)
        elif args.benchmark:
            df = get_eval_dataflow(cfg.DATA.VAL[0])
            df.reset_state()
            predictor = OfflinePredictor(predcfg)
            for _, img in enumerate(tqdm.tqdm(df, total=len(df), smoothing=0.5)):
                # This includes post-processing time, which is done on CPU and not optimized
                # To exclude it, modify `predict_image`.
                predict_image(img[0], predictor)
