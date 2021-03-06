import os

from dataset.data_config import DataConfig


images_data_base_dir = os.path.abspath('../../../data/datasets_coco/')
data_conf = {
        DataConfig.IMAGE_BASEDIR: images_data_base_dir,
        DataConfig.TRAIN: [
            {
                DataConfig.NICKNAME: 'decay_train',
                DataConfig.ANN_PATH: os.path.join(os.path.abspath('../../../data/'),
                                                  'coco_stack_out/web_decay_600-5.json')
            }
        ]
        ,
        DataConfig.EVAL: [
            {
                DataConfig.NICKNAME: 'decay_eval',
                DataConfig.ANN_PATH: os.path.join(os.path.abspath('../../../data/'),
                                                  'coco_stack_out/legacy_decay-3.json')
            }
        ]
    }


# images_data_base_dir = os.path.abspath('../../../data/datasets_coco/')
data_conf_tooth_only = {
        DataConfig.IMAGE_BASEDIR: os.path.abspath('../../../data/datasets_coco/'),
        DataConfig.TRAIN: [
            {
                DataConfig.NICKNAME: 'decay_train',
                DataConfig.ANN_PATH: os.path.join(os.path.abspath('../../../data/'),
                                                  'coco_stack_out/web_decay_600-6-tooth.json')
            }
        ]
        ,
        DataConfig.EVAL: [
            {
                DataConfig.NICKNAME: 'decay_eval',
                DataConfig.ANN_PATH: os.path.join(os.path.abspath('../../../data/'),
                                                  'coco_stack_out/legacy_decay-7-tooth.json')  #
            }
        ]
    }

data_conf_tooth_legacy_of = {
        DataConfig.IMAGE_BASEDIR: os.path.abspath('../../../data/datasets_coco/'),
        DataConfig.TRAIN: [
            {
                DataConfig.NICKNAME: 'decay_train',
                DataConfig.ANN_PATH: os.path.join(os.path.abspath('../../../data/'),
                                                  'coco_stack_out/legacy_decay-7-tooth.json')
            }
        ]
        ,
        DataConfig.EVAL: [
            {
                DataConfig.NICKNAME: 'decay_eval',
                DataConfig.ANN_PATH: os.path.join(os.path.abspath('../../../data/'),
                                                  'coco_stack_out/legacy_decay-7-tooth.json')  #
            }
        ]
    }


data_conf_tooth_web_of = {
        DataConfig.IMAGE_BASEDIR: os.path.abspath('../../../data/datasets_coco/'),
        DataConfig.TRAIN: [
            {
                DataConfig.NICKNAME: 'decay_train',
                DataConfig.ANN_PATH: os.path.join(os.path.abspath('../../../data/'),
                                                  'coco_stack_out/web_decay_600-6-tooth.json')
            }
        ]
        ,
        DataConfig.EVAL: [
            {
                DataConfig.NICKNAME: 'decay_eval',
                DataConfig.ANN_PATH: os.path.join(os.path.abspath('../../../data/'),
                                                  'coco_stack_out/web_decay_600-6-tooth.json')  #
            }
        ]
    }

data_conf_lesion_only = {
        DataConfig.IMAGE_BASEDIR: os.path.abspath('../../../data/datasets_coco/'),
        DataConfig.TRAIN: [
            {
                DataConfig.NICKNAME: 'decay_train',
                DataConfig.ANN_PATH: os.path.join(os.path.abspath('../../../data/'),
                                                  'coco_stack_out/web_decay_600-9-lesion.json')
            }
        ]
        ,
        DataConfig.EVAL: [
            {
                DataConfig.NICKNAME: 'decay_eval',
                DataConfig.ANN_PATH: os.path.join(os.path.abspath('../../../data/'),
                                                  'coco_stack_out/legacy_decay-8-lesion.json')  #
            }
        ]
    }

data_conf_gingivitis_only = {
        DataConfig.IMAGE_BASEDIR: os.path.abspath('../../../data/datasets_coco/'),
        DataConfig.TRAIN: [
            {
                DataConfig.NICKNAME: 'decay_train',
                DataConfig.ANN_PATH: os.path.join(os.path.abspath('../../../data/'),
                                                  'coco_stack_out/gingivitis_web_490-13-ging.json')
            }
        ]
        ,
        DataConfig.EVAL: [
            {
                DataConfig.NICKNAME: 'decay_eval',
                DataConfig.ANN_PATH: os.path.join(os.path.abspath('../../../data/'),
                                                  'coco_stack_out/legacy_decay-14-ging.json')  #
            }
        ]
    }
