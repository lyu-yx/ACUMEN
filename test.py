import argparse
import os
import warnings

import cv2
import torch
import torch.nn.parallel
import torch.utils.data as data
from loguru import logger


import utils.config as config
from engine.engine import test
from model import build_segmenter
from utils.dataset_cod import TestDataset
from utils.misc import setup_logger

warnings.filterwarnings("ignore")
cv2.setNumThreads(0)


def get_parser():
    parser = argparse.ArgumentParser(
        description='ACUMEN')
    parser.add_argument('--config',
                        default='config/codclip_vit_L14@336_noattr_3_1_50.yaml',
                        type=str,
                        help='config file')
    parser.add_argument('--opts',
                        default=None,
                        nargs=argparse.REMAINDER,
                        help='override some settings in the config.')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    if args.opts is not None:
        cfg = config.merge_cfg_from_list(cfg, args.opts)
    return cfg


@logger.catch
def main():
    args = get_parser()
    args.output_dir = os.path.join(args.map_save_path, args.exp_name)
    if args.visualize:
        args.vis_dir = os.path.join(args.output_dir, "vis")
        os.makedirs(args.vis_dir, exist_ok=True)

    # build model
    model, _ = build_segmenter(args)
    model = torch.nn.DataParallel(model).cuda()
    

    args.model_dir = os.path.join(args.output_dir, "Net_epoch_best.pth")
    if os.path.isfile(args.model_dir):
        logger.info("=> loading checkpoint '{}'".format(args.model_dir))
        checkpoint = torch.load(args.model_dir, map_location="cuda:0")
        model.load_state_dict(checkpoint, strict=True)
        logger.info("=> loaded checkpoint '{}'".format(args.model_dir))
    else:
        raise ValueError(
            "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
            .format(args.model_dir))

    # load data
    for cur_dataset in args.test_dataset:
        test_root = os.path.join(args.test_root, cur_dataset)
        print(f"Loading {cur_dataset}...")
        test_data = TestDataset(image_root=test_root + '/Imgs/',
                                gt_root=test_root + '/GT/',
                                testsize=args.input_size,
                                word_length=args.word_len)
        
        test_loader = data.DataLoader(test_data,
                                    batch_size=args.batch_size_val,
                                    shuffle=False,
                                    num_workers=args.workers_val,
                                    pin_memory=True,
                                    drop_last=False)
        if args.visualize:
            args.vis_dir = os.path.join(args.output_dir, "vis", cur_dataset)
            os.makedirs(args.vis_dir, exist_ok=True)
        if args.save_fix_attr:
            args.fix_dir = os.path.join(args.output_dir, "fix", cur_dataset)
            os.makedirs(args.fix_dir, exist_ok=True)
            args.attr_dir = os.path.join(args.output_dir, "attr", cur_dataset)
            os.makedirs(args.attr_dir, exist_ok=True)

        print(f"Loading {cur_dataset} done.")
    

        # inference
        results = test(test_loader, model, cur_dataset, args)
        print(results)

if __name__ == '__main__':
    main()
