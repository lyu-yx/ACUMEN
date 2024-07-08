import argparse
import datetime
import os
import shutil
import sys
import time
import warnings
from functools import partial

import cv2
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data as data
from loguru import logger
from torch.optim.lr_scheduler import MultiStepLR
from multiprocessing import Manager

import utils.config as config
# import wandb
from utils.dataset_cod import CamObjDataset, TestDataset
from engine.engine import train, val
from model import build_segmenter
from utils.misc import (init_random_seed, set_random_seed, setup_logger,
                        worker_init_fn)

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
    args.manual_seed = init_random_seed(args.manual_seed)
    set_random_seed(args.manual_seed, deterministic=False)
    args.ngpus_per_node = torch.cuda.device_count()
    args.world_size = args.ngpus_per_node * args.world_size 
    
    # make shared dictionary among mp 
    manager = mp.Manager()
    shared_vars = manager.dict()
    shared_vars['best_score'] = 0
    shared_vars['best_epoch'] = 0
    shared_vars['best_metric_dict'] = dict()
    
    mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args, shared_vars))


def main_worker(gpu, args, shared_vars):
    args.output_dir = os.path.join(args.map_save_path, args.exp_name)

    # local rank & global rank
    args.gpu = gpu
    args.rank = args.rank * args.ngpus_per_node + gpu
    torch.cuda.set_device(args.gpu)

    # logger
    setup_logger(args.output_dir,
                 distributed_rank=args.gpu,
                 filename="train.log",
                 mode="a")

    # dist init
    dist.init_process_group(backend=args.dist_backend,
                            init_method=args.dist_url,
                            world_size=args.world_size,
                            rank=args.rank)

    # wandb
    # if args.rank == 0:
    #     wandb.init(job_type="training",
    #                mode="online",
    #                config=args,
    #                project="CLIPCOD-model-with-desc$vision&fix",
    #                name=args.exp_name,
    #                tags=[args.dataset, args.clip_pretrain])
    dist.barrier(device_ids=[args.gpu])

    # build model
    print("building model")
    model, param_list = build_segmenter(args)
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    # logger.info(model)
    model = nn.parallel.DistributedDataParallel(model.cuda(),
                                                device_ids=[args.gpu],
                                                find_unused_parameters=True)

    # build optimizer & lr scheduler
    optimizer = torch.optim.Adam(param_list,
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer,
                            milestones=args.milestones,
                            gamma=args.lr_decay)
    scaler = amp.GradScaler()

    
    # build dataset
    print('building dataset...')
    train_data = CamObjDataset(image_root=args.train_root + 'Imgs/',
                              gt_root=args.train_root + 'GT/',
                              fix_root=args.train_root + 'Fix/',
                              overall_desc_root=args.train_root + 'Desc/overall_description/',
                              camo_desc_root=args.train_root + 'Desc/attribute_description/',
                              attri_root=args.train_root + 'Desc/attribute_contribution/',
                              trainsize=args.input_size,
                              word_length=args.word_len)
    
    val_data = TestDataset(image_root=args.val_root + 'Imgs/',
                              gt_root=args.val_root + 'GT/',
                              testsize=args.input_size,
                              word_length=args.word_len)
    # total_step = len(train_data)

    # build dataloader
    print('building dataloader...')
    init_fn = partial(worker_init_fn,
                      num_workers=args.workers,
                      rank=args.rank,
                      seed=args.manual_seed)
    train_sampler = data.distributed.DistributedSampler(train_data, shuffle=True)
    val_sampler = data.distributed.DistributedSampler(val_data, shuffle=False)


    train_loader = data.DataLoader(train_data,
                                   batch_size=args.batch_size,
                                   shuffle=False,
                                   num_workers=args.workers,
                                   pin_memory=True,
                                   worker_init_fn=init_fn,
                                   sampler=train_sampler,
                                   drop_last=False)
    val_loader = data.DataLoader(val_data,
                                 batch_size=args.batch_size_val,
                                 shuffle=False,
                                 num_workers=args.workers_val,
                                 pin_memory=True,
                                 sampler=val_sampler,
                                 drop_last=False)
    
    # resume
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume, map_location=lambda storage: storage.cuda())
            args.start_epoch = checkpoint['epoch']
            best_IoU = checkpoint["best_iou"]
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            logger.info("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']))
        else:
            raise ValueError(
                "=> resume failed! no checkpoint found at '{}'. Please check args.resume again!"
                .format(args.resume))

    # start training
    start_time = time.time()
    manager = Manager()
    lock = manager.Lock()
    
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1

        # shuffle loader
        train_sampler.set_epoch(epoch_log)

        # train
        if epoch_log == 1:
            print("start training")
            
        train(train_loader, model, optimizer, scheduler, scaler, epoch_log, args)

        # evaluation & save
        # if epoch > args.epochs//2:
        val(val_loader, model, epoch_log, args, shared_vars, lock)

        # update lr
        scheduler.step(epoch_log)

        # cache cleaning
        if args.clean_cache:
            torch.cuda.empty_cache()

    # time.sleep(2)
    # if dist.get_rank() == 0:
    #     wandb.finish()

    # logger.info("* Best IoU={} * ".format(best_IoU))
    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    logger.info('* Training time {} *'.format(total_time_str))


if __name__ == '__main__':
    main()
    sys.exit(0)
