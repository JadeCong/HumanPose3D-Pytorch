import sys
import os
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath('__file__')), '../../..'))
# sys.path.append(os.path.join(os.path.dirname(os.path.abspath('__file__')), '../../../nns'))
sys.path.append(os.path.join(os.getcwd(), '../../..'))
sys.path.append(os.path.join(os.getcwd(), '../../../nns'))

import shutil
import argparse
import time
import json
from datetime import datetime
from collections import defaultdict
from itertools import islice
import pickle
import copy

import numpy as np
import cv2

import torch
from torch import nn
from torch import autograd
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel

from tensorboardX import SummaryWriter

from nns.hpn.models.triangulation import RANSACTriangulationNet, AlgebraicTriangulationNet, VolumetricTriangulationNet
from nns.hpn.models.loss import KeypointsMSELoss, KeypointsMSESmoothLoss, KeypointsMAELoss, KeypointsL2Loss, VolumetricCELoss

from nns.hpn.utilities import img, multiview, op, vis, misc, cfg
from nns.hpn.datasets import dataset_human36m
from nns.hpn.datasets import dataset_utils


def parseArgs():
    parser = argparse.ArgumentParser(description='Process the argument inputs...')

    parser.add_argument("--config", type=str, required=True, help="Path, where config file is stored")
    parser.add_argument('--eval', action='store_true', help="If set, then only evaluation will be done")
    parser.add_argument('--eval_dataset', type=str, default='val', help="Dataset split on which evaluate. Can be 'train' and 'val'")
    parser.add_argument("--local_rank", type=int, help="Local rank of the process on the node")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--logdir", type=str, default="../../logs/hpn", help="Path, where logs will be stored")

    args = parser.parse_args()
    
    return args


def setupHuman36mDataloaders(config, is_train, distributed_train):
    train_dataloader = None
    
    if is_train:
        # train
        train_dataset = dataset_human36m.Human36MMultiViewDataset(
            h36m_root=config.dataset.train.h36m_root,
            pred_results_path=config.dataset.train.pred_results_path if hasattr(config.dataset.train, "pred_results_path") else None,
            train=True,
            test=False,
            image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
            labels_path=config.dataset.train.labels_path,
            with_damaged_actions=config.dataset.train.with_damaged_actions,
            scale_bbox=config.dataset.train.scale_bbox,
            kind=config.kind,
            undistort_images=config.dataset.train.undistort_images,
            ignore_cameras=config.dataset.train.ignore_cameras if hasattr(config.dataset.train, "ignore_cameras") else [],
            crop=config.dataset.train.crop if hasattr(config.dataset.train, "crop") else True,
        )

        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset) if distributed_train else None # need DistributedSampler only in training network but evaluating network

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=config.opt.batch_size,
            shuffle=config.dataset.train.shuffle and (train_sampler is None), # debatable
            sampler=train_sampler,
            collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.train.randomize_n_views,
                                                     min_n_views=config.dataset.train.min_n_views,
                                                     max_n_views=config.dataset.train.max_n_views),
            num_workers=config.dataset.train.num_workers,
            worker_init_fn=dataset_utils.worker_init_fn,
            pin_memory=True
        )

    # val
    val_dataset = dataset_human36m.Human36MMultiViewDataset(
        h36m_root=config.dataset.val.h36m_root,
        pred_results_path=config.dataset.val.pred_results_path if hasattr(config.dataset.val, "pred_results_path") else None,
        train=False,
        test=True,
        image_shape=config.image_shape if hasattr(config, "image_shape") else (256, 256),
        labels_path=config.dataset.val.labels_path,
        with_damaged_actions=config.dataset.val.with_damaged_actions,
        retain_every_n_frames_in_test=config.dataset.val.retain_every_n_frames_in_test,
        scale_bbox=config.dataset.val.scale_bbox,
        kind=config.kind,
        undistort_images=config.dataset.val.undistort_images,
        ignore_cameras=config.dataset.val.ignore_cameras if hasattr(config.dataset.val, "ignore_cameras") else [],
        crop=config.dataset.val.crop if hasattr(config.dataset.val, "crop") else True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=config.opt.val_batch_size if hasattr(config.opt, "val_batch_size") else config.opt.batch_size,
        shuffle=config.dataset.val.shuffle,
        collate_fn=dataset_utils.make_collate_fn(randomize_n_views=config.dataset.val.randomize_n_views,
                                                 min_n_views=config.dataset.val.min_n_views,
                                                 max_n_views=config.dataset.val.max_n_views),
        num_workers=config.dataset.val.num_workers,
        worker_init_fn=dataset_utils.worker_init_fn,
        pin_memory=True
    )

    return train_dataloader, val_dataloader, train_sampler


def setupDataloaders(config, is_train=True, distributed_train=False):
    if config.dataset.kind == 'human36m':
        train_dataloader, val_dataloader, train_sampler = setupHuman36mDataloaders(config, is_train, distributed_train)
    else:
        raise NotImplementedError("Unknown dataset: {}".format(config.dataset.kind))
        
    return train_dataloader, val_dataloader, train_sampler


def setupExperiments(config, model_name, is_train=True):
    prefix = "train_" if is_train else "eval_"

    if config.title:
        experiment_title = config.title + "_" + model_name
    else:
        experiment_title = model_name

    experiment_title = prefix + experiment_title
    experiment_name = '{}@{}'.format(experiment_title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    print("Experiment name: {}".format(experiment_name))

    # create the dirctory for logging the experiment
    experiment_dir = os.path.join(args.logdir, experiment_name)
    os.makedirs(experiment_dir, exist_ok=True)
    checkpoints_dir = os.path.join(experiment_dir, "checkpoints")
    os.makedirs(checkpoints_dir, exist_ok=True)

    # copy the config yaml file to the experiment directory
    shutil.copy(args.config, os.path.join(experiment_dir, "config.yaml"))

    # create the symmary writer for tensorboard
    writer = SummaryWriter(os.path.join(experiment_dir, "tb"))

    # dump config to tensorboard
    writer.add_text(misc.config_to_str(config), "config", 0)

    return experiment_dir, writer


def oneEpoch(model, criterion, opt, config, dataloader, device, epoch, n_iters_total=0, is_train=True, caption='', master=False, experiment_dir=None, writer=None):
    name = "train" if is_train else "val"
    model_type = config.model.name

    # determine the type of runing model
    if is_train:
        model.train()
    else:
        model.eval() # equivalent with model.train(mode=False)

    # define the dict for metrics and results
    metric_dict = defaultdict(list)
    results = defaultdict(list)

    # used to turn on/off gradients
    grad_context = torch.autograd.enable_grad if is_train else torch.no_grad
    with grad_context():
        end = time.time() # get the current time stamp

        # prepare the data iterator
        iterator = enumerate(dataloader)
        if is_train and config.opt.n_iters_per_epoch is not None:
            iterator = islice(iterator, config.opt.n_iters_per_epoch) # islice为获取迭代器的切片，消耗迭代器（本次取出的切片从迭代器中删除）

        for iter_i, batch in iterator:
            with autograd.detect_anomaly():
                # measure data loading time
                data_time = time.time() - end

                if batch is None:
                    print("Found None batch...")
                    continue
                
                # prepare the data bath for network forward training
                images_batch, keypoints_3d_gt, keypoints_3d_validity_gt, proj_matricies_batch = dataset_utils.prepare_batch(batch, device, config)

                # run the network foward training and get the predictions
                keypoints_2d_pred, cuboids_pred, base_points_pred = None, None, None
                if model_type == "alg" or model_type == "ransac":
                    keypoints_3d_pred, keypoints_2d_pred, heatmaps_pred, confidences_pred = model(images_batch, proj_matricies_batch, batch)
                elif model_type == "vol":
                    keypoints_3d_pred, heatmaps_pred, volumes_pred, confidences_pred, cuboids_pred, coord_volumes_pred, base_points_pred = model(images_batch, proj_matricies_batch, batch)

                batch_size, n_views, image_shape = images_batch.shape[0], images_batch.shape[1], tuple(images_batch.shape[3:])
                n_joints = keypoints_3d_pred.shape[1]

                keypoints_3d_binary_validity_gt = (keypoints_3d_validity_gt > 0.0).type(torch.float32)

                scale_keypoints_3d = config.opt.scale_keypoints_3d if hasattr(config.opt, "scale_keypoints_3d") else 1.0

                # 1-view case(for evaluating network with only one view(picture))
                if n_views == 1:
                    if config.kind == "human36m":
                        base_joint = 6 # set the keypoint 6 as base joint with only one view for projection
                    elif config.kind == "coco":
                        base_joint = 11

                    keypoints_3d_gt_transformed = keypoints_3d_gt.clone()
                    keypoints_3d_gt_transformed[:, torch.arange(n_joints) != base_joint] -= keypoints_3d_gt_transformed[:, base_joint:base_joint + 1]
                    keypoints_3d_gt = keypoints_3d_gt_transformed

                    keypoints_3d_pred_transformed = keypoints_3d_pred.clone()
                    keypoints_3d_pred_transformed[:, torch.arange(n_joints) != base_joint] -= keypoints_3d_pred_transformed[:, base_joint:base_joint + 1]
                    keypoints_3d_pred = keypoints_3d_pred_transformed

                # calculate loss for update the network weights
                total_loss = 0.0
                loss = criterion(keypoints_3d_pred * scale_keypoints_3d, keypoints_3d_gt * scale_keypoints_3d, keypoints_3d_binary_validity_gt)
                total_loss += loss
                metric_dict[f'{config.opt.criterion}'].append(loss.item())

                # calculate volumetric ce loss(check bool value for calculating volumetric triangulation network loss)
                use_volumetric_ce_loss = config.opt.use_volumetric_ce_loss if hasattr(config.opt, "use_volumetric_ce_loss") else False
                if use_volumetric_ce_loss:
                    volumetric_ce_criterion = VolumetricCELoss()

                    loss = volumetric_ce_criterion(coord_volumes_pred, volumes_pred, keypoints_3d_gt, keypoints_3d_binary_validity_gt)
                    metric_dict['volumetric_ce_loss'].append(loss.item())

                    weight = config.opt.volumetric_ce_loss_weight if hasattr(config.opt, "volumetric_ce_loss_weight") else 1.0
                    total_loss += weight * loss

                metric_dict['total_loss'].append(total_loss.item())

                # run the network backward training and update the network weights
                if is_train:
                    opt.zero_grad() # clean up the gradients
                    total_loss.backward() # run the backward for network training

                    if hasattr(config.opt, "grad_clip"):
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config.opt.grad_clip / config.opt.lr)

                    metric_dict['grad_norm_times_lr'].append(config.opt.lr * misc.calc_gradient_norm(filter(lambda x: x[1].requires_grad, model.named_parameters())))

                    opt.step() # update the network weights

                # calculate l2 metrics
                l2 = KeypointsL2Loss()(keypoints_3d_pred * scale_keypoints_3d, keypoints_3d_gt * scale_keypoints_3d, keypoints_3d_binary_validity_gt)
                metric_dict['l2'].append(l2.item())

                # calculate base point l2 metrics
                if base_points_pred is not None:
                    base_point_l2_list = []
                    for batch_i in range(batch_size):
                        base_point_pred = base_points_pred[batch_i]

                        if config.model.kind == "coco":
                            base_point_gt = (keypoints_3d_gt[batch_i, 11, :3] + keypoints_3d[batch_i, 12, :3]) / 2
                        elif config.model.kind == "mpii":
                            base_point_gt = keypoints_3d_gt[batch_i, 6, :3]

                        base_point_l2_list.append(torch.sqrt(torch.sum((base_point_pred * scale_keypoints_3d - base_point_gt * scale_keypoints_3d) ** 2)).item())

                    base_point_l2 = 0.0 if len(base_point_l2_list) == 0 else np.mean(base_point_l2_list)
                    metric_dict['base_point_l2'].append(base_point_l2)

                # save answers for evalulation
                if not is_train:
                    results['keypoints_3d'].append(keypoints_3d_pred.detach().cpu().numpy())
                    results['indexes'].append(batch['indexes'])

                # plot visualization
                if master:
                    if n_iters_total % config.vis_freq == 0: # or total_l2.item() > 500.0:
                        vis_kind = config.kind
                        if (config.transfer_cmu_to_human36m if hasattr(config, "transfer_cmu_to_human36m") else False):
                            vis_kind = "coco"

                        for batch_i in range(min(batch_size, config.vis_n_elements)):
                            keypoints_vis = vis.visualize_batch(
                                images_batch, heatmaps_pred, keypoints_2d_pred, proj_matricies_batch,
                                keypoints_3d_gt, keypoints_3d_pred,
                                kind=vis_kind,
                                cuboids_batch=cuboids_pred,
                                confidences_batch=confidences_pred,
                                batch_index=batch_i, size=5,
                                max_n_cols=10
                            )
                            writer.add_image(f"{name}/keypoints_vis/{batch_i}", keypoints_vis.transpose(2, 0, 1), global_step=n_iters_total)

                            heatmaps_vis = vis.visualize_heatmaps(
                                images_batch, heatmaps_pred,
                                kind=vis_kind,
                                batch_index=batch_i, size=5,
                                max_n_rows=10, max_n_cols=10
                            )
                            writer.add_image(f"{name}/heatmaps/{batch_i}", heatmaps_vis.transpose(2, 0, 1), global_step=n_iters_total)

                            if model_type == "vol":
                                volumes_vis = vis.visualize_volumes(
                                    images_batch, volumes_pred, proj_matricies_batch,
                                    kind=vis_kind,
                                    cuboids_batch=cuboids_pred,
                                    batch_index=batch_i, size=5,
                                    max_n_rows=1, max_n_cols=16
                                )
                                writer.add_image(f"{name}/volumes/{batch_i}", volumes_vis.transpose(2, 0, 1), global_step=n_iters_total)

                    # dump weights to tensoboard
                    if n_iters_total % config.vis_freq == 0:
                        for p_name, p in model.named_parameters():
                            try:
                                writer.add_histogram(p_name, p.clone().cpu().data.numpy(), n_iters_total)
                            except ValueError as e:
                                print(e)
                                print(p_name, p)
                                exit() # exit the python script if error occurs

                    # dump to tensorboard per-iter loss/metric stats
                    if is_train:
                        for title, value in metric_dict.items():
                            writer.add_scalar(f"{name}/{title}", value[-1], n_iters_total)

                    # measure elapsed time
                    batch_time = time.time() - end
                    end = time.time()

                    # dump to tensorboard per-iter time stats
                    writer.add_scalar(f"{name}/batch_time", batch_time, n_iters_total)
                    writer.add_scalar(f"{name}/data_time", data_time, n_iters_total)

                    # dump to tensorboard per-iter stats about sizes
                    writer.add_scalar(f"{name}/batch_size", batch_size, n_iters_total)
                    writer.add_scalar(f"{name}/n_views", n_views, n_iters_total)

                    n_iters_total += 1

    # calculate evaluation metrics
    if master:
        if not is_train:
            results['keypoints_3d'] = np.concatenate(results['keypoints_3d'], axis=0)
            results['indexes'] = np.concatenate(results['indexes'])

            try:
                scalar_metric, full_metric = dataloader.dataset.evaluate(results['keypoints_3d'])
            except Exception as e:
                print("Failed to evaluate. Reason: ", e)
                scalar_metric, full_metric = 0.0, {}

            metric_dict['dataset_metric'].append(scalar_metric)

            checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
            os.makedirs(checkpoint_dir, exist_ok=True)

            # dump results
            with open(os.path.join(checkpoint_dir, "results.pkl"), 'wb') as fout:
                pickle.dump(results, fout)

            # dump full metric
            with open(os.path.join(checkpoint_dir, "metric.json".format(epoch)), 'w') as fout:
                json.dump(full_metric, fout, indent=4, sort_keys=True)

        # dump to tensorboard per-epoch stats
        for title, value in metric_dict.items():
            writer.add_scalar(f"{name}/{title}_epoch", np.mean(value), epoch)

    return n_iters_total


def initDistributed(args):
    # check the number of aviliable gpu for whether using distributed training(WORLD_SIZE由torch.distributed.launch.py产生,具体数值为nproc_per_node*node,node为主机数)
    # world_size指进程总数，在这里就是我们使用的卡数
    if "WORLD_SIZE" not in os.environ or int(os.environ["WORLD_SIZE"]) < 1:
        print("The hardware capacity of device cannot support distributed training.")
        return False

    # local_rank指本地序号,用于本地设备分配
    torch.cuda.set_device(args.local_rank)
    assert os.environ["MASTER_PORT"], "set the MASTER_PORT variable or use pytorch launcher"
    assert os.environ["RANK"], "use pytorch launcher and explicityly state the rank of the process"
    print("The hardware capacity of device can support distributed training.")
    
    # 为GPU设置种子用于生成随机数，以使得分布式训练时各GPU初始化参数一致
    torch.manual_seed(args.seed)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")
    print("The distributed training has been initialized.")

    return True


# define the main function for dealing the hpn
def mainFunc(args):
    # check the number of avaiable GPUs for training network
    print("Number of available GPUs: {}".format(torch.cuda.device_count()))

    # check whether using the distributed training for hpn
    is_distributed = initDistributed(args)
    master = True
    if is_distributed and os.environ["RANK"]:
        master = int(os.environ["RANK"]) == 0 # 判断分布式训练中该主机是否为MASTER主节点

    if is_distributed:
        device = torch.device(args.local_rank)
    else:
        device = torch.device(0) # 指定使用哪一个GPU

    # configure the hyperparameters for trianing/evaluating network
    config = cfg.load_config(args.config) # convert the config xml file to dict type
    config.opt.n_iters_per_epoch = config.opt.n_objects_per_epoch // config.opt.batch_size

    # choose the model for training/evaluating network
    model = {
        "ransac": RANSACTriangulationNet,
        "alg": AlgebraicTriangulationNet,
        "vol": VolumetricTriangulationNet
    }[config.model.name](config, device=device).to(device) # choose the model and load the backbone network pretrained weights
    print("Choose the {} model for training network.".format(config.model.name))

    # reload the state_dict of the network if there are pretrained weights
    if config.model.init_weights:
        state_dict = torch.load(config.model.checkpoint) # load the network pretrained weights
        for key in list(state_dict.keys()):
            new_key = key.replace("module.", "") # rename the keys of state_dict
            state_dict[new_key] = state_dict.pop(key)   

        model.load_state_dict(state_dict, strict=True)
        # model.load_state_dict(state_dict, strict=False)
        print("Successfully loaded pretrained weights for whole model.")

    # criterion
    criterion_class = {
        "MSE": KeypointsMSELoss,
        "MSESmooth": KeypointsMSESmoothLoss,
        "MAE": KeypointsMAELoss
    }[config.opt.criterion]
    print("Choose the {} loss criterion for evaluating network.".format(config.opt.criterion))
    if config.opt.criterion == "MSESmooth":
        criterion = criterion_class(config.opt.mse_smooth_threshold)
    else:
        criterion = criterion_class()

    # optimizer
    opt = None
    if not args.eval: # define the optimizer for training network but no need for evaluating network
        if config.model.name == "vol":
            opt = torch.optim.Adam( # use different learning rate for different modules in vol network for training
                [{'params': model.backbone.parameters()},
                 {'params': model.process_features.parameters(), 'lr': config.opt.process_features_lr if hasattr(config.opt, "process_features_lr") else config.opt.lr},
                 {'params': model.volume_net.parameters(), 'lr': config.opt.volume_net_lr if hasattr(config.opt, "volume_net_lr") else config.opt.lr}],
                lr=config.opt.lr
            )
        else:
            opt = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=config.opt.lr)

    # set up the datasets for training or evaluating network
    print("Loading data...")
    train_dataloader, val_dataloader, train_sampler = setupDataloaders(config, distributed_train=is_distributed)
    print("The datasets have been loaded for training or evaluating network.")

    # set up the experiment configs for training or evaluating network
    experiment_dir, writer = None, None
    if master: # write the logs for training or evaluating network in master node
        experiment_dir, writer = setupExperiments(config, type(model).__name__, is_train=not args.eval)

    # check whether using multi-gpu for parallel training or evaluating network
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[device])
        print("Multiple GUPs for network training/evaluating have been configured for data parallel.")

    # choose the mode(train/val) for epoch iteration loop
    if not args.eval:
        # train loop
        n_iters_total_train, n_iters_total_val = 0, 0
        for epoch in range(config.opt.n_epochs):
            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            n_iters_total_train = oneEpoch(model, criterion, opt, config, train_dataloader, device, epoch, n_iters_total=n_iters_total_train, is_train=True, master=master, experiment_dir=experiment_dir, writer=writer)
            n_iters_total_val = oneEpoch(model, criterion, opt, config, val_dataloader, device, epoch, n_iters_total=n_iters_total_val, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer)

            if master: # check the host whether is the master node
                checkpoint_dir = os.path.join(experiment_dir, "checkpoints", "{:04}".format(epoch))
                os.makedirs(checkpoint_dir, exist_ok=True)

                torch.save(model.state_dict(), os.path.join(checkpoint_dir, "weights.pth")) # seve the state_dict of training network for every epoch in the directory checkpoints

            print(f"{n_iters_total_train} iterations done.")
    else:
        if args.eval_dataset == 'train':
            oneEpoch(model, criterion, opt, config, train_dataloader, device, 0, n_iters_total=0, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer)
        else:
            oneEpoch(model, criterion, opt, config, val_dataloader, device, 0, n_iters_total=0, is_train=False, master=master, experiment_dir=experiment_dir, writer=writer)

    print("Done.")


# define the main function entry for training hpn(human pose network)
if __name__ == '__main__':
    args = parseArgs()
    print("Argument inputs: {}".format(args))
    mainFunc(args)
