import argparse
import time
import csv
import datetime
from path import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import models

import custom_transforms
from utils import tensor2array, save_checkpoint
from datasets.sequence_folders import SequenceFolder
from datasets.pair_folders import PairFolder
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_errors
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter

# ---- W&B: NEW ----
import wandb
# -------------------
# -------- W&B helpers & logger (ADD) -----------------------------------------
import numpy as np, matplotlib.cm as cm, torch

_MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
_STD  = np.array([0.225, 0.225, 0.225], dtype=np.float32)

def _to_np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

def _denorm_chw(img_chw, times=1):
    """Undo Normalize(mean=0.45,std=0.225); set times=2 if images look too dark."""
    x = _to_np(img_chw).astype(np.float32)  # CxHxW
    for _ in range(times):
        x = x * _STD[:, None, None] + _MEAN[:, None, None]
    return x

def _chw_to_hwc_uint8(x01_chw):
    x = np.transpose(x01_chw, (1, 2, 0))        # HWC, float
    x = np.clip(x, 0.0, 1.0) * 255.0
    return x.astype(np.uint8)                   # HxWx3 uint8

def _colorize_depth_auto(depth_hw, cmap="magma"):
    """depth HxW -> HxWx3 uint8 with robust auto-range."""
    d = _to_np(depth_hw).astype(np.float32)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.isfinite(d) & (d > 0)
    vmax = np.percentile(d[mask], 95.0) if mask.any() else 1.0
    vmin = 0.0
    d = np.clip((d - vmin) / max(vmax - vmin, 1e-6), 0.0, 1.0)
    rgba = cm.get_cmap(cmap)(d)                 # HxWx4 in [0,1]
    return (rgba[..., :3] * 255.0).astype(np.uint8)

def log(mode, step, *, losses=None, tgt_img=None, ref_imgs=None, pred_depth=None, gt_depth=None):
    """W&B logger.
    mode: 'train' or 'val'
    step: int (iteration or epoch)
    losses: dict of floats/tensors
    tgt_img: torch [B,3,H,W] normalized with mean=0.45,std=0.225
    ref_imgs: list of torch [B,3,H,W] (optional)
    pred_depth: torch [B,1,H,W] (optional)
    gt_depth:   torch [B,H,W]   (optional)
    """
    log_dict = {}

    # 1) Scalars
    if losses:
        for k, v in losses.items():
            if isinstance(v, torch.Tensor):
                v = float(v.detach().cpu().item())
            log_dict[f"{mode}/{k}"] = v

    # 2) Images (take first sample in batch for compact logs)
    imgs = []
    try:
        j = 0

        # RGB target (de-normalize once; if too dark, de-normalize twice)
        if tgt_img is not None and tgt_img.shape[0] > 0:
            rgb_once  = _denorm_chw(tgt_img[j], times=1)
            rgb_hwc   = _chw_to_hwc_uint8(rgb_once)
            if rgb_hwc.mean() < 5:              # almost black? try undo twice
                rgb_hwc = _chw_to_hwc_uint8(_denorm_chw(tgt_img[j], times=2))
            imgs.append(wandb.Image(rgb_hwc, caption=f"{mode}/input"))

        # reference frames
        if ref_imgs:
            for idx, r in enumerate(ref_imgs[:2]):         # at most 2 refs to keep light
                r_once  = _denorm_chw(r[j], times=1)
                r_hwc   = _chw_to_hwc_uint8(r_once)
                if r_hwc.mean() < 5:
                    r_hwc = _chw_to_hwc_uint8(_denorm_chw(r[j], times=2))
                imgs.append(wandb.Image(r_hwc, caption=f"{mode}/ref{idx}"))

        # predicted depth
        if pred_depth is not None and pred_depth.shape[0] > 0:
            # match depth size to RGB
            rgb = tgt_img[j]
            rgb = _denorm_chw(rgb, times=2)        # undo normalization
            rgb = _chw_to_hwc_uint8(rgb)

            import torch.nn.functional as F
            rgb_h, rgb_w = rgb.shape[0], rgb.shape[1]

            # ensure depth matches RGB resolution
            depth_resized = F.interpolate(
                pred_depth, (rgb_h, rgb_w),
                mode="bilinear", align_corners=False
            )

            d_color = _colorize_depth_auto(depth_resized[j, 0])

            wandb.log({
                f"{mode}/image_input": wandb.Image(rgb),
                f"{mode}/depth_magma": wandb.Image(d_color)
            }, step=step)


        # ground-truth depth (if available)
        if gt_depth is not None:
            depth_gt_resized = F.interpolate(
                gt_depth.unsqueeze(1), (rgb_h, rgb_w),
                mode="nearest"
            ).squeeze(1)
            gt_color = _colorize_depth_auto(depth_gt_resized[j])


    except Exception as e:
        print(f"[wandb log warn] image packing failed at step {step}: {e}")

    if imgs:
        log_dict[f"{mode}/images"] = imgs

    if log_dict:
        wandb.log(log_dict, step=step)
# ------------------------------------------------------------------------------

parser = argparse.ArgumentParser(description='Structure from Motion Learner training on KITTI and CityScapes Dataset',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--folder-type', type=str, choices=['sequence', 'pair'], default='sequence', help='the dataset dype to train')
parser.add_argument('--sequence-length', type=int, metavar='N', help='sequence length for training', default=3)
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N', help='number of data loading workers')
parser.add_argument('--epochs', default=200, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--epoch-size', default=0, type=int, metavar='N', help='manual epoch size (will match dataset size if not set)')
parser.add_argument('-b', '--batch-size', default=4, type=int, metavar='N', help='mini-batch size')
parser.add_argument('--lr', '--learning-rate', default=1e-4, type=float, metavar='LR', help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum for sgd, alpha parameter for adam')
parser.add_argument('--beta', default=0.999, type=float, metavar='M', help='beta parameters for adam')
parser.add_argument('--weight-decay', '--wd', default=0, type=float, metavar='W', help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N', help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed for random functions, and network initialization')
parser.add_argument('--log-summary', default='progress_log_summary.csv', metavar='PATH', help='csv where to save per-epoch train and valid stats')
parser.add_argument('--log-full', default='progress_log_full.csv', metavar='PATH', help='csv where to save per-gradient descent train stats')
parser.add_argument('--log-output', action='store_true', help='will log dispnet outputs at validation step')
parser.add_argument('--resnet-layers',  type=int, default=18, choices=[18, 50], help='number of ResNet layers for depth estimation.')
parser.add_argument('--num-scales', '--number-of-scales', type=int, help='the number of scales', metavar='W', default=1)
parser.add_argument('-p', '--photo-loss-weight', type=float, help='weight for photometric loss', metavar='W', default=1)
parser.add_argument('-s', '--smooth-loss-weight', type=float, help='weight for disparity smoothness loss', metavar='W', default=0.1)
parser.add_argument('-c', '--geometry-consistency-weight', type=float, help='weight for depth consistency loss', metavar='W', default=0.5)
parser.add_argument('--with-ssim', type=int, default=1, help='with ssim or not')
parser.add_argument('--with-mask', type=int, default=1, help='with the the mask for moving objects and occlusions or not')
parser.add_argument('--with-auto-mask', type=int,  default=0, help='with the the mask for stationary points')
parser.add_argument('--with-pretrain', type=int,  default=1, help='with or without imagenet pretrain for resnet')
parser.add_argument('--dataset', type=str, choices=['kitti', 'nyu', 'endovis'], default='kitti', help='the dataset to train')
parser.add_argument('--pretrained-disp', dest='pretrained_disp', default=None, metavar='PATH', help='path to pre-trained dispnet model')
parser.add_argument('--pretrained-pose', dest='pretrained_pose', default=None, metavar='PATH', help='path to pre-trained Pose net model')
parser.add_argument('--name', dest='name', type=str, required=True, help='name of the experiment, checkpoints are stored in checpoints/name')
parser.add_argument('--padding-mode', type=str, choices=['zeros', 'border'], default='zeros',
                    help='padding mode for image warping : this is important for photometric differenciation when going outside target image.'
                         ' zeros will null gradients outside target image.'
                         ' border will only null gradients of the coordinate outside (x or y)')
parser.add_argument('--with-gt', action='store_true', help='use ground truth for validation. \
                    You need to store it in npy 2D arrays see data/kitti_raw_loader.py for an example')


best_error = -1
n_iter = 0

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)


def main():
    global best_error, n_iter, device
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = Path(args.name)
    args.save_path = 'checkpoints'/save_path/timestamp
    print('=> will save everything to {}'.format(args.save_path))
    args.save_path.makedirs_p()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

    # ---- W&B: init (uses your VM account; project/name can be overridden via env) ----
    wandb_run_name = f"{args.name}/{timestamp}"
    wandb.init(name=wandb_run_name, config=vars(args), allow_val_change=True)
    # -------------------------------------------------------------------------------

    # Data loading code
    normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])

    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([custom_transforms.ArrayToTensor(), normalize])

    print("=> fetching scenes in '{}'".format(args.data))
    if args.folder_type == 'sequence':
        train_set = SequenceFolder(
            args.data,
            transform=train_transform,
            seed=args.seed,
            train=True,
            sequence_length=args.sequence_length,
            dataset=args.dataset
        )
    else:
        train_set = PairFolder(
            args.data,
            seed=args.seed,
            train=True,
            transform=train_transform
        )


    # if no Groundtruth is avalaible, Validation set is the same type as training set to measure photometric loss from warping
    if args.with_gt:
        from datasets.validation_folders import ValidationSet
        val_set = ValidationSet(
            args.data,
            transform=valid_transform,
            dataset=args.dataset
        )
    else:
        val_set = SequenceFolder(
            args.data,
            transform=valid_transform,
            seed=args.seed,
            train=False,
            sequence_length=args.sequence_length,
            dataset=args.dataset
        )
    print('{} samples found in {} train scenes'.format(len(train_set), len(train_set.scenes)))
    print('{} samples found in {} valid scenes'.format(len(val_set), len(val_set.scenes)))
    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # create model
    print("=> creating model")
    disp_net = models.DispResNet(args.resnet_layers, args.with_pretrain).to(device)
    pose_net = models.PoseResNet(18, args.with_pretrain).to(device)

    # ---- W&B: watch models (light frequency to avoid overhead) ----
    wandb.watch(disp_net, log='gradients', log_freq=1000)
    wandb.watch(pose_net, log='gradients', log_freq=1000)
    # ---------------------------------------------------------------

    # load parameters
    if args.pretrained_disp:
        print("=> using pre-trained weights for DispResNet")
        weights = torch.load(args.pretrained_disp)
        disp_net.load_state_dict(weights['state_dict'], strict=False)

    if args.pretrained_pose:
        print("=> using pre-trained weights for PoseResNet")
        weights = torch.load(args.pretrained_pose)
        pose_net.load_state_dict(weights['state_dict'], strict=False)

    disp_net = torch.nn.DataParallel(disp_net)
    pose_net = torch.nn.DataParallel(pose_net)

    print('=> setting adam solver')
    optim_params = [
        {'params': disp_net.parameters(), 'lr': args.lr},
        {'params': pose_net.parameters(), 'lr': args.lr}
    ]
    optimizer = torch.optim.Adam(optim_params,
                                 betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    with open(args.save_path/args.log_summary, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'validation_loss'])

    with open(args.save_path/args.log_full, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter='\t')
        writer.writerow(['train_loss', 'photo_loss', 'smooth_loss', 'geometry_consistency_loss'])

    logger = TermLogger(n_epochs=args.epochs, train_size=min(len(train_loader), args.epoch_size), valid_size=len(val_loader))
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        # train for one epoch
        logger.reset_train_bar()
        train_loss = train(args, train_loader, disp_net, pose_net, optimizer, args.epoch_size, logger, training_writer)
        logger.train_writer.write(' * Avg Loss : {:.3f}'.format(train_loss))
        # ---- W&B: log average train loss per epoch ----
        wandb.log({'epoch': epoch, 'train/avg_total_loss': train_loss})
        # ------------------------------------------------

        # evaluate on validation set
        logger.reset_valid_bar()
        if args.with_gt:
            errors, error_names = validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers)
        else:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, output_writers)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        # ---- W&B: log validation metrics per epoch ----
        for error, name in zip(errors, error_names):
            wandb.log({f'val/{name}': error, 'epoch': epoch})
        # ------------------------------------------------

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)

        # Up to you to chose the most relevant error to measure your model's performance, careful some measures are to maximize (such as a1,a2,a3)
        decisive_error = errors[1]
        if best_error < 0:
            best_error = decisive_error

        # remember lowest error and save checkpoint
        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)
        save_checkpoint(
            args.save_path, {
                'epoch': epoch + 1,
                'state_dict': disp_net.module.state_dict()
            }, {
                'epoch': epoch + 1,
                'state_dict': pose_net.module.state_dict()
            },
            is_best)

        with open(args.save_path/args.log_summary, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([train_loss, decisive_error])
    logger.epoch_bar.finish()


def train(args, train_loader, disp_net, pose_net, optimizer, epoch_size, logger, train_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight

    # switch to train mode
    disp_net.train()
    pose_net.train()

    end = time.time()
    logger.train_bar.update(0)

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0

        # measure data loading time
        data_time.update(time.time() - end)
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)

        # compute output
        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, args.with_auto_mask, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)
        loss = w1*loss_1 + w2*loss_2 + w3*loss_3

        if log_losses:
            train_writer.add_scalar('photometric_error', loss_1.item(), n_iter)
            train_writer.add_scalar('disparity_smoothness_loss', loss_2.item(), n_iter)
            train_writer.add_scalar('geometry_consistency_loss', loss_3.item(), n_iter)
            train_writer.add_scalar('total_loss', loss.item(), n_iter)

        # ---- W&B: per-step scalars ----
        wandb.log({
            'step': n_iter,
            'train/photometric_error': loss_1.item(),
            'train/disparity_smoothness_loss': loss_2.item(),
            'train/geometry_consistency_loss': loss_3.item(),
            'train/total_loss': loss.item()
        }, step=n_iter)
        # --------------------------------

        # ---- W&B: images every 250 steps (input RGB + colorized depth @ scale 0) ----
        if n_iter % 250 == 0:
            # tgt_depth is a list over scales; tgt_depth[0][0] is [B,1,H,W]
            log(
                "train", n_iter,
                losses={
                    "total_loss": loss.item(),
                    "photometric_error": loss_1.item(),
                    "disparity_smoothness_loss": loss_2.item(),
                    "geometry_consistency_loss": loss_3.item()
                },
                tgt_img=tgt_img,               # [B,3,H,W] normalized
                ref_imgs=ref_imgs,             # list of [B,3,H,W]
                pred_depth=tgt_depth[0][0]     # [B,1,H,W] finest scale
            )

        # -----------------------------------------------------------------------------

        # record loss and EPE
        losses.update(loss.item(), args.batch_size)

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path/args.log_full, 'a') as csvfile:
            writer = csv.writer(csvfile, delimiter='\t')
            writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])
        logger.train_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.train_writer.write('Train: Time {} Data {} Loss {}'.format(batch_time, data_time, losses))
        if i >= epoch_size - 1:
            break

        n_iter += 1

    return losses.avg[0]


@torch.no_grad()
def validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, output_writers=[]):
    global device
    batch_time = AverageMeter()
    losses = AverageMeter(i=4, precision=4)
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()
    pose_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

        # compute output
        tgt_depth = [1 / disp_net(tgt_img)]
        ref_depths = []
        for ref_img in ref_imgs:
            ref_depth = [1 / disp_net(ref_img)]
            ref_depths.append(ref_depth)

        if log_outputs and i < len(output_writers):
            if epoch == 0:
                output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)

            output_writers[i].add_image('val Dispnet Output Normalized',
                                        tensor2array(1/tgt_depth[0][0], max_value=None, colormap='magma'),
                                        epoch)
            output_writers[i].add_image('val Depth Output',
                                        tensor2array(tgt_depth[0][0], max_value=10),
                                        epoch)

        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
                                                         poses, poses_inv, args.num_scales, args.with_ssim,
                                                         args.with_mask, False, args.padding_mode)

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss_1 = loss_1.item()
        loss_2 = loss_2.item()
        loss_3 = loss_3.item()

        loss = loss_1
        losses.update([loss, loss_1, loss_2, loss_3])
        if i == 0:  # log just the first batch of the epoch
            log(
                "val", step=epoch,
                losses={
                    "Total loss": loss,
                    "Photo loss": loss_1,
                    "Smooth loss": loss_2,
                    "Consistency loss": loss_3
                },
                tgt_img=tgt_img,
                ref_imgs=ref_imgs,
                pred_depth=tgt_depth[0][0]
            )


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

    logger.valid_bar.update(len(val_loader))
    return losses.avg, ['Total loss', 'Photo loss', 'Smooth loss', 'Consistency loss']


@torch.no_grad()
def validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers=[]):
    global device
    batch_time = AverageMeter()
    error_names = ['abs_diff', 'abs_rel', 'sq_rel', 'a1', 'a2', 'a3']
    errors = AverageMeter(i=len(error_names))
    log_outputs = len(output_writers) > 0

    # switch to evaluate mode
    disp_net.eval()

    end = time.time()
    logger.valid_bar.update(0)
    for i, (tgt_img, depth) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        depth = depth.to(device)

        # check gt
        if depth.nelement() == 0:
            continue

        # compute output
        output_disp = disp_net(tgt_img)
        output_depth = 1/output_disp[:, 0]
        if i == 0:
            # output_depth: [B,H,W]; depth: [B,H,W]
            log(
                "val", step=epoch,
                losses={name: err for name, err in zip(error_names, errors.avg)},  # or just leave None
                tgt_img=tgt_img,
                pred_depth=output_depth.unsqueeze(1),   # [B,1,H,W]
                gt_depth=depth                          # [B,H,W]
            )


        if log_outputs and i < len(output_writers):
            if epoch == 0:
                output_writers[i].add_image('val Input', tensor2array(tgt_img[0]), 0)
                depth_to_show = depth[0]
                output_writers[i].add_image('val target Depth',
                                            tensor2array(depth_to_show, max_value=10),
                                            epoch)
                depth_to_show[depth_to_show == 0] = 1000
                disp_to_show = (1/depth_to_show).clamp(0, 10)
                output_writers[i].add_image('val target Disparity Normalized',
                                            tensor2array(disp_to_show, max_value=None, colormap='magma'),
                                            epoch)

            output_writers[i].add_image('val Dispnet Output Normalized',
                                        tensor2array(output_disp[0], max_value=None, colormap='magma'),
                                        epoch)
            output_writers[i].add_image('val Depth Output',
                                        tensor2array(output_depth[0], max_value=10),
                                        epoch)

        if depth.nelement() != output_depth.nelement():
            b, h, w = depth.size()
            output_depth = torch.nn.functional.interpolate(output_depth.unsqueeze(1), [h, w]).squeeze(1)

        errors.update(compute_errors(depth, output_depth, args.dataset))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Abs Error {:.4f} ({:.4f})'.format(batch_time, errors.val[0], errors.avg[0]))
    logger.valid_bar.update(len(val_loader))
    return errors.avg, error_names


def compute_depth(disp_net, tgt_img, ref_imgs):
    tgt_depth = [1/disp for disp in disp_net(tgt_img)]

    ref_depths = []
    for ref_img in ref_imgs:
        ref_depth = [1/disp for disp in disp_net(ref_img)]
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))

    return poses, poses_inv


if __name__ == '__main__':
    main()
