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
import matplotlib
matplotlib.use("Agg")  # safe for headless servers
import matplotlib.cm as cm
# -------------------

# ------------------- W&B helpers (pure logging; no training changes) -------------------
def _to_np(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    return np.asarray(x)

_CMAP = cm.get_cmap('magma', 256)

def _colorize(chw_or_hw, normalize=True):
    """
    Accepts HxW or 1xHxW arrays/tensors; returns HxWx3 uint8 colored with 'magma'.
    """
    x = _to_np(chw_or_hw)
    if x.ndim == 3 and x.shape[0] == 1:  # 1xHxW
        x = x[0]
    assert x.ndim == 2, "Expected single-channel HxW or 1xHxW."
    v = x.astype(np.float32)
    if normalize:
        vmin, vmax = float(v.min()), float(v.max())
        denom = (vmax - vmin) if vmax > vmin else 1.0
        v = (v - vmin) / denom
    rgb = _CMAP(v)[..., :3]   # [H,W,3] in 0..1
    rgb = (rgb * 255.0).clip(0, 255).astype(np.uint8)
    return rgb

def _wandb_log_images(step, prefix, tgt_img, disp=None, depth=None, ref_imgs=None, max_images=2):
    """
    Logs:
      - target input images (RGB)
      - predicted disparity (colorized 'magma')
      - predicted depth (normalized grayscale via tensor2array, forced to HWC)
      - up to 2 reference images
    """
    if not wandb.run:
        return

    B = tgt_img.shape[0]
    k = min(max_images, B)

    # Target inputs (tensors are Bx3xH×W)
    for j in range(k):
        # prefer raw tensor for inputs (avoid tensor2array since it returned CHW)
        wandb.log({f"{prefix}/input/{j}": wandb.Image(_ensure_hwc_uint8(tgt_img[j]))}, step=step)

    # Disparity (expects Bx1xHxW or HxW) -> colorize to RGB
    if disp is not None:
        d = disp
        if isinstance(d, torch.Tensor):
            d = d.detach().cpu()
        if d.ndim == 4:  # Bx1xHxW
            for j in range(k):
                wandb.log({f"{prefix}/disp/{j}": wandb.Image(_colorize(d[j, 0]))}, step=step)
        elif d.ndim == 2:  # HxW
            wandb.log({f"{prefix}/disp/0": wandb.Image(_colorize(d))}, step=step)

    # Depth (use tensor2array for normalization, then force HWC)
    if depth is not None:
        dep = depth
        if isinstance(dep, torch.Tensor):
            dep = dep.detach().cpu()
        if dep.ndim == 4:  # Bx1xHxW
            for j in range(k):
                dimg = tensor2array(dep[j], max_value=10)  # may be CHW
                wandb.log({f"{prefix}/depth/{j}": wandb.Image(_ensure_hwc_uint8(dimg))}, step=step)
        elif dep.ndim == 3:  # BxHxW
            for j in range(k):
                dimg = tensor2array(dep[j], max_value=10)
                wandb.log({f"{prefix}/depth/{j}": wandb.Image(_ensure_hwc_uint8(dimg))}, step=step)
        elif dep.ndim == 2:  # HxW
            dimg = tensor2array(dep, max_value=10)
            wandb.log({f"{prefix}/depth/0": wandb.Image(_ensure_hwc_uint8(dimg))}, step=step)

    # Reference frames (list of tensors each Bx3xHxW)
    if ref_imgs:
        for ridx, r in enumerate(ref_imgs[:2]):      # at most 2 neighbor frames
            for j in range(k):
                wandb.log({f"{prefix}/ref{ridx}/{j}": wandb.Image(_ensure_hwc_uint8(r[j]))}, step=step)

def _ensure_hwc_uint8(x):
    """
    Accepts torch.Tensor or np.ndarray in [C,H,W] or [H,W,C] or [H,W].
    Returns HxWxC uint8 (C=1 or 3). If single-channel, keeps 1 channel.
    Assumes input is either 0..1 or 0..255; clips and converts to uint8.
    """
    a = x
    if isinstance(a, torch.Tensor):
        a = a.detach().cpu().numpy()
    a = np.asarray(a)

    if a.ndim == 3 and a.shape[0] in (1, 3):  # CHW -> HWC
        a = np.transpose(a, (1, 2, 0))
    elif a.ndim == 2:  # HW -> HW1
        a = a[..., None]

    # scale/clamp to 0..255 then uint8
    a = a.astype(np.float32)
    # Heuristic: if max<=1.5 treat as 0..1
    if a.max() <= 1.5:
        a = a * 255.0
    a = np.clip(a, 0, 255).astype(np.uint8)
    return a

# --------------------------------------------------------------------------------------


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

# ---- W&B args: NEW (optional; logging only) ----
parser.add_argument('--wandb', action='store_true', help='enable Weights & Biases logging')
parser.add_argument('--wandb-project', type=str, default='endosfmlearner', help='W&B project name')
parser.add_argument('--wandb-entity', type=str, default=None, help='W&B entity (team/user)')
parser.add_argument('--wandb-run-name', type=str, default=None, help='W&B run name')
parser.add_argument('--wandb-log-every', type=int, default=100, help='log images every N steps')
parser.add_argument('--wandb-max-images', type=int, default=2, help='how many samples from the batch to log')


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

    # ---- W&B init (optional) ----
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or args.name,
            config={
                "resnet_layers": args.resnet_layers,
                "num_scales": args.num_scales,
                "photo_loss_weight": args.photo_loss_weight,
                "smooth_loss_weight": args.smooth_loss_weight,
                "geometry_consistency_weight": args.geometry_consistency_weight,
                "with_ssim": args.with_ssim,
                "with_mask": args.with_mask,
                "with_auto_mask": args.with_auto_mask,
                "dataset": args.dataset,
                "batch_size": args.batch_size,
                "lr": args.lr,
            }
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path/'valid'/str(i)))

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

        # evaluate on validation set
        logger.reset_valid_bar()
        if args.with_gt:
            errors, error_names = validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers)
        else:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, output_writers)
        error_string = ', '.join('{} : {:.3f}'.format(name, error) for name, error in zip(error_names, errors))
        logger.valid_writer.write(' * Avg {}'.format(error_string))

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)
            if args.wandb:
                wandb.log({f"val/{name}": float(error), "epoch": epoch}, step=epoch)

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

    if args.wandb:
        wandb.finish()


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

            # ---- W&B scalars ----
            if args.wandb:
                wandb.log({
                    "train/photometric_error": loss_1.item(),
                    "train/disparity_smoothness_loss": loss_2.item(),
                    "train/geometry_consistency_loss": loss_3.item(),
                    "train/total_loss": loss.item(),
                }, step=n_iter)

                # ---- W&B images every N steps ----
                if (n_iter % args.wandb_log_every) == 0:
                    depth0 = tgt_depth[0]                 # Bx1xHxW
                    disp0 = 1.0 / (depth0 + 1e-9)         # Bx1xHxW

                    _wandb_log_images(
                        step=n_iter,
                        prefix="train",
                        tgt_img=tgt_img,
                        disp=disp0,
                        depth=depth0,
                        ref_imgs=ref_imgs,
                        max_images=args.wandb_max_images
                    )

        # record loss
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
        tgt_depth = [1 / disp_net(tgt_img)]  # list of tensors; use index 0
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

            # ---- W&B images (one batch is enough) ----
            if args.wandb and i == 0:
                _wandb_log_images(
                    step=epoch,
                    prefix="val/without_gt",
                    tgt_img=tgt_img,
                    disp=1.0 / (tgt_depth[0] + 1e-9),  # Bx1xHxW
                    depth=tgt_depth[0],                # Bx1xHxW
                    ref_imgs=ref_imgs,
                    max_images=args.wandb_max_images
                )

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

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i+1)
        if i % args.print_freq == 0:
            logger.valid_writer.write('valid: Time {} Loss {}'.format(batch_time, losses))

    logger.valid_bar.update(len(val_loader))

    # Log averaged validation scalars to W&B
    if args.wandb:
        avg_vals = losses.avg  # ['Total loss','Photo loss','Smooth loss','Consistency loss']
        wandb.log({
            "val_no_gt/total_loss": float(avg_vals[0]),
            "val_no_gt/photo": float(avg_vals[1]),
            "val_no_gt/smooth": float(avg_vals[2]),
            "val_no_gt/consistency": float(avg_vals[3]),
            "epoch": epoch
        }, step=epoch)

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
        output_disp = disp_net(tgt_img)           # Bx1xHxW
        output_depth = 1/output_disp[:, 0]        # BxHxW

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

            # ---- W&B images (one batch) ----
            if args.wandb and i == 0:
                depth_b1hw = output_depth.unsqueeze(1)  # Bx1xHxW for uniform handling
                _wandb_log_images(
                    step=epoch,
                    prefix="val/with_gt",
                    tgt_img=tgt_img,
                    disp=output_disp,       # Bx1xHxW
                    depth=depth_b1hw,       # Bx1xHxW
                    ref_imgs=None,
                    max_images=args.wandb_max_images
                )

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

    # Log averaged GT metrics to W&B
    if args.wandb:
        for v, name in zip(errors.avg, error_names):
            wandb.log({f"val/{name}": float(v), "epoch": epoch}, step=epoch)

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
