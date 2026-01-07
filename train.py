import argparse
import time
import csv
import datetime
from path import Path

import os
import cv2
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import matplotlib.cm as cm
import cv2


import models

import custom_transforms
from utils import tensor2array, save_checkpoint
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_errors
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
import datetime as dt

import wandb


# =========================================================
# W&B helpers
# =========================================================
def tensor_to_rgb(img_tensor, mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)):
    img = img_tensor.detach().cpu()
    if img.dim() == 4:
        img = img[0]
    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)
    img = img * std_t + mean_t
    img = img.clamp(0, 1)
    return (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def tensor_to_colormap(disp_or_depth_tensor, cmap="plasma"):
    x = disp_or_depth_tensor.detach().cpu()
    if x.dim() == 4:
        x = x[0, 0]
    elif x.dim() == 3:
        x = x[0]
    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    colored = cm.get_cmap(cmap)(x_norm.numpy())[..., :3]
    return (colored * 255).astype(np.uint8)


# =========================================================
# Split + Dataset (Hamlyn/EndoVis)
# =========================================================
def _readlines(p):
    with open(p, "r") as f:
        return [l.strip() for l in f.readlines() if l.strip()]


def _load_image_any_ext(path_no_ext, exts=(".jpg", ".png", ".jpeg")):
    for e in exts:
        p = path_no_ext + e
        if os.path.isfile(p):
            return p
    return None


def _default_intrinsics(w, h):
    fx = 0.58 * w
    fy = 0.58 * w
    cx = 0.5 * w
    cy = 0.5 * h
    return np.array([[fx, 0, cx],
                     [0, fy, cy],
                     [0,  0,  1]], dtype=np.float32)


def _load_intrinsics_txt(path):
    if path is None:
        return None
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Intrinsics file not found: {path}")
    if path.endswith(".npy"):
        K = np.load(path).astype(np.float32)
        assert K.shape == (3, 3)
        return K
    vals = []
    with open(path, "r") as f:
        for line in f:
            for x in line.strip().split():
                vals.append(float(x))
    if len(vals) != 9:
        raise ValueError(f"Expected 9 numbers for 3x3 intrinsics, got {len(vals)} in {path}")
    return np.array(vals, dtype=np.float32).reshape(3, 3)


import os
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset

class SplitSequenceFolder(Dataset):
    """
    Dataset basado en un split file (filenames: lista de líneas).
    Soporta múltiples datasets, incluyendo Hamlyn monocular izquierda.

    Firma (según tu repo):
      (self, data_root, filenames, transform, sequence_length=3,
       intrinsics_K=None, max_seek=50, dataset='kitti')
    """

    def __init__(self, data_root, filenames, transform,
                 sequence_length=3, intrinsics_K=None, max_seek=50, dataset='kitti'):
        super().__init__()

        self.data_root = data_root
        self.filenames = filenames
        self.transform = transform
        self.sequence_length = int(sequence_length)
        self.intrinsics_K = intrinsics_K
        self.max_seek = int(max_seek)
        self.dataset = dataset

        assert self.sequence_length >= 2, "sequence_length debe ser >= 2"

        # Carga paths de imágenes (centros). En entrenamiento se construyen secuencias alrededor del centro.
        self.img_paths = []
        self.bad_lines = 0

        for line in self.filenames:
            line = line.strip()
            if not line:
                self.img_paths.append(None)
                self.bad_lines += 1
                continue

            if self.dataset == "hamlyn":
                img_path = self._resolve_hamlyn_left(line)
            else:
                img_path = self._resolve_default(line)

            if img_path is None:
                self.bad_lines += 1

            self.img_paths.append(img_path)

        if self.bad_lines > 0:
            print(
                f"[SplitSequenceFolder] WARNING: {self.bad_lines}/{len(self.img_paths)} lines did not resolve to an image path.\n"
                f"  This often means your split format/path pattern does not match your dataset folder structure."
            )

        # Para evitar que __getitem__ se quede sin salida si el split está roto
        if self.bad_lines == len(self.img_paths):
            raise RuntimeError(
                f"[SplitSequenceFolder] All split lines are invalid for dataset='{self.dataset}'. "
                f"Fix split format or resolver."
            )

    # ---------------------- PATH RESOLVERS ----------------------

    def _resolve_hamlyn_left(self, line):
        """
        Split line real (tu caso):
        rectified08/rectified08/image01 1 l

        Dataset:
        {data_root}/rectified08/rectified08/image01/0000000001.jpg

        Monocular izquierda: ya viene image01, ignoramos side.
        """
        parts = line.split()
        if len(parts) < 2:
            return None

        rel_dir = parts[0]          # e.g. rectified08/rectified08/image01
        frame_id = parts[1]         # e.g. 1

        frame_id = os.path.splitext(frame_id)[0]
        if frame_id.isdigit():
            frame_id = frame_id.zfill(10)
        else:
            # si viene raro, igual intenta padding
            frame_id = frame_id.zfill(10)

        # probar .jpg primero
        img_path = os.path.join(self.data_root, rel_dir, f"{frame_id}.jpg")
        if os.path.isfile(img_path):
            return img_path

        # fallback .png
        img_path = os.path.join(self.data_root, rel_dir, f"{frame_id}.png")
        if os.path.isfile(img_path):
            return img_path

        return None

    def _resolve_default(self, line):
        """
        Resolver genérico (kitti/endovis/etc).
        Mantén aquí tu lógica previa si ya tenías algo específico.
        """
        parts = line.split()

        # Caso 3 tokens estilo monodepth2 kitti: folder frame side
        if len(parts) >= 3:
            folder, frame_id, side = parts[0], parts[1], parts[2]
            frame_id = os.path.splitext(frame_id)[0]
            # kitti suele ser 10 dígitos también
            if frame_id.isdigit():
                frame_id = frame_id.zfill(10)

            # Ejemplo genérico:
            # data_root/folder/image_0{2 or 3}/data/0000000000.jpg
            # (si tu repo tiene una lógica distinta, reemplázala aquí)
            cam = "2" if side in ["l", "left", "0"] else "3"
            cand = os.path.join(self.data_root, folder, f"image_0{cam}", "data", f"{frame_id}.jpg")
            if os.path.isfile(cand):
                return cand
            cand = os.path.join(self.data_root, folder, f"image_0{cam}", "data", f"{frame_id}.png")
            if os.path.isfile(cand):
                return cand

        # Caso 1 token: ruta relativa o absoluta
        if len(parts) == 1:
            rel = parts[0]
            if os.path.isabs(rel) and os.path.isfile(rel):
                return rel
            cand = os.path.join(self.data_root, rel)
            if os.path.isfile(cand):
                return cand
            base = os.path.splitext(cand)[0]
            for e in [".jpg", ".png", ".jpeg"]:
                if os.path.isfile(base + e):
                    return base + e

        return None

    # ---------------------- SAMPLING HELPERS ----------------------

    def _find_valid_center_index(self, idx):
        """
        Busca un índice con img_path válido cerca de idx.
        NO recursivo para evitar RecursionError.
        """
        n = len(self.img_paths)
        idx = int(idx)
        if idx < 0:
            idx = 0
        if idx >= n:
            idx = n - 1

        if self.img_paths[idx] is not None:
            return idx

        # intenta hacia adelante/atrás hasta max_seek
        for k in range(1, self.max_seek + 1):
            j1 = idx + k
            if j1 < n and self.img_paths[j1] is not None:
                return j1
            j2 = idx - k
            if j2 >= 0 and self.img_paths[j2] is not None:
                return j2

        raise RuntimeError(
            f"[SplitSequenceFolder] Could not find a valid sequence sample near idx={idx}. "
            f"Check split paths and dataset structure. bad_lines={self.bad_lines}/{len(self.img_paths)}."
        )

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        """
        Debe devolver:
          tgt_img, ref_imgs, intrinsics, intrinsics_inv
        según lo que espera tu loop de train():
          for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(train_loader):
        """
        idx = self._find_valid_center_index(idx)

        # centro
        tgt_path = self.img_paths[idx]

        # refs: alrededor del centro (monocular: solo temporal)
        half = self.sequence_length // 2
        # Ejemplo: seq_len=3 => offsets [-1, +1]
        offsets = [o for o in range(-half, half + 1) if o != 0]
        # Si seq_len es par, esto deja un ref menos; forzamos a seq_len-1 refs:
        if len(offsets) != (self.sequence_length - 1):
            # fallback: usa los primeros seq_len-1 offsets hacia adelante
            offsets = list(range(1, self.sequence_length))

        ref_paths = []
        for o in offsets:
            j = idx + o
            j = max(0, min(len(self.img_paths) - 1, j))
            j = self._find_valid_center_index(j)
            ref_paths.append(self.img_paths[j])

        tgt_img = self._read_image(tgt_path)
        ref_imgs = [self._read_image(p) for p in ref_paths]

        # intrinsics: si te pasan K fijo, úsalo; si no, identity
        if self.intrinsics_K is not None:
            K = np.array(self.intrinsics_K, dtype=np.float32)
        else:
            K = np.eye(3, dtype=np.float32)

        K_inv = np.linalg.inv(K).astype(np.float32)

        # aplica transform (típicamente: ArrayToTensor + Normalize etc.)
        # intrinsics: si te pasan K fijo, úsalo; si no, identity
        # --- intrinsics base ---
        if self.intrinsics_K is not None:
            K = np.array(self.intrinsics_K, dtype=np.float32)
        else:
            K = np.eye(3, dtype=np.float32)

        # --- apply transforms: Compose(images, intrinsics) ---
        if self.transform is not None:
            images = [tgt_img] + ref_imgs
            images, K = self.transform(images, K)
            tgt_img = images[0]
            ref_imgs = images[1:]

        # --- inverse intrinsics ---
        # --- inverse intrinsics ---
        K = np.ascontiguousarray(K.astype(np.float32))
        K_inv = np.ascontiguousarray(np.linalg.inv(K).astype(np.float32))

        # IMPORTANT: return torch tensors to avoid collate storage issues
        K_t = torch.from_numpy(K).float().clone()
        K_inv_t = torch.from_numpy(K_inv).float().clone()

        return tgt_img, ref_imgs, K_t, K_inv_t




    def _read_image(self, path):
        """
        Lee imagen como RGB float32 [0,1].
        """
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            raise RuntimeError(f"[SplitSequenceFolder] Failed to read image: {path}")
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0
        return img


# =========================================================
# Argparse
# =========================================================
parser = argparse.ArgumentParser(
    description="Structure from Motion Learner training (Hamlyn split support)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

parser.add_argument("data", metavar="DIR", help="path to dataset root")
parser.add_argument("--folder-type", type=str, choices=["sequence", "pair"], default="sequence")
parser.add_argument("--sequence-length", type=int, default=3)
parser.add_argument("-j", "--workers", default=4, type=int)
parser.add_argument("--epochs", default=200, type=int)
parser.add_argument("--epoch-size", default=0, type=int)
parser.add_argument("-b", "--batch-size", default=4, type=int)
parser.add_argument("--lr", "--learning-rate", default=1e-4, type=float)
parser.add_argument("--momentum", default=0.9, type=float)
parser.add_argument("--beta", default=0.999, type=float)
parser.add_argument("--weight-decay", "--wd", default=0, type=float)
parser.add_argument("--print-freq", default=10, type=int)
parser.add_argument("--seed", default=0, type=int)

parser.add_argument("--log-summary", default="progress_log_summary.csv")
parser.add_argument("--log-full", default="progress_log_full.csv")
parser.add_argument("--log-output", action="store_true")

parser.add_argument("--resnet-layers", type=int, default=18, choices=[18, 50])
parser.add_argument("--num-scales", type=int, default=1)
parser.add_argument("-p", "--photo-loss-weight", type=float, default=1)
parser.add_argument("-s", "--smooth-loss-weight", type=float, default=0.1)
parser.add_argument("-c", "--geometry-consistency-weight", type=float, default=0.5)

parser.add_argument("--with-ssim", type=int, default=1)
parser.add_argument("--with-mask", type=int, default=1)
parser.add_argument("--with-auto-mask", type=int, default=0)
parser.add_argument("--with-pretrain", type=int, default=1)

parser.add_argument("--dataset", type=str, choices=["kitti", "nyu", "endovis", "hamlyn"], default="kitti")
parser.add_argument("--pretrained-disp", default=None)
parser.add_argument("--pretrained-pose", default=None)
parser.add_argument("--name", type=str, required=True)
parser.add_argument("--padding-mode", type=str, choices=["zeros", "border"], default="zeros")
parser.add_argument("--with-gt", action="store_true")

# ✅ Split options (Monodepth2-style)
parser.add_argument("--split", type=str, default=None)
parser.add_argument("--splits-root", type=str, default=None)
parser.add_argument("--intrinsics_txt", type=str, default=None)

# W&B
parser.add_argument("--wandb", action="store_true")
parser.add_argument("--wandb_project", type=str, default="endosfm-scarED")
parser.add_argument("--wandb_entity", type=str, default=None)
parser.add_argument("--wandb_log_images_every", type=int, default=100)

best_error = -1
n_iter = 0
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
torch.autograd.set_detect_anomaly(True)


# =========================================================
# Core
# =========================================================
def compute_depth(disp_net, tgt_img, ref_imgs):
    out_tgt = disp_net(tgt_img)
    if isinstance(out_tgt, (list, tuple)):
        tgt_depth = [1 / disp for disp in out_tgt]
    else:
        tgt_depth = [1 / out_tgt]

    ref_depths = []
    for ref_img in ref_imgs:
        out_ref = disp_net(ref_img)
        if isinstance(out_ref, (list, tuple)):
            ref_depth = [1 / disp for disp in out_ref]
        else:
            ref_depth = [1 / out_ref]
        ref_depths.append(ref_depth)

    return tgt_depth, ref_depths


def compute_pose_with_inv(pose_net, tgt_img, ref_imgs):
    poses = []
    poses_inv = []
    for ref_img in ref_imgs:
        poses.append(pose_net(tgt_img, ref_img))
        poses_inv.append(pose_net(ref_img, tgt_img))
    return poses, poses_inv


def train(args, train_loader, disp_net, pose_net, optimizer, epoch_size, logger, train_writer):
    global n_iter, device
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter(precision=4)
    w1, w2, w3 = args.photo_loss_weight, args.smooth_loss_weight, args.geometry_consistency_weight

    disp_net.train()
    pose_net.train()

    end = time.time()
    logger.train_bar.update(0)

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(train_loader):
        log_losses = i > 0 and n_iter % args.print_freq == 0

        data_time.update(time.time() - end)

        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)

        tgt_depth, ref_depths = compute_depth(disp_net, tgt_img, ref_imgs)
        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(
            tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
            poses, poses_inv, args.num_scales, args.with_ssim,
            args.with_mask, args.with_auto_mask, args.padding_mode
        )

        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)
        loss = w1 * loss_1 + w2 * loss_2 + w3 * loss_3

        if log_losses:
            train_writer.add_scalar("photometric_error", loss_1.item(), n_iter)
            train_writer.add_scalar("disparity_smoothness_loss", loss_2.item(), n_iter)
            train_writer.add_scalar("geometry_consistency_loss", loss_3.item(), n_iter)
            train_writer.add_scalar("total_loss", loss.item(), n_iter)

        losses.update(loss.item(), args.batch_size)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if args.wandb and (n_iter % args.wandb_log_images_every == 0):
            wandb.log({
                "train/input": wandb.Image(tensor_to_rgb(tgt_img)),
                "train/pred_depth": wandb.Image(tensor_to_colormap(tgt_depth[0][0])),
            }, step=n_iter)

        batch_time.update(time.time() - end)
        end = time.time()

        with open(args.save_path / args.log_full, "a") as csvfile:
            writer = csv.writer(csvfile, delimiter="\t")
            writer.writerow([loss.item(), loss_1.item(), loss_2.item(), loss_3.item()])

        logger.train_bar.update(i + 1)
        if i % args.print_freq == 0:
            logger.train_writer.write(f"Train: Time {batch_time} Data {data_time} Loss {losses}")

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

    disp_net.eval()
    pose_net.eval()

    end = time.time()
    logger.valid_bar.update(0)

    for i, (tgt_img, ref_imgs, intrinsics, intrinsics_inv) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        ref_imgs = [img.to(device) for img in ref_imgs]
        intrinsics = intrinsics.to(device)
        intrinsics_inv = intrinsics_inv.to(device)

        tgt_depth = [1 / disp_net(tgt_img)]
        ref_depths = []
        for ref_img in ref_imgs:
            ref_depths.append([1 / disp_net(ref_img)])

        poses, poses_inv = compute_pose_with_inv(pose_net, tgt_img, ref_imgs)

        loss_1, loss_3 = compute_photo_and_geometry_loss(
            tgt_img, ref_imgs, intrinsics, tgt_depth, ref_depths,
            poses, poses_inv, args.num_scales, args.with_ssim,
            args.with_mask, False, args.padding_mode
        )
        loss_2 = compute_smooth_loss(tgt_depth, tgt_img, ref_depths, ref_imgs)

        loss_1 = loss_1.item()
        loss_2 = loss_2.item()
        loss_3 = loss_3.item()

        loss = loss_1
        losses.update([loss, loss_1, loss_2, loss_3])

        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i + 1)
        if i % args.print_freq == 0:
            logger.valid_writer.write(f"valid: Time {batch_time} Loss {losses}")

    logger.valid_bar.update(len(val_loader))

    if args.wandb:
        wandb.log({
            "epoch": epoch,
            "val/total_loss": losses.avg[0],
            "val/photo_loss": losses.avg[1],
            "val/smooth_loss": losses.avg[2],
            "val/consistency_loss": losses.avg[3],
        }, step=epoch)

    return losses.avg, ["Total loss", "Photo loss", "Smooth loss", "Consistency loss"]


@torch.no_grad()
def validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers=[]):
    global device
    batch_time = AverageMeter()
    error_names = ["abs_diff", "abs_rel", "sq_rel", "a1", "a2", "a3"]
    errors = AverageMeter(i=len(error_names))

    disp_net.eval()

    end = time.time()
    logger.valid_bar.update(0)

    for i, (tgt_img, depth) in enumerate(val_loader):
        tgt_img = tgt_img.to(device)
        depth = depth.to(device)

        if depth.nelement() == 0:
            continue

        output_disp = disp_net(tgt_img)
        output_depth = 1 / output_disp[:, 0]

        if depth.nelement() != output_depth.nelement():
            b, h, w = depth.size()
            output_depth = torch.nn.functional.interpolate(output_depth.unsqueeze(1), [h, w]).squeeze(1)

        errors.update(compute_errors(depth, output_depth, args.dataset))

        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i + 1)
        if i % args.print_freq == 0:
            logger.valid_writer.write(
                f"valid: Time {batch_time} Abs Error {errors.val[0]:.4f} ({errors.avg[0]:.4f})"
            )

    logger.valid_bar.update(len(val_loader))

    if args.wandb:
        wandb.log({f"val_gt/{n}": float(v) for n, v in zip(error_names, errors.avg)}, step=epoch)
        wandb.log({"epoch": epoch}, step=epoch)

    return errors.avg, error_names

def safe_collate(batch):
    """
    batch: list of (tgt_img, ref_imgs, K, K_inv)
      - tgt_img: Tensor [C,H,W]
      - ref_imgs: list length Nref of Tensor [C,H,W]
      - K: Tensor [3,3]
      - K_inv: Tensor [3,3]
    Returns:
      - tgt: Tensor [B,C,H,W]
      - refs: list length Nref of Tensor [B,C,H,W]
      - K: Tensor [B,3,3]
      - K_inv: Tensor [B,3,3]
    """
    tgt_imgs, ref_imgs_list, Ks, K_invs = zip(*batch)

    # tgt
    tgt = torch.stack([t.contiguous().clone() for t in tgt_imgs], dim=0)

    # refs (list of lists -> list of batched tensors)
    nref = len(ref_imgs_list[0])
    refs = []
    for r in range(nref):
        refs.append(torch.stack([ref_imgs_list[b][r].contiguous().clone() for b in range(len(batch))], dim=0))

    # intrinsics
    K = torch.stack([k.contiguous().clone() for k in Ks], dim=0)
    K_inv = torch.stack([k.contiguous().clone() for k in K_invs], dim=0)

    return tgt, refs, K, K_inv

def main():
    global best_error, n_iter, device
    args = parser.parse_args()

    timestamp = datetime.datetime.now().strftime("%m-%d-%H:%M")
    save_path = Path(args.name)
    args.save_path = "checkpoints" / save_path / timestamp
    print("=> will save everything to {}".format(args.save_path))
    args.save_path.makedirs_p()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cudnn.deterministic = True
    cudnn.benchmark = True

    training_writer = SummaryWriter(args.save_path)
    output_writers = []
    if args.log_output:
        for i in range(3):
            output_writers.append(SummaryWriter(args.save_path / "valid" / str(i)))

    # W&B init (solo si --wandb)
    if args.wandb:
        run_name = f"{args.name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_kwargs = dict(project=args.wandb_project, name=run_name, config=vars(args))
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        wandb.init(**wandb_kwargs)

    normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])

    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.Resize((288, 512)),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([
        custom_transforms.Resize((288, 512)),
        custom_transforms.ArrayToTensor(),
        normalize
    ])


    print("=> fetching data in '{}'".format(args.data))

    use_split_mode = (args.dataset in ["endovis", "hamlyn"]) and (args.split is not None)

    if use_split_mode:
        splits_root = args.splits_root or os.path.join(os.path.dirname(__file__), "splits")
        split_dir = os.path.join(splits_root, args.split)

        train_list_path = os.path.join(split_dir, "train_files.txt")
        val_list_path = os.path.join(split_dir, "val_files.txt")

        if not os.path.isfile(train_list_path) or not os.path.isfile(val_list_path):
            raise FileNotFoundError(
                f"Expected:\n  {train_list_path}\n  {val_list_path}\n"
                f"Check --splits-root and --split"
            )

        train_filenames = _readlines(train_list_path)
        val_filenames = _readlines(val_list_path)

        K_fixed = _load_intrinsics_txt(args.intrinsics_txt) if args.intrinsics_txt else None

        train_set = SplitSequenceFolder(args.data, train_filenames, train_transform,
                                        sequence_length=args.sequence_length, intrinsics_K=K_fixed,
                                        dataset=args.dataset)
        val_set = SplitSequenceFolder(args.data, val_filenames, valid_transform,
                                      sequence_length=args.sequence_length, intrinsics_K=K_fixed,
                                      dataset=args.dataset)
    else:
        # Modo original (si lo necesitas)
        from datasets.sequence_folders import SequenceFolder
        from datasets.pair_folders import PairFolder

        if args.folder_type == "sequence":
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

        if args.with_gt:
            from datasets.validation_folders import ValidationSet
            val_set = ValidationSet(args.data, transform=valid_transform, dataset=args.dataset)
        else:
            val_set = SequenceFolder(
                args.data,
                transform=valid_transform,
                seed=args.seed,
                train=False,
                sequence_length=args.sequence_length,
                dataset=args.dataset
            )

    print(f"{len(train_set)} samples found in train set")
    print(f"{len(val_set)} samples found in valid set")

    train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True,
    num_workers=args.workers, pin_memory=True, drop_last=True,
    collate_fn=safe_collate
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False,
        collate_fn=safe_collate
    )


    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    print("=> creating model")
    disp_net = models.DispResNet(args.resnet_layers, args.with_pretrain).to(device)
    pose_net = models.PoseResNet(18, args.with_pretrain).to(device)

    if args.pretrained_disp:
        print("=> using pre-trained weights for DispResNet")
        weights = torch.load(args.pretrained_disp, map_location=device)
        disp_net.load_state_dict(weights["state_dict"], strict=False)

    if args.pretrained_pose:
        print("=> using pre-trained weights for PoseResNet")
        weights = torch.load(args.pretrained_pose, map_location=device)
        pose_net.load_state_dict(weights["state_dict"], strict=False)

    disp_net = torch.nn.DataParallel(disp_net)
    pose_net = torch.nn.DataParallel(pose_net)

    if args.wandb:
        wandb.watch(disp_net, log="gradients", log_freq=200)
        wandb.watch(pose_net, log="gradients", log_freq=200)

    print("=> setting adam solver")
    optim_params = [
        {"params": disp_net.parameters(), "lr": args.lr},
        {"params": pose_net.parameters(), "lr": args.lr}
    ]
    optimizer = torch.optim.Adam(optim_params, betas=(args.momentum, args.beta),
                                 weight_decay=args.weight_decay)

    with open(args.save_path / args.log_summary, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(["train_loss", "validation_loss"])

    with open(args.save_path / args.log_full, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(["train_loss", "photo_loss", "smooth_loss", "geometry_consistency_loss"])

    logger = TermLogger(n_epochs=args.epochs,
                        train_size=min(len(train_loader), args.epoch_size),
                        valid_size=len(val_loader))
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        logger.reset_train_bar()
        train_loss = train(args, train_loader, disp_net, pose_net, optimizer, args.epoch_size, logger, training_writer)
        logger.train_writer.write(" * Avg Loss : {:.3f}".format(train_loss))

        logger.reset_valid_bar()
        if args.with_gt:
            errors, error_names = validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers)
        else:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, output_writers)

        error_string = ", ".join("{} : {:.3f}".format(n, e) for n, e in zip(error_names, errors))
        logger.valid_writer.write(" * Avg {}".format(error_string))

        for e, n in zip(errors, error_names):
            training_writer.add_scalar(n, e, epoch)

        decisive_error = errors[1]  # abs_rel
        global best_error
        if best_error < 0:
            best_error = decisive_error

        is_best = decisive_error < best_error
        best_error = min(best_error, decisive_error)

        save_checkpoint(
            args.save_path,
            {"epoch": epoch + 1, "state_dict": disp_net.module.state_dict()},
            {"epoch": epoch + 1, "state_dict": pose_net.module.state_dict()},
            is_best
        )

        with open(args.save_path / args.log_summary, "a") as csvfile:
            writer = csv.writer(csvfile, delimiter="\t")
            writer.writerow([train_loss, decisive_error])

    logger.epoch_bar.finish()

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
