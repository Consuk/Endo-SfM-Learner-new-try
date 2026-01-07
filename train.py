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

import models

import custom_transforms
from utils import tensor2array, save_checkpoint
from loss_functions import compute_smooth_loss, compute_photo_and_geometry_loss, compute_errors
from logger import TermLogger, AverageMeter
from tensorboardX import SummaryWriter
import datetime as dt

# ---- W&B ----
import wandb


# =========================================================
# Utils (W&B image helpers)
# =========================================================
def tensor_to_rgb(img_tensor, mean=(0.45, 0.45, 0.45), std=(0.225, 0.225, 0.225)):
    """
    Tensor normalizado -> imagen RGB uint8 (H,W,3).
    Acepta:
      - (B,C,H,W) -> toma la primera
      - (C,H,W)
    """
    img = img_tensor.detach().cpu()
    if img.dim() == 4:
        img = img[0]

    mean_t = torch.tensor(mean).view(3, 1, 1)
    std_t = torch.tensor(std).view(3, 1, 1)

    img = img * std_t + mean_t
    img = img.clamp(0, 1)
    np_img = (img.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
    return np_img


def tensor_to_colormap(disp_or_depth_tensor, cmap="plasma"):
    """
    Convierte (disp o depth) a imagen coloreada uint8 (H,W,3).
    Acepta:
      - (B,1,H,W) -> toma la primera
      - (1,H,W) o (C,H,W) -> toma canal 0
      - (H,W)
    """
    x = disp_or_depth_tensor.detach().cpu()

    if x.dim() == 4:
        x = x[0, 0]
    elif x.dim() == 3:
        x = x[0]

    x_norm = (x - x.min()) / (x.max() - x.min() + 1e-8)
    x_np = x_norm.numpy()
    colored = cm.get_cmap(cmap)(x_np)[..., :3]
    return (colored * 255).astype(np.uint8)


# =========================================================
# Hamlyn / EndoVis split-based dataset (Monodepth2-style)
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
    """
    Intrinsics “razonables” si no te pasan nada.
    OJO: para métricas reales, pásalas con --intrinsics_txt o integra las reales de Hamlyn.
    """
    fx = 0.58 * w
    fy = 0.58 * w
    cx = 0.5 * w
    cy = 0.5 * h
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float32)
    return K


def _load_intrinsics_txt(path):
    """
    Espera 3x3 (9 floats) en un txt o npy.
    """
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
    K = np.array(vals, dtype=np.float32).reshape(3, 3)
    return K


class SplitSequenceFolder(torch.utils.data.Dataset):
    """
    Dataset para entrenar/validar usando splits tipo Monodepth2:
      splits/<split>/train_files.txt
      splits/<split>/val_files.txt

    Soporta dos formatos comunes por línea:
      A) "subdir frame_id side"  (Monodepth2)
         -> busca imagen: <data_root>/<subdir>/data/<frame_id>.{jpg/png}
      B) "relative/path/to/image.{jpg/png}"
         -> usa ese path relativo a data_root

    Genera:
      tgt_img, ref_imgs(list), intrinsics, intrinsics_inv
    Donde ref_imgs = [prev, next] para sequence_length=3 (default).
    """
    def __init__(self, data_root, filenames, transform, sequence_length=3, intrinsics_K=None):
        super().__init__()
        assert sequence_length == 3, "Este wrapper asume sequence_length=3 (prev/tgt/next)"
        self.data_root = data_root
        self.filenames = filenames
        self.transform = transform
        self.sequence_length = sequence_length
        self.K_fixed = intrinsics_K  # 3x3 o None

        # Pre-resolver paths a imágenes para que idx±1 funcione por orden del split
        self.img_paths = []
        for line in self.filenames:
            parts = line.split()
            img_path = None

            if len(parts) == 3:
                subdir, frame_id, _side = parts
                base = os.path.join(self.data_root, subdir, "data", frame_id)
                found = _load_image_any_ext(base)
                if found is not None:
                    img_path = found
            else:
                # path relativo (puede incluir extensión)
                cand = os.path.join(self.data_root, parts[0])
                if os.path.isfile(cand):
                    img_path = cand
                else:
                    # si viene sin extensión
                    base = os.path.splitext(cand)[0]
                    found = _load_image_any_ext(base)
                    if found is not None:
                        img_path = found

            if img_path is None:
                # lo dejamos como None, y lo skippeamos en __getitem__
                self.img_paths.append(None)
            else:
                self.img_paths.append(img_path)

    def __len__(self):
        return len(self.img_paths)

    def _load_rgb(self, p):
        bgr = cv2.imread(p)
        if bgr is None:
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return rgb

    def __getitem__(self, idx):
        # Necesitamos prev y next
        if idx - 1 < 0 or idx + 1 >= len(self.img_paths):
            # sample inválido para secuencia
            return self.__getitem__(max(1, min(len(self.img_paths) - 2, idx)))

        p_prev = self.img_paths[idx - 1]
        p_tgt = self.img_paths[idx]
        p_next = self.img_paths[idx + 1]

        # Si alguno es None, busca cercano
        if p_tgt is None:
            # fallback: mueve idx
            new_idx = max(1, min(len(self.img_paths) - 2, idx + 1))
            return self.__getitem__(new_idx)
        if p_prev is None or p_next is None:
            new_idx = max(1, min(len(self.img_paths) - 2, idx + 1))
            return self.__getitem__(new_idx)

        prev_rgb = self._load_rgb(p_prev)
        tgt_rgb = self._load_rgb(p_tgt)
        next_rgb = self._load_rgb(p_next)

        if prev_rgb is None or tgt_rgb is None or next_rgb is None:
            new_idx = max(1, min(len(self.img_paths) - 2, idx + 1))
            return self.__getitem__(new_idx)

        h, w = tgt_rgb.shape[:2]

        # intrinsics
        K = self.K_fixed if self.K_fixed is not None else _default_intrinsics(w, h)
        K_inv = np.linalg.inv(K).astype(np.float32)

        # transforms (tu pipeline espera ArrayToTensor() + Normalize())
        # Train transform incluye RandomScaleCrop y RandomHorizontalFlip, pero esos
        # asumen inputs consistentes: por eso aplicamos la MISMA transform a las 3.
        # Para mantenerlo simple y consistente, aplicamos transform por separado
        # (si tu RandomScaleCrop requiere sincronía, entonces hay que implementarla sincronizada).
        # En muchos forks, RandomScaleCrop está implementado para listas, pero aquí no lo asumimos.
        tgt = self.transform(tgt_rgb)
        prev = self.transform(prev_rgb)
        nxt = self.transform(next_rgb)

        # intrinsics como tensor
        intrinsics = torch.from_numpy(K)
        intrinsics_inv = torch.from_numpy(K_inv)

        ref_imgs = [prev, nxt]
        return tgt, ref_imgs, intrinsics, intrinsics_inv


# =========================================================
# Argparse
# =========================================================
parser = argparse.ArgumentParser(
    description="Structure from Motion Learner training (with Hamlyn split support, Monodepth2-style)",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument("data", metavar="DIR", help="path to dataset root")
parser.add_argument("--folder-type", type=str, choices=["sequence", "pair"], default="sequence",
                    help="dataset type to train (sequence uses tgt+refs)")
parser.add_argument("--sequence-length", type=int, metavar="N", default=3,
                    help="sequence length for training (this script supports 3 for split-based)")
parser.add_argument("-j", "--workers", default=4, type=int, metavar="N")
parser.add_argument("--epochs", default=200, type=int, metavar="N")
parser.add_argument("--epoch-size", default=0, type=int, metavar="N",
                    help="manual epoch size (will match dataset size if not set)")
parser.add_argument("-b", "--batch-size", default=4, type=int, metavar="N")
parser.add_argument("--lr", "--learning-rate", default=1e-4, type=float, metavar="LR")
parser.add_argument("--momentum", default=0.9, type=float, metavar="M")
parser.add_argument("--beta", default=0.999, type=float, metavar="M")
parser.add_argument("--weight-decay", "--wd", default=0, type=float, metavar="W")
parser.add_argument("--print-freq", default=10, type=int, metavar="N")
parser.add_argument("--seed", default=0, type=int)

parser.add_argument("--log-summary", default="progress_log_summary.csv", metavar="PATH")
parser.add_argument("--log-full", default="progress_log_full.csv", metavar="PATH")
parser.add_argument("--log-output", action="store_true")

parser.add_argument("--resnet-layers", type=int, default=18, choices=[18, 50])
parser.add_argument("--num-scales", "--number-of-scales", type=int, default=1)
parser.add_argument("-p", "--photo-loss-weight", type=float, default=1)
parser.add_argument("-s", "--smooth-loss-weight", type=float, default=0.1)
parser.add_argument("-c", "--geometry-consistency-weight", type=float, default=0.5)

parser.add_argument("--with-ssim", type=int, default=1)
parser.add_argument("--with-mask", type=int, default=1)
parser.add_argument("--with-auto-mask", type=int, default=0)
parser.add_argument("--with-pretrain", type=int, default=1)

parser.add_argument("--dataset", type=str, choices=["kitti", "nyu", "endovis", "hamlyn"], default="kitti",
                    help="dataset name (affects losses/errors + split mode)")
parser.add_argument("--pretrained-disp", dest="pretrained_disp", default=None, metavar="PATH")
parser.add_argument("--pretrained-pose", dest="pretrained_pose", default=None, metavar="PATH")

parser.add_argument("--name", dest="name", type=str, required=True,
                    help="experiment name; checkpoints stored under checkpoints/<name>/<timestamp>")

parser.add_argument("--padding-mode", type=str, choices=["zeros", "border"], default="zeros")
parser.add_argument("--with-gt", action="store_true",
                    help="use ground truth for validation (depends on your ValidationSet implementation)")

# ✅ Hamlyn/EndoVis Monodepth2-style splits
parser.add_argument("--split", type=str, default=None,
                    help="split name under <splits_root>/<split> (expects train_files.txt/val_files.txt)")
parser.add_argument("--splits-root", type=str, default=None,
                    help="root containing split folders. Default: <this_script_dir>/splits")
parser.add_argument("--intrinsics_txt", type=str, default=None,
                    help="optional path to 3x3 intrinsics txt/npy (used for split-based datasets)")

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
# Train / Validate
# =========================================================
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
            # tgt_depth es lista por escala; tomamos [0] y visualizamos disp=1/depth
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

        if log_outputs and i < len(output_writers):
            if epoch == 0:
                output_writers[i].add_image("val Input", tensor2array(tgt_img[0]), 0)

            output_writers[i].add_image(
                "val Dispnet Output Normalized",
                tensor2array(1 / tgt_depth[0][0], max_value=None, colormap="magma"),
                epoch
            )
            output_writers[i].add_image(
                "val Depth Output",
                tensor2array(tgt_depth[0][0], max_value=10),
                epoch
            )

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

    if args.wandb and (epoch % max(1, args.wandb_log_images_every) == 0):
        wandb.log({
            "val/input": wandb.Image(tensor_to_rgb(tgt_img)),
            "val/pred_depth": wandb.Image(tensor_to_colormap(tgt_depth[0][0])),
            "val/total_loss": losses.avg[0],
            "val/photo_loss": losses.avg[1],
            "val/smooth_loss": losses.avg[2],
            "val/consistency_loss": losses.avg[3],
            "epoch": epoch
        }, step=epoch)

    return losses.avg, ["Total loss", "Photo loss", "Smooth loss", "Consistency loss"]


@torch.no_grad()
def validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers=[]):
    global device
    batch_time = AverageMeter()
    error_names = ["abs_diff", "abs_rel", "sq_rel", "a1", "a2", "a3"]
    errors = AverageMeter(i=len(error_names))
    log_outputs = len(output_writers) > 0

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

        if log_outputs and i < len(output_writers):
            if epoch == 0:
                output_writers[i].add_image("val Input", tensor2array(tgt_img[0]), 0)
                depth_to_show = depth[0]
                output_writers[i].add_image("val target Depth", tensor2array(depth_to_show, max_value=10), epoch)
                depth_tmp = depth_to_show.clone()
                depth_tmp[depth_tmp == 0] = 1000
                disp_to_show = (1 / depth_tmp).clamp(0, 10)
                output_writers[i].add_image("val target Disparity Normalized",
                                            tensor2array(disp_to_show, max_value=None, colormap="magma"), epoch)

            output_writers[i].add_image("val Dispnet Output Normalized",
                                        tensor2array(output_disp[0], max_value=None, colormap="magma"), epoch)
            output_writers[i].add_image("val Depth Output",
                                        tensor2array(output_depth[0], max_value=10), epoch)

        if depth.nelement() != output_depth.nelement():
            b, h, w = depth.size()
            output_depth = torch.nn.functional.interpolate(
                output_depth.unsqueeze(1), [h, w]
            ).squeeze(1)

        errors.update(compute_errors(depth, output_depth, args.dataset))

        batch_time.update(time.time() - end)
        end = time.time()
        logger.valid_bar.update(i + 1)
        if i % args.print_freq == 0:
            logger.valid_writer.write(
                f"valid: Time {batch_time} Abs Error {errors.val[0]:.4f} ({errors.avg[0]:.4f})"
            )

    logger.valid_bar.update(len(val_loader))

    if args.wandb and (epoch % max(1, args.wandb_log_images_every) == 0):
        wandb.log({f"val_gt/{name}": val for name, val in zip(error_names, errors.avg)}, step=epoch)
        wandb.log({"epoch": epoch}, step=epoch)

    return errors.avg, error_names


def compute_depth(disp_net, tgt_img, ref_imgs):
    # disp_net devuelve lista por escala (en algunos forks) o tensor; lo manejamos robusto
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


# =========================================================
# Main
# =========================================================
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

    # ---------------- W&B init (ONLY if --wandb) ----------------
    if args.wandb:
        run_name = f"{args.name}_{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        wandb_kwargs = dict(project=args.wandb_project, name=run_name, config=vars(args))
        if args.wandb_entity:
            wandb_kwargs["entity"] = args.wandb_entity
        wandb.init(**wandb_kwargs)
    # -----------------------------------------------------------

    # Data transforms (mismo MEAN/STD que tu primer código)
    normalize = custom_transforms.Normalize(mean=[0.45, 0.45, 0.45],
                                            std=[0.225, 0.225, 0.225])

    train_transform = custom_transforms.Compose([
        custom_transforms.RandomHorizontalFlip(),
        custom_transforms.RandomScaleCrop(),
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    valid_transform = custom_transforms.Compose([
        custom_transforms.ArrayToTensor(),
        normalize
    ])

    # ---------------- Data loading ----------------
    print(f"=> fetching data in '{args.data}'")

    use_split_mode = (args.dataset in ["endovis", "hamlyn"]) and (args.split is not None)

    if use_split_mode:
        splits_root = args.splits_root
        if splits_root is None:
            splits_root = os.path.join(os.path.dirname(__file__), "splits")

        split_dir = os.path.join(splits_root, args.split)
        train_list_path = os.path.join(split_dir, "train_files.txt")
        val_list_path = os.path.join(split_dir, "val_files.txt")

        if not os.path.isfile(train_list_path) or not os.path.isfile(val_list_path):
            raise FileNotFoundError(
                f"Expected train/val files at:\n"
                f"  {train_list_path}\n"
                f"  {val_list_path}\n"
                f"Set --splits-root correctly or provide --split."
            )

        train_filenames = _readlines(train_list_path)
        val_filenames = _readlines(val_list_path)

        K_fixed = _load_intrinsics_txt(args.intrinsics_txt) if args.intrinsics_txt else None

        train_set = SplitSequenceFolder(
            data_root=args.data,
            filenames=train_filenames,
            transform=train_transform,
            sequence_length=args.sequence_length,
            intrinsics_K=K_fixed
        )
        val_set = SplitSequenceFolder(
            data_root=args.data,
            filenames=val_filenames,
            transform=valid_transform,
            sequence_length=args.sequence_length,
            intrinsics_K=K_fixed
        )

    else:
        # --- comportamiento original (escaneo por escenas) ---
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

    print(f"{len(train_set)} samples found in train set")
    print(f"{len(val_set)} samples found in valid set")

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True, drop_last=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False
    )

    if args.epoch_size == 0:
        args.epoch_size = len(train_loader)

    # ---------------- Model ----------------
    print("=> creating model")
    disp_net = models.DispResNet(args.resnet_layers, args.with_pretrain).to(device)
    pose_net = models.PoseResNet(18, args.with_pretrain).to(device)

    # load pretrained
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
    optimizer = torch.optim.Adam(
        optim_params,
        betas=(args.momentum, args.beta),
        weight_decay=args.weight_decay
    )

    # logs
    with open(args.save_path / args.log_summary, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(["train_loss", "validation_loss"])

    with open(args.save_path / args.log_full, "w") as csvfile:
        writer = csv.writer(csvfile, delimiter="\t")
        writer.writerow(["train_loss", "photo_loss", "smooth_loss", "geometry_consistency_loss"])

    logger = TermLogger(
        n_epochs=args.epochs,
        train_size=min(len(train_loader), args.epoch_size),
        valid_size=len(val_loader)
    )
    logger.epoch_bar.start()

    for epoch in range(args.epochs):
        logger.epoch_bar.update(epoch)

        logger.reset_train_bar()
        train_loss = train(args, train_loader, disp_net, pose_net, optimizer, args.epoch_size, logger, training_writer)
        logger.train_writer.write(f" * Avg Loss : {train_loss:.3f}")

        if args.wandb:
            wandb.log({"epoch": epoch, "train/avg_total_loss": train_loss}, step=epoch)

        logger.reset_valid_bar()
        if args.with_gt:
            errors, error_names = validate_with_gt(args, val_loader, disp_net, epoch, logger, output_writers)
        else:
            errors, error_names = validate_without_gt(args, val_loader, disp_net, pose_net, epoch, logger, output_writers)

        error_string = ", ".join(f"{name} : {error:.3f}" for name, error in zip(error_names, errors))
        logger.valid_writer.write(f" * Avg {error_string}")

        for error, name in zip(errors, error_names):
            training_writer.add_scalar(name, error, epoch)
            if args.wandb:
                wandb.log({f"val/{name}": float(error), "epoch": epoch}, step=epoch)

        decisive_error = errors[1]  # abs_rel (por convención de este repo)
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
