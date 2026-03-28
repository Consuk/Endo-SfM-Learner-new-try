import os
import cv2
import argparse
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.cm as cm


def _normalize_side(side):
    s = str(side).lower()
    if s in ["l", "left", "0", "2"]:
        return "l"
    if s in ["r", "right", "1", "3"]:
        return "r"
    return "l"


def compute_depth_metrics(gt_depth, pred_depth):
    """
    Compute error metrics between predicted and ground truth depths for a single image.
    Returns: (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3)
    """
    gt_depth = gt_depth.flatten()
    pred_depth = pred_depth.flatten()

    # Filter out invalid GT (avoid division by zero / log issues)
    mask = gt_depth > 1e-6
    gt_depth = gt_depth[mask]
    pred_depth = pred_depth[mask]

    if gt_depth.size == 0 or pred_depth.size == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

    # Threshold accuracies
    thresh = np.maximum(gt_depth / pred_depth, pred_depth / gt_depth)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    # Errors
    rmse = np.sqrt(np.mean((gt_depth - pred_depth) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt_depth) - np.log(pred_depth)) ** 2))
    abs_rel = np.mean(np.abs(gt_depth - pred_depth) / gt_depth)
    sq_rel = np.mean(((gt_depth - pred_depth) ** 2) / gt_depth)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def visualize_depth(pred_depth, gt_depth, rgb_image):
    """
    Side-by-side visualization: RGB | Pred depth | GT depth
    Uses magma colormap on inverse depth. Missing GT regions are black.
    """
    H, W = gt_depth.shape[:2]

    inv_pred = np.zeros_like(pred_depth, dtype=np.float32)
    inv_gt = np.zeros_like(gt_depth, dtype=np.float32)
    valid_mask = gt_depth > 0

    inv_pred = 1.0 / (pred_depth + 1e-6)
    if valid_mask.any():
        inv_gt[valid_mask] = 1.0 / (gt_depth[valid_mask] + 1e-6)

    # Normalize using GT inverse depth if available
    if valid_mask.any():
        vmin = float(inv_gt[valid_mask].min())
        vmax = float(np.percentile(inv_gt[valid_mask], 95))
    else:
        vmin = float(inv_pred.min())
        vmax = float(np.percentile(inv_pred, 95))

    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap="magma")

    vis_pred = (mapper.to_rgba(inv_pred)[:, :, :3] * 255).astype(np.uint8)
    vis_gt = (mapper.to_rgba(inv_gt)[:, :, :3] * 255).astype(np.uint8)
    vis_gt[~valid_mask] = 0

    # Ensure rgb_image matches H,W and is uint8 RGB
    if rgb_image.dtype != np.uint8:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    if rgb_image.shape[0] != H or rgb_image.shape[1] != W:
        rgb_image = cv2.resize(rgb_image, (W, H))
    if rgb_image.ndim == 2 or rgb_image.shape[2] != 3:
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)

    return np.concatenate((rgb_image, vis_pred, vis_gt), axis=1)


def _read_split_file(split_path):
    with open(split_path, "r") as f:
        return [line.strip() for line in f.readlines() if line.strip()]


def _find_existing_image_path(data_path, entry, exts=(".jpg", ".png", ".jpeg")):
    """
    Tries to resolve 'entry' into a real image path under data_path.
    Supports:
    - absolute paths
    - relative paths that already include extension
    - relative paths without extension (tries .jpg/.png/.jpeg)
    - entries that may contain extra tokens -> uses first token as path fallback
    """
    if not entry:
        return None

    # Some splits contain extra fields; keep full line for the 3-field format handler,
    # otherwise fallback to first token for "path-like" splits.
    candidate = entry

    # If it's absolute and exists
    if os.path.isabs(candidate) and os.path.isfile(candidate):
        return candidate

    # Try joining directly
    p = os.path.join(data_path, candidate)
    if os.path.isfile(p):
        return p

    # If no extension, try appending
    root, ext = os.path.splitext(candidate)
    if ext == "":
        for e in exts:
            p2 = os.path.join(data_path, candidate + e)
            if os.path.isfile(p2):
                return p2
            p3 = os.path.join(data_path, root + e)
            if os.path.isfile(p3):
                return p3
    else:
        # Has extension but didn't exist above; try with first token
        pass

    # Fallback: first token only
    tok = candidate.split()[0]
    if tok != candidate:
        if os.path.isabs(tok) and os.path.isfile(tok):
            return tok
        p4 = os.path.join(data_path, tok)
        if os.path.isfile(p4):
            return p4
        r2, e2 = os.path.splitext(tok)
        if e2 == "":
            for e in exts:
                p5 = os.path.join(data_path, tok + e)
                if os.path.isfile(p5):
                    return p5

    return None


def _try_existing_file(data_path, rel_or_abs_path, exts=(".jpg", ".png", ".jpeg")):
    """
    Resolves an absolute/relative path with or without extension.
    """
    if not rel_or_abs_path:
        return None

    candidate = rel_or_abs_path.replace("\\", "/").strip()
    candidate_abs = candidate if os.path.isabs(candidate) else os.path.join(data_path, candidate)
    if os.path.isfile(candidate_abs):
        return candidate_abs

    root, ext = os.path.splitext(candidate_abs)
    if ext == "":
        for e in exts:
            p = root + e
            if os.path.isfile(p):
                return p
    return None


def _resolve_hamlyn_image_path(data_path, entry):
    """
    Resolve Hamlyn split lines robustly.
    Expected common format: "rectified24 1 l"
    """
    line = entry.strip()
    parts = line.split()
    if not parts:
        return None

    tok0 = parts[0].replace("\\", "/").strip("/")
    frame_tok = os.path.splitext(parts[1])[0] if len(parts) >= 2 else None
    side = _normalize_side(parts[2]) if len(parts) >= 3 else "l"
    cam_folder = "image01" if side == "l" else "image02"

    # If token already points to an image/file, use it.
    p = _try_existing_file(data_path, tok0)
    if p is not None:
        return p

    if frame_tok is not None and frame_tok.isdigit():
        frame10 = str(int(frame_tok)).zfill(10)

        # Case: tok0 already includes image01/image02 folder
        if tok0.endswith("/image01") or tok0.endswith("/image02"):
            p = _try_existing_file(data_path, f"{tok0}/{frame10}")
            if p is not None:
                return p

        # Case: tok0 == rectifiedXX/rectifiedXX
        segs = tok0.split("/")
        if len(segs) == 2 and segs[0] == segs[1]:
            p = _try_existing_file(data_path, f"{tok0}/{cam_folder}/{frame10}")
            if p is not None:
                return p

        # Case: tok0 == rectifiedXX
        base = segs[0]
        p = _try_existing_file(data_path, f"{base}/{base}/{cam_folder}/{frame10}")
        if p is not None:
            return p

    # Fallback to generic resolver
    return _find_existing_image_path(data_path, entry)


def _resolve_hamlyn_depth_path(data_path, entry, img_path=None):
    """
    Resolve Hamlyn GT depth path from split line (or from resolved image path).
    Looks for depth01/depth02 with common depth extensions.
    """
    line = entry.strip()
    parts = line.split()
    tok0 = parts[0].replace("\\", "/").strip("/") if parts else ""
    frame_tok = os.path.splitext(parts[1])[0] if len(parts) >= 2 else None
    side = _normalize_side(parts[2]) if len(parts) >= 3 else "l"

    depth_exts = (".png", ".tiff", ".tif", ".npy", ".npz")

    if frame_tok is not None and frame_tok.isdigit():
        frame10 = str(int(frame_tok)).zfill(10)
        depth_folder = "depth01" if side == "l" else "depth02"

        segs = tok0.split("/") if tok0 else []

        # Case: tok0 == rectifiedXX/rectifiedXX
        if len(segs) == 2 and segs[0] == segs[1]:
            p = _try_existing_file(data_path, f"{tok0}/{depth_folder}/{frame10}", exts=depth_exts)
            if p is not None:
                return p

        # Case: tok0 == rectifiedXX
        if segs:
            base = segs[0]
            p = _try_existing_file(data_path, f"{base}/{base}/{depth_folder}/{frame10}", exts=depth_exts)
            if p is not None:
                return p

    # Derive from image path if available.
    if img_path is not None:
        rel = os.path.relpath(img_path, data_path).replace("\\", "/")
        rel_root, _ = os.path.splitext(rel)
        if "/image01/" in rel_root:
            rel_depth = rel_root.replace("/image01/", "/depth01/")
        elif "/image02/" in rel_root:
            rel_depth = rel_root.replace("/image02/", "/depth02/")
        else:
            rel_depth = rel_root

        p = _try_existing_file(data_path, rel_depth, exts=depth_exts)
        if p is not None:
            return p

    return None


def _load_depth_array(depth_path):
    """
    Load GT depth from npz/npy/image into float32.
    """
    ext = os.path.splitext(depth_path)[1].lower()

    if ext == ".npy":
        arr = np.load(depth_path, allow_pickle=True)
        return arr.astype(np.float32)

    if ext == ".npz":
        depth_data = np.load(depth_path, allow_pickle=True)
        key = "depth" if "depth" in depth_data.files else ("data" if "data" in depth_data.files else depth_data.files[0])
        return depth_data[key].astype(np.float32)

    # Image depth (e.g., uint16 png/tiff in Hamlyn)
    arr = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"Unable to read depth file: {depth_path}")
    return arr.astype(np.float32)


def _load_gt_depths_npz(gt_path):
    """
    Mirrors your Monodepth2 Hamlyn gt_depths loading behavior:
    - expects key 'data'
    - if object array -> convert to list
    """
    data_npz = np.load(gt_path, fix_imports=True, encoding="latin1", allow_pickle=True)

    # Prefer "data" (Monodepth2 convention for custom splits)
    if "data" in data_npz.files:
        gt_depths = data_npz["data"]
    else:
        # fallback: first key
        gt_depths = data_npz[data_npz.files[0]]

    if isinstance(gt_depths, np.ndarray) and gt_depths.dtype == object:
        gt_depths = list(gt_depths)

    return gt_depths


def main():
    parser = argparse.ArgumentParser(description="Depth evaluation (EndoSfM / Hamlyn / Monodepth-style splits)")
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--log_dir", type=str, default=".")
    parser.add_argument("--load_weights_folder", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)

    # ✅ Added "hamlyn"
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["endovis", "hamlyn", "kitti", "nyu"],
        default="endovis",
        help="Dataset name (affects GT loading + depth range)",
    )

    # Split can be:
    # - a .txt path
    # - a split name like "hamlyn" (expects splits/<split>/test_files.txt)
    parser.add_argument("--split", type=str, required=True)

    parser.add_argument(
        "--resnet_layers",
        type=int,
        choices=[18, 50],
        default=18,
        help="Must match training (18 or 50)",
    )
    parser.add_argument(
        "--img_height",
        type=int,
        default=None,
        help="Network input height for evaluation. "
             "If omitted, defaults to 288 for endovis/hamlyn and original height otherwise.",
    )
    parser.add_argument(
        "--img_width",
        type=int,
        default=None,
        help="Network input width for evaluation. "
             "If omitted, defaults to 512 for endovis/hamlyn and original width otherwise.",
    )
    parser.add_argument(
        "--eval_mono",
        action="store_true",
        help="If set, uses median scaling for monocular evaluation",
    )
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--min_depth",
        type=float,
        default=None,
        help="Minimum valid depth for metrics. If omitted, dataset default is used.",
    )
    parser.add_argument(
        "--max_depth",
        type=float,
        default=None,
        help="Maximum valid depth for metrics. If omitted, dataset default is used.",
    )
    parser.add_argument(
        "--gt_depths_path",
        type=str,
        default=None,
        help="Optional explicit path to gt_depths.npz. "
             "If omitted, expects gt_depths.npz next to split file. "
             "For Hamlyn, if not found, falls back to depth01/depth02 files.",
    )
    parser.add_argument(
        "--hamlyn_official_protocol",
        action="store_true",
        help="Use Hamlyn protocol reported by Endo-Depth-and-Motion: valid GT in [1, 300] (depth map units). "
             "You can still override with --min_depth/--max_depth.",
    )

    # Optional: if you want to force using splits folder from this script location
    parser.add_argument(
        "--splits_root",
        "--splits-root",
        type=str,
        default=None,
        help="Root directory containing splits/<split>/test_files.txt and gt_depths.npz. "
             "If not set, defaults to directory of this script.",
    )

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------- LOAD MODEL -----------------
    import models  # assumes your EndoSfM repo has models.py

    disp_net = models.DispResNet(args.resnet_layers, False).to(device)
    disp_net.eval()

    # ----------------- LOAD WEIGHTS -----------------
    if os.path.isdir(args.load_weights_folder):
        disp_path = os.path.join(args.load_weights_folder, "dispnet_model_best.pth.tar")
        alt_disp_path = os.path.join(args.load_weights_folder, "dispnet_checkpoint.pth.tar")

        chosen_disp_path = None
        for cand in [disp_path, alt_disp_path]:
            if os.path.isfile(cand):
                chosen_disp_path = cand
                break

        if chosen_disp_path is None:
            # fallback scan
            for f in os.listdir(args.load_weights_folder):
                lf = f.lower()
                if "disp" in lf or "depth" in lf:
                    chosen_disp_path = os.path.join(args.load_weights_folder, f)
                    break

        if chosen_disp_path is None:
            raise FileNotFoundError(f"Could not find DispResNet weights in {args.load_weights_folder}")

        print(f"-> Loading DispResNet weights from {chosen_disp_path}")
        disp_weights = torch.load(chosen_disp_path, map_location=device)
        if isinstance(disp_weights, dict) and "state_dict" in disp_weights:
            disp_weights = disp_weights["state_dict"]
        disp_net.load_state_dict(disp_weights, strict=False)

    else:
        weight_path = args.load_weights_folder
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Weights file {weight_path} not found")
        print(f"-> Loading model weights from {weight_path}")
        weights = torch.load(weight_path, map_location=device)
        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
        disp_net.load_state_dict(weights, strict=False)

    # ----------------- RESOLVE SPLIT PATH -----------------
    splits_root = args.splits_root
    if splits_root is None:
        splits_root = os.path.join(os.path.dirname(__file__), "splits")

    # If args.split is a .txt file path, use it.
    # Else treat as split name: splits/<split>/test_files.txt
    if args.split.endswith(".txt") and os.path.isfile(args.split):
        split_path = args.split
    else:
        split_path = os.path.join(splits_root, args.split, "test_files.txt")

    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"Split file not found: {split_path}")

    print(f"-> Evaluating on split: {split_path}")
    image_list = _read_split_file(split_path)

    # ----------------- LOAD GT DEPTHS (endovis/hamlyn) -----------------
    gt_depths = None
    use_indexed_gt_depths = False
    if args.dataset in ["endovis", "hamlyn"]:
        gt_path = args.gt_depths_path if args.gt_depths_path is not None else os.path.join(os.path.dirname(split_path), "gt_depths.npz")
        if os.path.isfile(gt_path):
            print(f"-> Loading ground truth depths from {gt_path}")
            gt_depths = _load_gt_depths_npz(gt_path)
            use_indexed_gt_depths = True

            # Sanity: Monodepth2-style expects alignment by index
            num_gt = len(gt_depths) if isinstance(gt_depths, list) else gt_depths.shape[0]
            print(f"-> num_images_in_split: {len(image_list)}, num_gt: {num_gt}")
            if len(image_list) != num_gt:
                print(
                    "[WARN] split/test_files.txt and gt_depths.npz length mismatch. "
                    "Evaluation is index-aligned; samples without GT index will be skipped."
                )
        elif args.dataset == "hamlyn":
            print(
                "-> gt_depths.npz not found. "
                "Will load Hamlyn depth maps directly from depth01/depth02 folders."
            )
        else:
            raise FileNotFoundError(
                f"Ground truth depths file not found at {gt_path}. "
                f"Expected gt_depths.npz next to {split_path}"
            )

    # ----------------- VIS OUTPUT DIR -----------------
    vis_dir = None
    if args.save_vis:
        vis_dir = args.output_dir if args.output_dir else os.path.join(args.log_dir, args.model_name, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        print(f"-> Will save visualizations to: {vis_dir}")

    # ----------------- DEPTH RANGE -----------------
    default_min_depth = 1e-3
    default_max_depth = 150.0
    if args.dataset == "kitti":
        default_max_depth = 80.0
    elif args.dataset == "nyu":
        default_max_depth = 10.0
    elif args.dataset == "hamlyn" and args.hamlyn_official_protocol:
        default_min_depth = 1.0
        default_max_depth = 300.0
        print("-> Hamlyn official protocol enabled (valid GT in [1, 300], in depth-map units).")

    min_depth = args.min_depth if args.min_depth is not None else default_min_depth
    max_depth = args.max_depth if args.max_depth is not None else default_max_depth

    if max_depth <= min_depth:
        raise ValueError(f"Invalid depth range: min_depth={min_depth}, max_depth={max_depth}")
    print(f"-> Using depth range [{min_depth}, {max_depth}]")

    # ----------------- METRIC ACCUMULATORS -----------------
    abs_rel_errors, sq_rel_errors = [], []
    rmse_errors, rmse_log_errors = [], []
    a1_acc, a2_acc, a3_acc = [], [], []

    # ----------------- MAIN LOOP -----------------
    for idx, entry in enumerate(image_list):
        line = entry.strip()
        parts = line.split()

        # ---------- Resolve image path ----------
        img_path = None

        # Hamlyn split style: "rectifiedXX frame side"
        if args.dataset == "hamlyn":
            img_path = _resolve_hamlyn_image_path(args.data_path, line)

        # Legacy endovis-style: "subdir frame_id side"
        if img_path is None and len(parts) == 3:
            subdir, frame_id, side = parts
            img_rel_path = os.path.join(subdir, "data", f"{frame_id}.jpg")
            img_path = os.path.join(args.data_path, img_rel_path)
            if not os.path.isfile(img_path):
                # try png
                img_rel_path = os.path.join(subdir, "data", f"{frame_id}.png")
                img_path = os.path.join(args.data_path, img_rel_path)

        # Hamlyn / generic: line often is a relative image path
        if img_path is None or not os.path.isfile(img_path):
            img_path = _find_existing_image_path(args.data_path, line)

        if img_path is None or not os.path.isfile(img_path):
            print(f"Warning: Image not found for entry '{line}'. Skipping.")
            continue

        # ---------- Load GT depth (index-aligned like your Monodepth2 eval_depth) ----------
        if args.dataset in ["endovis", "hamlyn"]:
            if use_indexed_gt_depths:
                num_gt = len(gt_depths) if isinstance(gt_depths, list) else gt_depths.shape[0]
                if idx >= num_gt:
                    print(f"Warning: No GT depth for idx {idx} (img {img_path}). Skipping.")
                    continue
                gt_depth = gt_depths[idx].astype(np.float32)
            elif args.dataset == "hamlyn":
                gt_depth_path = _resolve_hamlyn_depth_path(args.data_path, line, img_path=img_path)
                if gt_depth_path is None or not os.path.isfile(gt_depth_path):
                    print(f"Warning: Hamlyn GT depth not found for entry '{line}'. Skipping.")
                    continue
                try:
                    gt_depth = _load_depth_array(gt_depth_path)
                except Exception as e:
                    print(f"Warning: Failed loading GT depth '{gt_depth_path}': {e}. Skipping.")
                    continue
            else:
                raise RuntimeError("No GT source available for this split.")
        else:
            # KITTI / NYU fallback per-image load (kept from your code)
            base_name = os.path.splitext(img_path)[0]
            depth_path = base_name + ".npz"
            if not os.path.isfile(depth_path):
                depth_path = base_name + "_depth.npz"
            if not os.path.isfile(depth_path):
                alt_npy = base_name + ".npy"
                if os.path.isfile(alt_npy):
                    depth_path = alt_npy
            if not os.path.isfile(depth_path):
                print(f"Warning: GT depth not found for {img_path}. Skipping.")
                continue

            depth_data = np.load(depth_path, allow_pickle=True)
            if isinstance(depth_data, np.ndarray):
                gt_depth = depth_data.astype(np.float32)
            else:
                key = (
                    "depth"
                    if "depth" in depth_data.files
                    else ("data" if "data" in depth_data.files else depth_data.files[0])
                )
                gt_depth = depth_data[key].astype(np.float32)

        # ---------- Load image ----------
        orig_bgr = cv2.imread(img_path)
        if orig_bgr is None:
            print(f"Warning: Unable to load image {img_path}. Skipping.")
            continue
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        orig_h, orig_w = orig_rgb.shape[:2]

        # ---------- Eval input size ----------
        if args.img_height is not None and args.img_width is not None:
            net_h, net_w = int(args.img_height), int(args.img_width)
        elif args.dataset in ["hamlyn", "endovis"]:
            # Match training transform default in this repo.
            net_h, net_w = 288, 512
        else:
            net_h, net_w = orig_h, orig_w

        if net_h <= 0 or net_w <= 0:
            raise ValueError(f"Invalid eval image size: {net_h}x{net_w}")

        eval_rgb = orig_rgb
        if (orig_h, orig_w) != (net_h, net_w):
            eval_rgb = cv2.resize(orig_rgb, (net_w, net_h), interpolation=cv2.INTER_LINEAR)

        # Robustness for UNet-like skip concatenations: enforce mult-of-32 by reflective padding.
        pad_h = (32 - (net_h % 32)) % 32
        pad_w = (32 - (net_w % 32)) % 32
        if pad_h > 0 or pad_w > 0:
            eval_rgb = cv2.copyMakeBorder(
                eval_rgb,
                0,
                pad_h,
                0,
                pad_w,
                borderType=cv2.BORDER_REFLECT_101,
            )

        # ---------- Normalize (kept as-is from your script) ----------
        img = eval_rgb.astype(np.float32) / 255.0
        img -= np.array([0.45, 0.45, 0.45], dtype=np.float32)
        img /= np.array([0.225, 0.225, 0.225], dtype=np.float32)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # ---------- Predict disparity ----------
        with torch.no_grad():
            pred_disp = disp_net(img_tensor)

        if isinstance(pred_disp, (list, tuple)):
            pred_disp = pred_disp[0]
        pred_disp = pred_disp.squeeze().cpu().numpy()

        # Remove network padding and bring disparity back to original image resolution.
        if pad_h > 0 or pad_w > 0:
            pred_disp = pred_disp[:net_h, :net_w]
        if (net_h, net_w) != (orig_h, orig_w):
            pred_disp = cv2.resize(pred_disp, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)

        # ---------- Convert to depth & resize to GT ----------
        H_gt, W_gt = gt_depth.shape[:2]

        pred_disp[pred_disp <= 0] = 1e-6
        pred_depth = 1.0 / pred_disp

        # Resize in inverse-depth space (more stable)
        pred_inv_depth = cv2.resize(
            1.0 / (pred_depth + 1e-6),
            (W_gt, H_gt),
            interpolation=cv2.INTER_LINEAR,
        )
        pred_depth_resized = 1.0 / (pred_inv_depth + 1e-6)

        # ---------- Valid mask (same spirit as Monodepth2) ----------
        mask = (gt_depth >= min_depth) & (gt_depth <= max_depth)
        # Some GTs mark missing as 0
        mask = mask & (gt_depth > 0)

        if not np.any(mask):
            print(f"Warning: No valid GT pixels for {img_path}. Skipping.")
            continue

        # KITTI crop (optional)
        if args.dataset == "kitti":
            gt_h, gt_w = gt_depth.shape
            crop = (
                int(0.40810811 * gt_h),
                int(0.99189189 * gt_h),
                int(0.03594771 * gt_w),
                int(0.96405229 * gt_w),
            )
            crop_mask = np.zeros_like(mask, dtype=bool)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = True
            mask = mask & crop_mask
            if not np.any(mask):
                continue

        valid_pred = pred_depth_resized[mask]
        valid_gt = gt_depth[mask]

        # Debug prints
        if idx in [0, 10, 50]:
            print("idx:", idx)
            print("  img:", img_path)
            print("  GT median:", float(np.median(valid_gt)))
            print("  Pred median:", float(np.median(valid_pred)))

        # ---------- Monocular median scaling ----------
        if args.eval_mono:
            median_pred = np.median(valid_pred)
            if median_pred < 1e-6:
                print(f"Warning: Median pred depth ~0 for {img_path}. Skipping.")
                continue
            scale_ratio = np.median(valid_gt) / (median_pred + 1e-6)
            pred_depth_resized *= scale_ratio
            valid_pred = valid_pred * scale_ratio

        valid_pred = np.clip(valid_pred, min_depth, max_depth)
        valid_gt = np.clip(valid_gt, min_depth, max_depth)

        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_depth_metrics(valid_gt, valid_pred)

        if not np.isnan(abs_rel):
            abs_rel_errors.append(abs_rel)
            sq_rel_errors.append(sq_rel)
            rmse_errors.append(rmse)
            rmse_log_errors.append(rmse_log)
            a1_acc.append(a1)
            a2_acc.append(a2)
            a3_acc.append(a3)

        # ---------- Visualization ----------
        if args.save_vis and vis_dir is not None:
            vis_img = visualize_depth(pred_depth_resized, gt_depth, orig_rgb)
            vis_filename = f"{idx:06d}.png"
            cv2.imwrite(
                os.path.join(vis_dir, vis_filename),
                cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR),
            )

    # ----------------- SUMMARY -----------------
    if len(abs_rel_errors) == 0:
        print("Error: No valid predictions were evaluated.")
        return

    mean_abs_rel = float(np.mean(abs_rel_errors))
    mean_sq_rel = float(np.mean(sq_rel_errors))
    mean_rmse = float(np.mean(rmse_errors))
    mean_rmse_log = float(np.mean(rmse_log_errors))
    mean_a1 = float(np.mean(a1_acc))
    mean_a2 = float(np.mean(a2_acc))
    mean_a3 = float(np.mean(a3_acc))

    num_samples = len(abs_rel_errors)
    print(f"\n-> Evaluated {num_samples} samples")
    print(
        f"   abs_rel = {mean_abs_rel:.4f}, sq_rel = {mean_sq_rel:.4f}, "
        f"rmse = {mean_rmse:.4f}, rmse_log = {mean_rmse_log:.4f}, "
        f"a1 = {mean_a1:.4f}, a2 = {mean_a2:.4f}, a3 = {mean_a3:.4f}"
    )


if __name__ == "__main__":
    main()
