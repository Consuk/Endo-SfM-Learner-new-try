import os
import cv2
import argparse
import numpy as np
import torch
import matplotlib as mpl
import matplotlib.cm as cm


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
        "--eval_mono",
        action="store_true",
        help="If set, uses median scaling for monocular evaluation",
    )
    parser.add_argument("--save_vis", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)

    # Optional: if you want to force using splits folder from this script location
    parser.add_argument(
        "--splits_root",
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
    if args.dataset in ["endovis", "hamlyn"]:
        gt_path = os.path.join(os.path.dirname(split_path), "gt_depths.npz")
        if not os.path.isfile(gt_path):
            raise FileNotFoundError(
                f"Ground truth depths file not found at {gt_path}. "
                f"Expected gt_depths.npz next to {split_path}"
            )
        print(f"-> Loading ground truth depths from {gt_path}")
        gt_depths = _load_gt_depths_npz(gt_path)

        # Sanity: Monodepth2-style expects alignment by index
        num_gt = len(gt_depths) if isinstance(gt_depths, list) else gt_depths.shape[0]
        print(f"-> num_images_in_split: {len(image_list)}, num_gt: {num_gt}")

    # ----------------- VIS OUTPUT DIR -----------------
    vis_dir = None
    if args.save_vis:
        vis_dir = args.output_dir if args.output_dir else os.path.join(args.log_dir, args.model_name, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        print(f"-> Will save visualizations to: {vis_dir}")

    # ----------------- DEPTH RANGE -----------------
    min_depth = 1e-3
    max_depth = 150.0
    if args.dataset == "kitti":
        max_depth = 80.0
    elif args.dataset == "nyu":
        max_depth = 10.0
    # endovis/hamlyn: keep 150.0 unless you want to change

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

        # Legacy endovis-style: "subdir frame_id side"
        if len(parts) == 3:
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
            if gt_depths is None:
                raise RuntimeError("gt_depths is None but dataset requires gt_depths.npz")

            num_gt = len(gt_depths) if isinstance(gt_depths, list) else gt_depths.shape[0]
            if idx >= num_gt:
                print(f"Warning: No GT depth for idx {idx} (img {img_path}). Skipping.")
                continue

            gt_depth = gt_depths[idx].astype(np.float32)
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

        # ---------- Normalize (kept as-is from your script) ----------
        img = orig_rgb.astype(np.float32) / 255.0
        img -= np.array([0.45, 0.45, 0.45], dtype=np.float32)
        img /= np.array([0.225, 0.225, 0.225], dtype=np.float32)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # ---------- Predict disparity ----------
        with torch.no_grad():
            pred_disp = disp_net(img_tensor)

        if isinstance(pred_disp, (list, tuple)):
            pred_disp = pred_disp[0]
        pred_disp = pred_disp.squeeze().cpu().numpy()

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
        mask = (gt_depth > min_depth) & (gt_depth < max_depth)
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
