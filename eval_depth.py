#!/usr/bin/env python3
import argparse
import numpy as np
import os
import cv2
import matplotlib as mpl
import matplotlib.cm as cm

def load_depths_from_npz(npz_path):
    """Load depth maps from a .npz file.
    Returns a list of numpy arrays (each array is a depth map)."""
    data = np.load(npz_path)
    # If multiple arrays stored (e.g., arr_0, arr_1, ...), load in order
    keys = data.files
    depth_maps = []
    if len(keys) == 0:
        raise ValueError(f"No arrays found in {npz_path}")
    if len(keys) == 1:
        # Single array: could be a stack of depth maps or a single depth map
        arr = data[keys[0]]
        if arr.ndim == 3:
            # Assume shape (N, H, W)
            depth_maps = [arr[i] for i in range(arr.shape[0])]
        else:
            # Only one depth map
            depth_maps = [arr]
    else:
        # Multiple arrays stored, e.g., arr_0 ... arr_N
        # Sort keys numerically if they follow arr_# naming
        try:
            # Sort by integer index if keys are of form "arr_0", "arr_1", ...
            keys_sorted = sorted(keys, key=lambda k: int(k.split('_')[1]) if k.startswith('arr_') else k)
        except Exception:
            # Fallback: lexicographic sort
            keys_sorted = sorted(keys)
        for k in keys_sorted:
            depth_maps.append(data[k])
    return depth_maps

def colorize_depth(depth_map):
    """Convert a depth map to a color image (inverse depth, magma colormap)."""
    # Compute inverse depth (add a small epsilon to avoid division by zero)
    inv_depth = 1.0 / (depth_map + 1e-6)
    # Use 95th percentile of inverse depth as vmax for better contrast
    vmax = np.percentile(inv_depth, 95)
    vmin = inv_depth.min()
    # Normalize inverse depth to [0, 1] for colormap
    normalizer = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    # Apply colormap. mapper.to_rgba returns an RGBA image (last channel is alpha)
    colorized = (mapper.to_rgba(inv_depth)[:, :, :3] * 255).astype(np.uint8)
    return colorized

def evaluate_depth_metrics(gt_depth, pred_depth, eval_mono=False):
    """Compute all evaluation metrics for a single depth map pair (ground truth and prediction).
    If eval_mono is True, apply median scaling to pred_depth relative to gt_depth."""
    # Mask out invalid ground truth pixels (assume 0 indicates missing depth)
    mask = gt_depth > 1e-6
    valid_gt = gt_depth[mask]
    valid_pred = pred_depth[mask]
    if valid_gt.size == 0:
        return None  # No valid pixels to compare (should not happen if data is correct)
    # Apply median scaling if required (monocular evaluation)
    if eval_mono:
        # Avoid division by zero in median computation
        med_pred = np.median(valid_pred) if valid_pred.size > 0 else 0
        med_gt   = np.median(valid_gt)   if valid_gt.size > 0 else 0
        if med_pred > 0:
            scale = med_gt / med_pred
        else:
            scale = 1.0  # if median of pred is zero, skip scaling to avoid inf
        valid_pred = valid_pred * scale
    # Clamp predicted depth to a minimum (to avoid issues with log domain calculations)
    min_depth = 1e-3
    valid_pred = np.clip(valid_pred, min_depth, None)
    valid_gt   = np.clip(valid_gt, min_depth, None)
    # Compute error metrics
    # 1. Absolute Relative Error
    abs_rel = np.mean(np.abs(valid_gt - valid_pred) / valid_gt)
    # 2. Squared Relative Error
    sq_rel = np.mean(((valid_gt - valid_pred) ** 2) / valid_gt)
    # 3. RMSE
    rmse = np.sqrt(np.mean((valid_gt - valid_pred) ** 2))
    # 4. RMSE (log)
    rmse_log = np.sqrt(np.mean((np.log(valid_gt) - np.log(valid_pred)) ** 2))
    # 5-7. Accuracy under thresholds
    # Compute ratio max(pred/gt, gt/pred) for all valid pixels
    ratio = np.maximum(valid_gt / valid_pred, valid_pred / valid_gt)
    a1  = np.mean(ratio < 1.25)
    a2  = np.mean(ratio < 1.25 ** 2)
    a3  = np.mean(ratio < 1.25 ** 3)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def main():
    parser = argparse.ArgumentParser(description="Evaluate monocular depth on SCARED dataset")
    parser.add_argument("--pred_depth", required=True, type=str, help="Path to predicted depth .npz file")
    parser.add_argument("--gt_depth",   required=True, type=str, help="Path to ground-truth depth .npz file")
    parser.add_argument("--test_list",  required=True, type=str, help="Path to text file listing test samples")
    parser.add_argument("--img_dir",    required=True, type=str, help="Root directory of RGB images for test samples")
    parser.add_argument("--eval_mono",  action='store_true', help="If set, perform median scaling for monocular predictions")
    parser.add_argument("--vis_dir",    type=str, help="Directory to save visualization images (optional)")
    args = parser.parse_args()

    # Load predictions and ground truth depth maps
    print(f"Loading predicted depths from {args.pred_depth}...")
    pred_depths = load_depths_from_npz(args.pred_depth)
    print(f"Loading ground-truth depths from {args.gt_depth}...")
    gt_depths   = load_depths_from_npz(args.gt_depth)
    if len(pred_depths) != len(gt_depths):
        raise ValueError(f"Number of predictions ({len(pred_depths)}) and ground truths ({len(gt_depths)}) do not match")
    
    # Load test sample list to ensure alignment and for visualization
    with open(args.test_list, 'r') as f:
        test_lines = [line.strip() for line in f if line.strip()]
    num_samples = len(test_lines)
    if num_samples != len(pred_depths):
        print(f"Warning: Test list has {num_samples} entries, but {len(pred_depths)} depth predictions were loaded.")
        # We will proceed assuming they correspond in order if counts differ.
    else:
        print(f"Test list loaded with {num_samples} samples.")
    
    # Prepare visualization output directory if needed
    if args.vis_dir:
        os.makedirs(args.vis_dir, exist_ok=True)
        print(f"Visualization images will be saved to: {args.vis_dir}")
    
    # Initialize accumulators for metrics
    metric_sums = np.zeros(7, dtype=np.float64)
    metric_count = 0
    
    for i, (pred, gt) in enumerate(zip(pred_depths, gt_depths)):
        # If needed, resize predicted depth to match ground truth resolution
        if pred.shape != gt.shape:
            # Resize using inverse depth to preserve edge sharpness
            gt_h, gt_w = gt.shape[:2]
            inv_pred = 1.0 / (pred + 1e-6)
            inv_pred_resized = cv2.resize(inv_pred, (gt_w, gt_h), interpolation=cv2.INTER_LINEAR)
            pred = 1.0 / (inv_pred_resized + 1e-6)
        # Compute metrics for this sample
        metrics = evaluate_depth_metrics(gt.astype(np.float64), pred.astype(np.float64), eval_mono=args.eval_mono)
        if metrics is None:
            # Skip if no valid pixels (should not happen for SCARED if data is correct)
            continue
        metric_sums += np.array(metrics)
        metric_count += 1
        # Visualization: save side-by-side image if requested
        if args.vis_dir:
            # Construct image path from test_list entry
            # Each line might be like: "datasetX/keyframeY\tZ\tl" (with tab or space separators)
            parts = test_lines[i].replace("\\", "/").split()
            # If the line uses tabs, it may result in multiple parts (e.g., [path, index, side])
            if len(parts) >= 2:
                rel_path = parts[0]
                frame_id = parts[1]
                side = parts[2] if len(parts) >= 3 else ""
                # Construct filename, e.g., "390_l.png" if side given, else just number
                img_name = frame_id + (f"_{side}" if side else "") + ".png"
                img_path = os.path.join(args.img_dir, rel_path, img_name)
            else:
                # If entire line is just a path (including extension)
                img_path = os.path.join(args.img_dir, test_lines[i])
            # Read and prepare the RGB image
            img = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if img is None:
                print(f"Warning: Image not found at {img_path}, skipping visualization for sample {i}")
            else:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                # Colorize predicted and GT depth maps
                pred_color = colorize_depth(pred)
                gt_color   = colorize_depth(gt)
                # Ensure all images have the same height and width
                h, w, _ = img_rgb.shape
                pred_color = cv2.resize(pred_color, (w, h), interpolation=cv2.INTER_NEAREST)
                gt_color   = cv2.resize(gt_color,   (w, h), interpolation=cv2.INTER_NEAREST)
                # Concatenate images side-by-side: [RGB | Predicted Depth | GT Depth]
                concat_img = np.zeros((h, 3*w, 3), dtype=np.uint8)
                concat_img[:, :w, :]       = img_rgb
                concat_img[:, w:2*w, :]    = pred_color
                concat_img[:, 2*w:3*w, :]  = gt_color
                # Save the concatenated image
                out_name = f"{i:04d}.png"
                out_path = os.path.join(args.vis_dir, out_name)
                cv2.imwrite(out_path, cv2.cvtColor(concat_img, cv2.COLOR_RGB2BGR))
    
    # Compute average metrics
    if metric_count == 0:
        print("Error: No metrics computed (check data inputs).")
        return
    avg_metrics = metric_sums / metric_count
    
    # Print results in a formatted table
    print("\nEvaluation results (averaged over {} samples):".format(metric_count))
    header = f"{'Metric':<12} | {'Value':>10}"
    print(header)
    print("-" * len(header))
    metric_names = ["AbsRel", "SqRel", "RMSE", "RMSE(log)", "a1", "a2", "a3"]
    for name, val in zip(metric_names, avg_metrics):
        # For accuracy metrics a1, a2, a3, it might be clearer to show as percentage or fraction.
        # Here we output as fraction (0 to 1) with four decimal places.
        print(f"{name:<12} | {val:10.4f}")

if __name__ == "__main__":
    main()
