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
    Returns a tuple: (abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3).
    """
    # Flatten the depth arrays for metric computation
    gt_depth = gt_depth.flatten()
    pred_depth = pred_depth.flatten()
    # Filter out any invalid zero values (to avoid division by zero)
    mask = gt_depth > 1e-6
    gt_depth = gt_depth[mask]
    pred_depth = pred_depth[mask]
    # Compute threshold-based accuracy
    thresh = np.maximum(gt_depth / pred_depth, pred_depth / gt_depth)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()
    # Compute RMSE and RMSE(log)
    rmse = np.sqrt(np.mean((gt_depth - pred_depth) ** 2))
    rmse_log = np.sqrt(np.mean((np.log(gt_depth) - np.log(pred_depth)) ** 2))
    # Compute Absolute Relative and Squared Relative errors
    abs_rel = np.mean(np.abs(gt_depth - pred_depth) / gt_depth)
    sq_rel = np.mean(((gt_depth - pred_depth) ** 2) / gt_depth)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3

def visualize_depth(pred_depth, gt_depth, rgb_image):
    """
    Create a side-by-side visualization of the RGB image, predicted depth, and ground truth depth.
    Depth maps are visualized using the 'magma' colormap on inverse depth.
    Missing ground truth regions are shown in black in the GT depth image.
    Returns a concatenated RGB image (numpy array).
    """
    H, W = gt_depth.shape
    # Compute inverse depth for colormap (avoid division by zero)
    inv_pred = np.zeros_like(pred_depth, dtype=np.float32)
    inv_gt = np.zeros_like(gt_depth, dtype=np.float32)
    valid_mask = gt_depth > 0
    if pred_depth.max() > 0:
        inv_pred = 1.0 / (pred_depth + 1e-6)
    if valid_mask.any():
        inv_gt[valid_mask] = 1.0 / (gt_depth[valid_mask] + 1e-6)
    # Set up colormap normalization based on GT inverse depth distribution
    if valid_mask.any():
        vmin = inv_gt[valid_mask].min()
        vmax = np.percentile(inv_gt[valid_mask], 95)
    else:
        # If no valid GT (should not happen if metrics computed), fallback to pred inv depth range
        vmin = inv_pred.min()
        vmax = np.percentile(inv_pred, 95)
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap='magma')
    vis_pred = (mapper.to_rgba(inv_pred)[:, :, :3] * 255).astype(np.uint8)
    vis_gt   = (mapper.to_rgba(inv_gt)[:, :, :3] * 255).astype(np.uint8)
    # Mark missing GT areas as pure black for clarity
    vis_gt[~valid_mask] = 0
    # Ensure the input image is uint8 RGB
    if rgb_image.dtype != np.uint8:
        rgb_image = (rgb_image * 255).astype(np.uint8)
    if rgb_image.shape[0] != H or rgb_image.shape[1] != W:
        rgb_image = cv2.resize(rgb_image, (W, H))
    if rgb_image.shape[2] != 3:
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_GRAY2RGB)
    # Concatenate RGB, predicted depth, and GT depth images side by side
    return np.concatenate((rgb_image, vis_pred, vis_gt), axis=1)

def main():
    parser = argparse.ArgumentParser(description="Depth evaluation for Endo-SfM (SCaRED)")
    parser.add_argument("--model_name", type=str, required=True,
                        help="Name of the experiment/model (for logging/display purposes)")
    parser.add_argument("--log_dir", type=str, default=".", help="Directory containing logs/checkpoints")
    parser.add_argument("--load_weights_folder", type=str, required=True,
                        help="Folder or file path to the model weights to load")
    parser.add_argument("--data_path", type=str, required=True, help="Root path of the SCaRED dataset")
    parser.add_argument("--dataset", type=str, choices=["endovis", "kitti", "nyu"], default="endovis",
                        help="Dataset name (affects evaluation parameters like depth range)")
    parser.add_argument("--split", type=str, required=True, help="Name of the data split to evaluate (e.g. 'endovis')")
    parser.add_argument("--resnet_layers", type=int, choices=[18, 50], default=18,
                        help="Number of ResNet layers for DispResNet (must match training, 18 or 50)")
    parser.add_argument("--eval_mono", action="store_true",
                        help="If set, uses median scaling for monocular evaluation (no ground truth scale)")
    parser.add_argument("--save_vis", action="store_true",
                        help="If set, saves depth prediction vs ground truth visualization images")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory to save visualization images (if --save_vis is used)")
    args = parser.parse_args()

    # Set device for model and data tensors
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the DispResNet model (depth prediction network)
    import models  # assume models module is in the Python path
    disp_net = models.DispResNet(args.resnet_layers, False).to(device)  # with_pretrain=False (weights will be loaded)
    disp_net.eval()
    pose_net = None  # will load if pose weights are present (not required for depth eval)

    # Load model weights from file(s)
    if os.path.isdir(args.load_weights_folder):
        # Construct expected file paths for depth and pose networks
        disp_path = os.path.join(args.load_weights_folder, "disp_net.pth")
        pose_path = os.path.join(args.load_weights_folder, "pose_net.pth")
        # If standard names not found, try to auto-detect files
        if not os.path.isfile(disp_path):
            for f in os.listdir(args.load_weights_folder):
                lf = f.lower()
                if "disp" in lf or "depth" in lf:
                    disp_path = os.path.join(args.load_weights_folder, f)
                if "pose" in lf:
                    pose_path = os.path.join(args.load_weights_folder, f)
        if os.path.isfile(disp_path):
            print(f"-> Loading DispResNet weights from {disp_path}")
            disp_weights = torch.load(disp_path, map_location=device)
            if isinstance(disp_weights, dict) and "state_dict" in disp_weights:
                disp_weights = disp_weights["state_dict"]
            disp_net.load_state_dict(disp_weights, strict=False)
        else:
            raise FileNotFoundError(f"Could not find DispResNet weights in {args.load_weights_folder}")
        if pose_path and os.path.isfile(pose_path):
            print(f"-> Loading PoseResNet weights from {pose_path}")
            pose_weights = torch.load(pose_path, map_location=device)
            if isinstance(pose_weights, dict) and "state_dict" in pose_weights:
                pose_weights = pose_weights["state_dict"]
            pose_net = models.PoseResNet(18, False).to(device)
            pose_net.load_state_dict(pose_weights, strict=False)
            pose_net.eval()
    else:
        # Single file case: try to load as DispResNet (or PoseResNet if appropriate)
        weight_path = args.load_weights_folder
        if not os.path.isfile(weight_path):
            raise FileNotFoundError(f"Weights file {weight_path} not found")
        print(f"-> Loading model weights from {weight_path}")
        weights = torch.load(weight_path, map_location=device)
        if isinstance(weights, dict) and "state_dict" in weights:
            weights = weights["state_dict"]
        # Attempt to load into DispResNet
        try:
            disp_net.load_state_dict(weights, strict=False)
            print("Loaded weights into DispResNet.")
        except Exception as e:
            # If it fails, try PoseResNet
            pose_net = models.PoseResNet(18, False).to(device)
            try:
                pose_net.load_state_dict(weights, strict=False)
                pose_net.eval()
                print("Loaded weights into PoseResNet.")
            except Exception as e2:
                raise RuntimeError("Failed to load the provided weights into DispResNet or PoseResNet.") from e2

    # Determine the path to the split file listing test images
    split_file = args.split if args.split.endswith(".txt") else f"{args.split}_files.txt"
    possible_paths = [
        split_file,
        os.path.join(os.path.dirname(__file__) if '__file__' in globals() else ".", split_file),
        os.path.join(args.data_path, split_file),
        os.path.join(args.data_path, "splits", args.split, "test_files.txt")
    ]
    split_path = None
    for sp in possible_paths:
        if os.path.isfile(sp):
            split_path = sp
            break
    if split_path is None:
        raise FileNotFoundError(f"Split file '{split_file}' not found. Please provide a valid split.")
    print(f"-> Evaluating depth on split: {split_path}")
    with open(split_path, 'r') as f:
        image_list = [line.strip() for line in f.readlines() if line.strip()]

    # Set up visualization output directory if needed
    vis_dir = None
    if args.save_vis:
        vis_dir = args.output_dir if args.output_dir else os.path.join(args.log_dir, args.model_name, "vis")
        os.makedirs(vis_dir, exist_ok=True)
        print(f"-> Will save visualizations to: {vis_dir}")

    # Dataset-specific depth range parameters
    min_depth = 1e-3
    max_depth = np.inf
    if args.dataset == "kitti":
        max_depth = 80.0
    elif args.dataset == "nyu":
        max_depth = 10.0
    # (For 'endovis', we use the full range of available depth data; depths are in mm)

    # Lists to accumulate metric results
    abs_rel_errors = []
    sq_rel_errors = []
    rmse_errors = []
    rmse_log_errors = []
    a1_acc = []
    a2_acc = []
    a3_acc = []

    # Iterate over all test images
    for idx, entry in enumerate(image_list):
        # Parse the entry to get image path relative to data_path
        line = entry.strip()
        base, fname = os.path.split(line)

        # Extract frame number (before underscore or dot)
        stem = fname.split("_")[0] if "_" in fname else os.path.splitext(fname)[0]

        # Construct expected SCARED path: datasetX/keyframeY/data/NNN.jpg
        rel_image_path = os.path.join(base, "data", stem + ".jpg")
        img_path = os.path.join(args.data_path, rel_image_path)
        if not os.path.isfile(img_path):
            print(f"Warning: Image file not found: {img_path}. Skipping.")
            continue

        # Determine corresponding ground truth depth file path (.npz or .npy)
        base_name = os.path.splitext(img_path)[0]
        depth_path = base_name + ".npz"
        if not os.path.isfile(depth_path):
            depth_path = base_name + "_depth.npz"
        if not os.path.isfile(depth_path):
            alt_npy = base_name + ".npy"
            if os.path.isfile(alt_npy):
                depth_path = alt_npy
        if not os.path.isfile(depth_path):
            print(f"Warning: Ground truth depth not found for {img_path}. Skipping.")
            continue

        # Load and preprocess the input image
        orig_bgr = cv2.imread(img_path)
        if orig_bgr is None:
            print(f"Warning: Unable to load image {img_path}. Skipping.")
            continue
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        img = orig_rgb.astype(np.float32) / 255.0
        # Normalize image using the same mean and std as training (ImageNet)
        img -= np.array([0.45, 0.45, 0.45], dtype=np.float32)
        img /= np.array([0.225, 0.225, 0.225], dtype=np.float32)
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # Run the model to get predicted disparity
        with torch.no_grad():
            pred_disp = disp_net(img_tensor)
        # If model returns multi-scale outputs, use the highest resolution output
        if isinstance(pred_disp, (list, tuple)):
            pred_disp = pred_disp[0]
        pred_disp = pred_disp.squeeze().cpu().numpy()  # shape: H_pred x W_pred (disparity values)

        # Load ground truth depth map
        depth_data = np.load(depth_path, allow_pickle=True)
        if isinstance(depth_data, np.ndarray):
            gt_depth = depth_data.astype(np.float32)
        else:
            # depth_data is an NpzFile if multiple arrays
            key = 'depth' if 'depth' in depth_data else ('data' if 'data' in depth_data else depth_data.files[0])
            gt_depth = depth_data[key].astype(np.float32)
        # If GT depth is in millimeters (as per SCaRED), we keep units as-is for evaluation.

        # Resize predicted depth to GT resolution using inverse depth (disparity) for accuracy
        H_gt, W_gt = gt_depth.shape[:2]
        # Compute predicted depth from disparity
        pred_disp[pred_disp <= 0] = 1e-6  # guard against non-positive disparities
        pred_depth = 1.0 / pred_disp  # depth at model output resolution (arbitrary scale)
        # Resize by inverse depth
        pred_inv_depth = cv2.resize(1.0/(pred_depth + 1e-6), (W_gt, H_gt), interpolation=cv2.INTER_LINEAR)
        pred_depth_resized = 1.0 / (pred_inv_depth + 1e-6)

        # Create mask of valid ground truth pixels (non-zero and within min/max depth bounds)
        mask = (gt_depth > min_depth) & (gt_depth < max_depth) & (gt_depth > 0)
        if not np.any(mask):
            # Skip images with no valid depth data
            print(f"Warning: No valid ground truth in {img_path}. Skipping.")
            continue

        # Optionally apply KITTI crop (only if evaluating KITTI dataset)
        if args.dataset == "kitti":
            gt_h, gt_w = gt_depth.shape
            crop = (int(0.40810811 * gt_h), int(0.99189189 * gt_h),
                    int(0.03594771 * gt_w), int(0.96405229 * gt_w))
            crop_mask = np.zeros_like(mask, dtype=bool)
            crop_mask[crop[0]:crop[1], crop[2]:crop[3]] = True
            mask = mask & crop_mask
            if not np.any(mask):
                continue

        # Extract valid depth values for evaluation
        valid_pred = pred_depth_resized[mask]
        valid_gt = gt_depth[mask]

        # Median scaling for monocular evaluation (align scale to GT median)
        if args.eval_mono:
            scale_ratio = np.median(valid_gt) / (np.median(valid_pred) + 1e-6)
            pred_depth_resized *= scale_ratio
            valid_pred = valid_pred * scale_ratio

        # Clamp predicted depth to eval min/max to avoid extreme outliers affecting metrics
        valid_pred = np.clip(valid_pred, min_depth, max_depth)
        valid_gt = np.clip(valid_gt, min_depth, max_depth)

        # Compute error metrics for this image
        abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = compute_depth_metrics(valid_gt, valid_pred)
        abs_rel_errors.append(abs_rel)
        sq_rel_errors.append(sq_rel)
        rmse_errors.append(rmse)
        rmse_log_errors.append(rmse_log)
        a1_acc.append(a1)
        a2_acc.append(a2)
        a3_acc.append(a3)

        # Save visualization if requested
        if args.save_vis and vis_dir:
            vis_img = visualize_depth(pred_depth_resized, gt_depth, orig_rgb)
            vis_filename = f"{idx:06d}.png"
            cv2.imwrite(os.path.join(vis_dir, vis_filename), cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR))

    # Calculate mean metrics over all images
    if len(abs_rel_errors) == 0:
        print("Error: No valid predictions were evaluated.")
        return
    mean_abs_rel = np.mean(abs_rel_errors)
    mean_sq_rel = np.mean(sq_rel_errors)
    mean_rmse = np.mean(rmse_errors)
    mean_rmse_log = np.mean(rmse_log_errors)
    mean_a1 = np.mean(a1_acc)
    mean_a2 = np.mean(a2_acc)
    mean_a3 = np.mean(a3_acc)

    # Print out evaluation results
    num_samples = len(abs_rel_errors)
    print(f"\n-> Evaluated {num_samples} samples")
    if args.dataset == "kitti":
        print(f"   abs_rel = {mean_abs_rel:.4f}, sq_rel = {mean_sq_rel:.4f}, "
              f"rmse = {mean_rmse:.4f}, rmse_log = {mean_rmse_log:.4f}, "
              f"a1 = {mean_a1:.4f}, a2 = {mean_a2:.4f}, a3 = {mean_a3:.4f}")
    else:
        # For endovis/nyu, sq_rel may not be a standard metric (still computed for completeness)
        print(f"   abs_rel = {mean_abs_rel:.4f}, sq_rel = {mean_sq_rel:.4f}, "
              f"rmse = {mean_rmse:.4f}, rmse_log = {mean_rmse_log:.4f}, "
              f"a1 = {mean_a1:.4f}, a2 = {mean_a2:.4f}, a3 = {mean_a3:.4f}")

if __name__ == "__main__":
    main()
