import os
import argparse
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors

def compute_errors(gt, pred):
    thresh = np.maximum(gt / pred, pred / gt)
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = np.sqrt(((gt - pred) ** 2).mean())
    rmse_log = np.sqrt((np.log(gt) - np.log(pred)).pow(2).mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)

    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


def load_npz_depth(npz_path):
    data = np.load(npz_path)
    if isinstance(data, np.lib.npyio.NpzFile):
        keys = sorted(data.files)
        return [data[k] for k in keys]
    else:
        return [data]


def colormap_inv_depth(inv_depth, percentile=95):
    valid = inv_depth[np.isfinite(inv_depth)]
    vmax = np.percentile(valid, percentile)
    vmin = valid.min()
    norm = colors.Normalize(vmin=vmin, vmax=vmax)
    mapper = cm.ScalarMappable(norm=norm, cmap='magma')
    color_disp = (mapper.to_rgba(inv_depth)[:, :, :3] * 255).astype(np.uint8)
    return color_disp


def main():
    parser = argparse.ArgumentParser(description="Evaluate depth predictions on SCaRED")
    parser.add_argument('--pred_depth', type=str, required=True, help='Path to predicted .npz depth file')
    parser.add_argument('--gt_depth', type=str, required=True, help='Path to ground truth .npz depth file')
    parser.add_argument('--test_list', type=str, required=True, help='Path to test_files.txt')
    parser.add_argument('--img_dir', type=str, required=False, help='Root directory for test RGB images')
    parser.add_argument('--vis_dir', type=str, required=False, help='Directory to save visualization images')
    parser.add_argument('--eval_mono', action='store_true', help='Enable median scaling (monocular eval)')

    args = parser.parse_args()

    print(f"Loading predicted depths from {args.pred_depth}...")
    pred_depths = load_npz_depth(args.pred_depth)

    print(f"Loading ground truth depths from {args.gt_depth}...")
    gt_depths = load_npz_depth(args.gt_depth)

    print(f"Reading test list from {args.test_list}...")
    with open(args.test_list, 'r') as f:
        test_filenames = [line.strip() for line in f if line.strip()]

    if len(test_filenames) != len(pred_depths) or len(pred_depths) != len(gt_depths):
        raise ValueError(f"Mismatch: {len(test_filenames)} test samples, "
                         f"{len(pred_depths)} predictions, {len(gt_depths)} GT depths")

    if args.vis_dir and not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)

    errors = []

    for i in range(len(test_filenames)):
        pred = pred_depths[i].squeeze()
        gt = gt_depths[i].squeeze()

        if pred.shape != gt.shape:
            # Resize using inverse depth interpolation
            inv_pred = 1.0 / (pred + 1e-6)
            inv_pred_resized = cv2.resize(inv_pred, (gt.shape[1], gt.shape[0]), interpolation=cv2.INTER_LINEAR)
            pred = 1.0 / (inv_pred_resized + 1e-6)

        mask = np.isfinite(gt) & (gt > 1e-3)
        gt_valid = gt[mask]
        pred_valid = pred[mask]

        if args.eval_mono:
            scale = np.median(gt_valid) / (np.median(pred_valid) + 1e-6)
            pred *= scale
            pred_valid *= scale

        errors.append(compute_errors(gt_valid, pred_valid))

        if args.vis_dir and args.img_dir:
            frame_path = os.path.join(args.img_dir, test_filenames[i])
            if os.path.exists(frame_path):
                img = cv2.imread(frame_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                h, w = gt.shape
                img = np.zeros((h, w, 3), dtype=np.uint8)

            inv_pred = 1.0 / (pred + 1e-6)
            inv_gt = 1.0 / (gt + 1e-6)
            vis_pred = colormap_inv_depth(inv_pred)
            vis_gt = colormap_inv_depth(inv_gt)

            h, w, _ = img.shape
            canvas = np.zeros((h, w * 3, 3), dtype=np.uint8)
            canvas[:, 0:w] = img
            canvas[:, w:2*w] = vis_pred
            canvas[:, 2*w:3*w] = vis_gt

            vis_path = os.path.join(args.vis_dir, f"{i:04d}.png")
            cv2.imwrite(vis_path, cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

    mean_errors = np.array(errors).mean(0)
    print("\n--- Average Metrics on SCARED Test Set ---")
    print(f"AbsRel:   {mean_errors[0]:.4f}")
    print(f"SqRel:    {mean_errors[1]:.4f}")
    print(f"RMSE:     {mean_errors[2]:.4f}")
    print(f"RMSE_log: {mean_errors[3]:.4f}")
    print(f"a1:       {mean_errors[4]:.4f}")
    print(f"a2:       {mean_errors[5]:.4f}")
    print(f"a3:       {mean_errors[6]:.4f}")


if __name__ == "__main__":
    main()
