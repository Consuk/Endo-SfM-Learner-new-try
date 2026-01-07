from __future__ import absolute_import, division, print_function
import os
import argparse
import csv
import numpy as np
import cv2
from collections import defaultdict

import torch


try:
    from PIL import Image as PILImage
except Exception as e:
    raise ImportError("Pillow es requerido: pip install pillow") from e

def readlines(filename):
    """Lee un txt y devuelve una lista de líneas sin saltos de línea."""
    with open(filename, "r") as f:
        return f.read().splitlines()

# ===== Constantes/metas =====
STEREO_SCALE_FACTOR = 5.4
MIN_DEPTH = 1e-3
MAX_DEPTH = 150.0

MEAN = np.array([0.45, 0.45, 0.45], dtype=np.float32)
STD = np.array([0.225, 0.225, 0.225], dtype=np.float32)


def compute_errors(gt, pred):
    """Métricas estándar de Monodepth/EndoDepth."""
    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    rmse = (gt - pred) ** 2
    rmse = np.sqrt(rmse.mean())

    rmse_log = (np.log(gt) - np.log(pred)) ** 2
    rmse_log = np.sqrt(rmse_log.mean())

    abs_rel = np.mean(np.abs(gt - pred) / gt)
    sq_rel = np.mean(((gt - pred) ** 2) / gt)
    return abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


# ---------- Endo-SfM-Learner model loading ----------

def load_dispnet(load_weights_folder, resnet_layers, device):
    """
    Carga DispResNet de Endo-SfM-Learner desde una carpeta de checkpoints
    tipo ./checkpoints/scared_endo/11-09-03:58
    (dispnet_model_best.pth.tar, dispnet_checkpoint.pth.tar, etc.)
    """
    import models  # del repo Endo-SfM-Learner

    if not os.path.isdir(load_weights_folder):
        raise FileNotFoundError(f"Cannot find weights folder: {load_weights_folder}")

    # Candidatos típicos (como en eval_depth.py)
    disp_path = os.path.join(load_weights_folder, "dispnet_model_best.pth.tar")
    alt_disp_path = os.path.join(load_weights_folder, "dispnet_checkpoint.pth.tar")

    chosen_disp_path = None
    for cand in [disp_path, alt_disp_path]:
        if os.path.isfile(cand):
            chosen_disp_path = cand
            break

    # Fallback: cualquier archivo que contenga "disp" o "depth"
    if chosen_disp_path is None:
        for f in os.listdir(load_weights_folder):
            lf = f.lower()
            if "disp" in lf or "depth" in lf:
                chosen_disp_path = os.path.join(load_weights_folder, f)
                break

    if chosen_disp_path is None:
        raise FileNotFoundError(
            f"Could not find DispResNet weights in {load_weights_folder}"
        )

    print(f"-> Loading DispResNet weights from {chosen_disp_path}")

    disp_net = models.DispResNet(resnet_layers, False).to(device)
    weights = torch.load(chosen_disp_path, map_location=device)

    if isinstance(weights, dict) and "state_dict" in weights:
        weights = weights["state_dict"]

    # Cargar en modo flexible
    disp_net.load_state_dict(weights, strict=False)
    disp_net.eval()

    return disp_net


# ---------- Helpers para paths del split ----------

def _parse_split_line(line: str):
    """
    Soporta formato tipo:
        dataset3/keyframe4 390 l

    Devuelve:
        folder (ej: "dataset3/keyframe4"),
        frame_idx (int),
        side (str, ej: "l")
    """
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"Línea de split inválida: {line!r}")

    folder = parts[0]
    frame_idx = int(parts[1])
    side = parts[2] if len(parts) > 2 else "l"

    return folder, frame_idx, side



def _build_img_path(root, folder, frame_idx, png=False):
    """
    Construye la ruta real:
      <root>/<folder>/data/<frame>.<ext>
    donde folder puede ser "dataset3/keyframe4".
    """
    ext = ".png" if png else ".jpg"
    return os.path.join(root, folder, "data", f"{frame_idx}{ext}")

# ---------- Evaluación para una raíz de datos (una severidad) ----------

def evaluate_one_root(
    data_path_root,
    filenames,
    gt_depths,
    disp_net,
    png=False,
    disable_median_scaling=False,
    pred_depth_scale_factor=1.0,
    strict=False,
    device="cuda",
):
    """
    Evalúa una raíz (p.ej., .../brightness/severity_1/endovis_data) usando
    el modelo DispResNet de Endo-SfM-Learner entrenado en SCaRED.
    - data_path_root: carpeta base de las imágenes corruptas (termina en endovis_data)
    - filenames: líneas de test_files.txt (dataset keyframe frame side)
    - gt_depths: np.array de GT alineado con el split original
    """

    n = len(filenames)
    errors = []
    ratios = []
    missing = 0
    used_indices = 0

    for idx, line in enumerate(filenames):
        line = line.strip()
        if not line:
            continue

        try:
            folder, frame_idx, side = _parse_split_line(line)
            img_path = _build_img_path(data_path_root, folder, frame_idx, png=png)

        except Exception:
            missing += 1
            if strict:
                raise ValueError(f"[STRICT] Línea inválida en split: {line!r}")
            continue

        img_path = _build_img_path(data_path_root, folder, frame_idx, png=png)


        if not os.path.isfile(img_path):
            missing += 1
            if strict:
                raise FileNotFoundError(
                    f"[STRICT] Falta la imagen {img_path} para línea '{line}'"
                )
            continue

        # ---------- Cargar GT ----------
        if idx >= gt_depths.shape[0]:
            missing += 1
            if strict:
                raise IndexError(
                    f"[STRICT] No hay GT para idx={idx} (len(gt_depths)={gt_depths.shape[0]})"
                )
            continue

        gt_depth = gt_depths[idx].astype(np.float32)
        gt_h, gt_w = gt_depth.shape[:2]

        # ---------- Cargar & preprocesar imagen ----------
        orig_bgr = cv2.imread(img_path)
        if orig_bgr is None:
            missing += 1
            if strict:
                raise RuntimeError(f"[STRICT] No se pudo leer la imagen {img_path}")
            continue

        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        img = orig_rgb.astype(np.float32) / 255.0
        img -= MEAN
        img /= STD
        img_tensor = torch.from_numpy(img.transpose(2, 0, 1)).unsqueeze(0).to(device)

        # ---------- Predicción de disparidad ----------
        with torch.no_grad():
            pred_disp = disp_net(img_tensor)

        if isinstance(pred_disp, (list, tuple)):
            pred_disp = pred_disp[0]
        pred_disp = pred_disp.squeeze().cpu().numpy()

        # ---------- Convertir a depth & resize ----------
        print(pred_disp.min(), pred_disp.max(), pred_disp.mean())
        pred_disp[pred_disp <= 0] = 1e-6
        pred_depth = 1.0 / pred_disp  # profundidad relativa, como en eval_depth.py

        # resize a resolución de GT (vía inverse-depth para estabilidad)
        pred_inv_depth = cv2.resize(
            1.0 / (pred_depth + 1e-6),
            (gt_w, gt_h),
            interpolation=cv2.INTER_LINEAR,
        )
        pred_depth_resized = 1.0 / (pred_inv_depth + 1e-6)

        # ---------- Mask de válidos ----------
        mask = np.logical_and(gt_depth > MIN_DEPTH, gt_depth < MAX_DEPTH)
        if not np.any(mask):
            missing += 1
            if strict:
                raise RuntimeError(
                    f"[STRICT] GT sin píxeles válidos en {img_path}"
                )
            continue

        pd = pred_depth_resized[mask]
        gd = gt_depth[mask]

        # ---------- Escalado base (estéreo/mono) ----------
        if pred_depth_scale_factor != 1.0:
            pd *= pred_depth_scale_factor

        # ---------- Median scaling monocular ----------
        if not disable_median_scaling:
            median_pred = np.median(pd)
            if median_pred < 1e-6:
                missing += 1
                if strict:
                    raise RuntimeError(
                        f"[STRICT] Median predicted depth ~0 en {img_path}"
                    )
                continue
            ratio = np.median(gd) / (median_pred + 1e-6)
            ratios.append(ratio)
            pd *= ratio

        # ---------- Clipping y métricas ----------
        pd = np.clip(pd, MIN_DEPTH, MAX_DEPTH)
        gd = np.clip(gd, MIN_DEPTH, MAX_DEPTH)

        errors.append(compute_errors(gd, pd))
        used_indices += 1

    if used_indices == 0:
        mode = "STRICT" if strict else "LENIENT"
        raise FileNotFoundError(
            f"[{mode}] Ninguna muestra utilizable en {data_path_root} "
            f"(faltantes/errores: {missing}/{n})."
        )

    if (not strict) and missing > 0:
        print(
            f"   [INFO] {data_path_root}: usando {used_indices}/{n} frames del split "
            f"(faltaron {missing})."
        )

    if not disable_median_scaling and len(ratios) > 0:
        ratios = np.array(ratios)
        med = np.median(ratios)
        print(
            f"    Scaling ratios | med: {med:0.3f} | std: {np.std(ratios / med):0.3f}"
        )

    mean_errors = np.array(errors).mean(0)
    return mean_errors  # abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3


# ---------- Corrupciones / severidades ----------

def list_corruption_dirs(root):
    """
    Devuelve los directorios de primer nivel que representan corrupciones.
    Si 'root' ya es una carpeta de una corrupción (que contiene severity_*), la devuelve tal cual.
    """
    if not os.path.isdir(root):
        return []
    # Si ya hay severity_* dentro, root es una sola corrupción
    severities = [
        d
        for d in os.listdir(root)
        if os.path.isdir(os.path.join(root, d)) and d.startswith("severity_")
    ]
    if len(severities) > 0:
        return [root]
    # Si no, asumimos que root contiene muchas corrupciones como subcarpetas
    return [
        os.path.join(root, d)
        for d in sorted(os.listdir(root))
        if os.path.isdir(os.path.join(root, d))
    ]


def main():
    parser = argparse.ArgumentParser(
        "Evaluate EndoVIS corruptions (16x5) with Endo-SfM-Learner DispResNet weights"
    )
    parser.add_argument(
        "--corruptions_root",
        type=str,
        required=True,
        help="Raíz de las corrupciones (o una sola corrupción). "
             "Ej: /workspace/endovis_corruptions_test",
    )
    parser.add_argument(
        "--load_weights_folder",
        type=str,
        required=True,
        help="Carpeta de checkpoints con dispnet_model_best.pth.tar, etc.",
    )
    parser.add_argument(
        "--splits_dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "splits"),
    )
    parser.add_argument(
        "--split",
        type=str,
        default="endovis",
        help="Nombre del split (carpeta dentro de splits/)",
    )
    parser.add_argument("--resnet_layers", type=int, default=18)
    parser.add_argument("--png", action="store_true", help="Usa .png en lugar de .jpg")
    parser.add_argument(
        "--eval_stereo",
        action="store_true",
        help="Forzar estéreo (desactiva median scaling y usa x5.4)",
    )
    parser.add_argument(
        "--output_csv", type=str, default="corruptions_summary_endosfm.csv"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Modo estricto: exige que todas las entradas del split existan en cada severidad.",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    cv2.setNumThreads(0)

    # --------- Cargar split y GTs ---------
    test_files_path = os.path.join(args.splits_dir, args.split, "test_files.txt")
    if not os.path.isfile(test_files_path):
        raise FileNotFoundError(f"No se encontró test_files.txt en {test_files_path}")

    test_files = readlines(test_files_path)

    gt_path = os.path.join(args.splits_dir, args.split, "gt_depths.npz")
    if not os.path.isfile(gt_path):
        raise FileNotFoundError(f"No se encontró gt_depths.npz en {gt_path}")

    gt_npz = np.load(gt_path, fix_imports=True, encoding="latin1")
    # asume que la clave es "data" (adáptalo si tu npz usa otra)
    if "data" in gt_npz.files:
        gt_depths = gt_npz["data"]
    elif "depths" in gt_npz.files:
        gt_depths = gt_npz["depths"]
    else:
        # primera entrada
        gt_depths = gt_npz[gt_npz.files[0]]

    if len(test_files) != gt_depths.shape[0]:
        print(
            "[WARN] test_files.txt y gt_depths.npz difieren en longitud. "
            "El script asume alineación por índice; asegúrate de que el npz se "
            "generó usando ese split."
        )

    # --------- Config mono/estéreo ---------
    disable_median_scaling = args.eval_stereo
    pred_depth_scale_factor = STEREO_SCALE_FACTOR if args.eval_stereo else 1.0

    # --------- Cargar modelo ---------
    print("-> Cargando DispResNet pesos desde:", args.load_weights_folder)
    disp_net = load_dispnet(args.load_weights_folder, args.resnet_layers, device)

    # --------- Detectar corrupciones ---------
    corr_dirs = list_corruption_dirs(args.corruptions_root)
    if len(corr_dirs) == 0:
        raise FileNotFoundError(
            f"No se encontraron carpetas de corrupción en {args.corruptions_root}"
        )

    rows = []
    print("-> Iniciando evaluación de corrupciones")
    for corr_dir in corr_dirs:
        corr_name = os.path.basename(corr_dir.rstrip("/"))
        severities = sorted(
            [
                d
                for d in os.listdir(corr_dir)
                if os.path.isdir(os.path.join(corr_dir, d))
                and d.startswith("severity_")
            ],
            key=lambda s: int(s.split("_")[-1])
            if s.split("_")[-1].isdigit()
            else 9999,
        )

        for sev in severities:
            data_root = os.path.join(corr_dir, sev, "endovis_data")
            print(f"\n>> {corr_name} / {sev} :: data_path = {data_root}")
            if not os.path.isdir(data_root):
                print(f"   [WARN] No existe {data_root}, se omite.")
                continue

            try:
                mean_errors = evaluate_one_root(
                    data_path_root=data_root,
                    filenames=test_files,
                    gt_depths=gt_depths,
                    disp_net=disp_net,
                    png=args.png,
                    disable_median_scaling=disable_median_scaling,
                    pred_depth_scale_factor=pred_depth_scale_factor,
                    strict=args.strict,
                    device=device,
                )
                abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 = mean_errors.tolist()
                rows.append(
                    [corr_name, sev, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3]
                )

                print(
                    "   Métricas (promedio): "
                    f"abs_rel={abs_rel:.3f} | sq_rel={sq_rel:.3f} | rmse={rmse:.3f} | "
                    f"rmse_log={rmse_log:.3f} | a1={a1:.3f} | a2={a2:.3f} | a3={a3:.3f}"
                )

            except FileNotFoundError as e:
                print(f"   [SKIP] {e}")

    # --------- Guardar CSV y resumen ---------
    if rows:
        header = [
            "corruption",
            "severity",
            "abs_rel",
            "sq_rel",
            "rmse",
            "rmse_log",
            "a1",
            "a2",
            "a3",
        ]
        with open(args.output_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            for r in rows:
                w.writerow(r)

        print(f"\n-> Resumen guardado en: {args.output_csv}")

        bucket = defaultdict(list)
        for r in rows:
            bucket[r[0]].append(r)

        print("\n======= RESUMEN (por corrupción) =======")
        for corr in sorted(bucket.keys()):
            print(f"\n{corr}")
            print(
                "severity | abs_rel |  sq_rel |  rmse  | rmse_log |   a1   |   a2   |   a3"
            )
            for _, sev, abs_rel, sq_rel, rmse, rmse_log, a1, a2, a3 in sorted(
                bucket[corr],
                key=lambda x: int(x[1].split("_")[-1])
                if x[1].split("_")[-1].isdigit()
                else 9999,
            ):
                print(
                    f"{sev:>9} | {abs_rel:7.3f} | {sq_rel:7.3f} | {rmse:7.3f} |  {rmse_log:7.3f} | "
                    f"{a1:6.3f} | {a2:6.3f} | {a3:6.3f}"
                )
    else:
        print(
            "\n-> No se generaron filas. Revisa rutas/archivos faltantes o estructura de corrupciones."
        )


if __name__ == "__main__":
    main()
