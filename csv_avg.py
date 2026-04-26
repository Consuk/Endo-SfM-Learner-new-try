import argparse
import os

import pandas as pd


def resolve_csv_path(eval_corruptions_root, csv_name, csv_path):
    if csv_path:
        if os.path.isabs(csv_path):
            return csv_path
        if eval_corruptions_root:
            return os.path.join(eval_corruptions_root, csv_path)
        return csv_path

    if eval_corruptions_root:
        return os.path.join(eval_corruptions_root, csv_name)

    return csv_name


def main():
    parser = argparse.ArgumentParser(
        description="Compute global/per-corruption averages from corruption evaluation CSV."
    )
    parser.add_argument(
        "--eval_corruptions_root",
        type=str,
        default=None,
        help="Root directory used for corruption evaluation. If set, relative CSV paths are resolved from here.",
    )
    parser.add_argument(
        "--csv_name",
        type=str,
        default="corruptions_summary_endosfm.csv",
        help="CSV filename when using --eval_corruptions_root and --csv_path is not given.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default=None,
        help="Optional explicit CSV path (absolute or relative). Overrides --csv_name.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default=None,
        help="Output CSV for per-corruption averages. Defaults next to input CSV as corruption_averages.csv.",
    )
    args = parser.parse_args()

    csv_path = resolve_csv_path(args.eval_corruptions_root, args.csv_name, args.csv_path)
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)
    metric_cols = ["abs_rel", "sq_rel", "rmse", "rmse_log", "a1", "a2", "a3"]

    missing_cols = [c for c in metric_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing metric columns in CSV: {missing_cols}")

    global_means = df[metric_cols].mean()

    print("=== Promedio Global de Todas las Corrupciones y Severidades ===")
    for metric, value in global_means.items():
        print(f"{metric:10s}: {value:.3f}")

    if "corruption" in df.columns:
        avg_per_corr = df.groupby("corruption")[metric_cols].mean().reset_index()
        print("\n=== Promedio por tipo de corrupción ===")
        print(avg_per_corr)

        out_csv = args.out_csv
        if out_csv is None:
            out_csv = os.path.join(os.path.dirname(os.path.abspath(csv_path)), "corruption_averages.csv")

        avg_per_corr.to_csv(out_csv, index=False)
        print(f"\nPromedios por corrupción guardados en '{out_csv}'")


if __name__ == "__main__":
    main()
