"""Export average-subject fMRI as one (voxel, TR) .npy file per run.

Run from the MindTransformer directory:
  python export_average_subject_to_resp.py
"""
import argparse
import glob
import os
import re

import joblib
import numpy as np

# Repository root from external/MindTransformer.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_AVG_DIR = "outputs/lpp_en_average_subject"
DEFAULT_OUT_DIR = os.path.join(REPO_ROOT, "data", "speech", "resp")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--avg_dir", default=DEFAULT_AVG_DIR,
                        help="average_subject_run-<i>.gz のディレクトリ")
    parser.add_argument("--out_dir", default=DEFAULT_OUT_DIR,
                        help="run 別 .npy の出力先(既定 data/speech/resp)")
    args = parser.parse_args()

    paths = sorted(
        glob.glob(os.path.join(args.avg_dir, "average_subject_run-*.gz")),
        key=lambda p: int(re.search(r"run-(\d+)", p).group(1)),
    )
    if not paths:
        raise FileNotFoundError(f"No average_subject_run-*.gz in {args.avg_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    for path in paths:
        run_0 = int(re.search(r"run-(\d+)", path).group(1))
        run_1 = run_0 + 1  # Output files use one-based run indices.
        arr = np.asarray(joblib.load(path), dtype=np.float32)  # (TR, voxel)
        out = os.path.join(args.out_dir, f"lpp_en_avg-subject_run-{run_1}.npy")
        np.save(out, arr.T)  # Save as (voxel, TR).
        print(f"run{run_1}: (TR,voxel)={arr.shape} -> saved (voxel,TR)={arr.T.shape} {out}")


if __name__ == "__main__":
    main()
