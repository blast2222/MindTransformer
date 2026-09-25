"""compute_average_subject_fmri.py が出す被験者平均 fMRI(.gz)を、音楽側 encoding と
同じデータ契約の run 別 .npy に書き出す。

音楽側の Resp_*.npy は (voxel, TR) 向きで、analysis 側(lpp_encoding.py の load_run_resp)が
.T して (TR, voxel) に直して使う。LPP も同じ向き・命名に揃えるための転置エクスポート。

入力: outputs/lpp_en_average_subject/average_subject_run-<i>.gz (i=0..n_runs-1)
    = (TR, voxel) float32。compute_average_subject_fmri.py が run ごとに standardize(axis=0)
      済みで保存(z-score)。trim/delay/窓平均は未(それらは analysis/lpp_encoding.py が実行時に行う)。
出力: <repo>/data/speech/resp/lpp_en_avg-subject_run-<N>.npy (N=i+1, 1始まり)
    = (voxel, TR) float32。standardize 済みのまま、転置するだけで値は変えない。

window_align_fmri.py(窓IDキーで Y を固める方式)は音楽の流儀(生 resp.npy + 実行時窓平均)と
非互換のため不採用で、本スクリプトが現行の Y 生成経路。

使い方(mindtransformer_env, cwd=external/MindTransformer):
  python export_average_subject_to_resp.py
  # 出力先や入力を変える場合:
  python export_average_subject_to_resp.py \
      --avg_dir outputs/lpp_en_average_subject \
      --out_dir ../../data/speech/resp
"""
import argparse
import glob
import os
import re

import joblib
import numpy as np

# external/MindTransformer から見たリポジトリルート(1 つ上の external の、さらに 1 つ上)
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
        run_1 = run_0 + 1  # .npy は 1 始まり
        arr = np.asarray(joblib.load(path), dtype=np.float32)  # (TR, voxel)
        out = os.path.join(args.out_dir, f"lpp_en_avg-subject_run-{run_1}.npy")
        np.save(out, arr.T)  # (voxel, TR) で保存(音楽 Resp_*.npy と同じ向き)
        print(f"run{run_1}: (TR,voxel)={arr.shape} -> saved (voxel,TR)={arr.T.shape} {out}")


if __name__ == "__main__":
    main()
