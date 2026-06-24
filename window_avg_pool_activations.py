"""Llama の単語単位 activations を「時間窓 avg pooling」して、音楽側 (analysis) の
encoding パイプラインと同じ NPZ 形式に変換する。

MindTransformer の本来の Mode1 は単語 activation を HRF 畳み込みして fMRI に整合させるが、
本研究では音楽側(GTZAN)と同じ枠組み（時間窓 avg pooling + 固定 delay）に揃えたい。
そこで、既に抽出済みの Llama 単語 activations と各単語の onset/offset を使い、
各 run を一定の窓幅・ストライドでスライスし、窓内に onset が入る単語を avg pooling して
「窓ごとに 1 ベクトル」にする。

出力は音楽側 emb_loader が読むのと同じ NPZ:
  keys: 窓ID 配列（例 "lppEN_run1_0.00_10.00"）
  vecs: (n_windows, dim) の avg-pooled embedding
layer × state ごとに 1 ファイル（series 名 = "llama-layer<L>-<state>"）。

入力:
  outputs/lpp_llms_activations/<sanitized_model>_lpp_en_run-run-<R>_part-*_activations.gz
    = dict{state: ndarray(n_layers, n_words, dim)}   ※ part が複数なら単語方向に連結
  outputs/lpp_llms_activations/onsets_offsets_lpp_en.gz
    = list[run] of tuple(onsets(n_words,), offsets(n_words,))   ※秒

出力:
  <out_root>/<model_key>/lpp_en-emb-window<W>s-stride<S>s-<model_key>-avg-<series>.npz

使い方（mindtransformer_env, cwd=external/MindTransformer）:
  python window_avg_pool_activations.py \
      --model meta-llama/Llama-3.2-1B-Instruct \
      --model_key llama-3.2-1b \
      --window_s 10 --stride_s 2 \
      --out_root /gpudata/ssd1/h-sato/fmri2music-alt/data/speech/lpp-emb
"""
import argparse
import glob
import os
import re

import numpy as np
import joblib

ACT_DIR = "outputs/lpp_llms_activations"


def sanitize_model(model_name):
    return model_name.replace("/", "_")


def load_run_activations(model_name, run_index_1based):
    """run の全 part を読み、dict{state: (n_layers, n_words_total, dim)} を返す。
    part は単語方向(axis=1)に連結する。"""
    sm = sanitize_model(model_name)
    pattern = os.path.join(
        ACT_DIR, f"{sm}_lpp_en_run-run-{run_index_1based}_part-*_activations.gz"
    )
    paths = sorted(
        glob.glob(pattern),
        key=lambda p: int(re.search(r"part-(\d+)", p).group(1)),
    )
    if not paths:
        raise FileNotFoundError(f"No activation files for run {run_index_1based}: {pattern}")
    parts = [joblib.load(p) for p in paths]
    states = list(parts[0].keys())
    merged = {}
    for state in states:
        arrs = [np.asarray(p[state]) for p in parts]
        merged[state] = np.concatenate(arrs, axis=1)  # (n_layers, n_words_total, dim)
    return merged


def make_windows(run_duration_s, window_s, stride_s):
    """[0, run_duration] を window_s 幅・stride_s 刻みでスライスした (start, end) のリスト。
    末尾は run 終端を超えない範囲。"""
    windows = []
    start = 0.0
    while start + window_s <= run_duration_s + 1e-9:
        windows.append((start, start + window_s))
        start += stride_s
    # 末尾に端数が残り、かつ窓が1つも無い場合の保険
    if not windows and run_duration_s > 0:
        windows.append((0.0, run_duration_s))
    return windows


def window_avg_pool_one_layer_state(layer_word_vecs, onsets, win_start, win_end):
    """窓 [win_start, win_end) に onset が入る単語ベクトルを平均。
    layer_word_vecs: (n_words, dim) / onsets: (n_words,)
    窓内に単語が無ければ None。"""
    mask = (onsets >= win_start) & (onsets < win_end)
    if not np.any(mask):
        return None
    return layer_word_vecs[mask].mean(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--model_key", default="llama-3.2-1b",
                        help="出力ディレクトリ・series 名に使う短いキー")
    parser.add_argument("--window_s", type=float, default=10.0)
    parser.add_argument("--stride_s", type=float, default=2.0)
    parser.add_argument("--out_root", required=True,
                        help="出力ルート（例 data/speech/lpp-emb）")
    parser.add_argument("--states", nargs="*", default=None,
                        help="抽出する state（既定: activations にある全 state）")
    args = parser.parse_args()

    # onsets/offsets（run別）
    oo = joblib.load(os.path.join(ACT_DIR, "onsets_offsets_lpp_en.gz"))
    n_runs = len(oo)
    print(f"runs: {n_runs}")

    out_dir = os.path.join(args.out_root, args.model_key)
    os.makedirs(out_dir, exist_ok=True)

    # window/stride のラベル（音楽の "window10s-stride1_5s" に倣い、小数点は _ に）
    def fmt(x):
        return (f"{x:g}").replace(".", "_")
    win_label = f"window{fmt(args.window_s)}s-stride{fmt(args.stride_s)}s"

    # series ごとに「全 run の窓」を貯めて 1 NPZ に保存する。
    # series = "llama-layer<L>-<state>"。L は 0..n_layers-1。
    # まず全 run を走査して、(state, layer) -> {keys:[], vecs:[]} を構築。
    accum = {}  # (state, layer) -> (keys list, vecs list)
    states_to_use = args.states

    for run_i in range(n_runs):
        run_1 = run_i + 1
        onsets = np.asarray(oo[run_i][0], dtype=float)   # (n_words,)
        offsets = np.asarray(oo[run_i][1], dtype=float)
        run_duration = float(offsets.max())
        acts = load_run_activations(args.model, run_1)
        if states_to_use is None:
            states_to_use = list(acts.keys())
        # 単語数の整合チェック
        any_state = states_to_use[0]
        n_layers, n_words, _ = np.asarray(acts[any_state]).shape
        if n_words != len(onsets):
            raise ValueError(
                f"run{run_1}: activation 単語数 {n_words} != onsets 数 {len(onsets)}"
            )
        windows = make_windows(run_duration, args.window_s, args.stride_s)
        print(f"  run{run_1}: dur={run_duration:.1f}s, words={n_words}, windows={len(windows)}")

        for state in states_to_use:
            arr = np.asarray(acts[state])  # (n_layers, n_words, dim)
            for layer in range(arr.shape[0]):
                lw = arr[layer]  # (n_words, dim)
                key0 = (state, layer)
                if key0 not in accum:
                    accum[key0] = ([], [])
                for (ws, we) in windows:
                    vec = window_avg_pool_one_layer_state(lw, onsets, ws, we)
                    if vec is None:
                        continue  # 単語が無い窓はスキップ（fMRI 側もこの窓IDは作らない）
                    win_id = f"lppEN_run{run_1}_{ws:.2f}_{we:.2f}"
                    accum[key0][0].append(win_id)
                    accum[key0][1].append(vec.astype(np.float32))

    # 保存
    n_saved = 0
    for (state, layer), (keys, vecs) in accum.items():
        if not keys:
            continue
        series = f"llama-layer{layer}-{state}"
        fname = f"lpp_en-emb-{win_label}-{args.model_key}-avg-{series}.npz"
        out_path = os.path.join(out_dir, fname)
        np.savez(
            out_path,
            keys=np.array(keys, dtype=object),
            vecs=np.stack(vecs, axis=0),
        )
        n_saved += 1
    print(f"saved {n_saved} npz files to {out_dir}")


if __name__ == "__main__":
    main()
