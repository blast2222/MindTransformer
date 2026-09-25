"""Pool word-level LLM activations into time windows and save one NPZ per layer and state.

Run from the MindTransformer directory:
  python window_avg_pool_activations.py \
      --model meta-llama/Llama-3.2-1B-Instruct \
      --model_key llama-3.2-1b \
      --window_s 10 --stride_s 2 \
      --out_root ../../data/speech/speech-emb
"""
import argparse
import glob
import os
import re

import numpy as np
import joblib

ACT_DIR = "outputs/lpp_llms_activations"

# Map MindTransformer hook states to output series names.
# input_hidden_state duplicates the previous layer's block-output and is omitted.
STATE_RENAME = {
    "pre_attn_norm":            "pre-attn-norm",
    "per_head_q":               "q",
    "per_head_k":               "k",
    "per_head_v":               "v",
    "per_head_q_rope":          "q-with-rope",
    "per_head_k_rope":          "k-with-rope",
    "attn_output":              "attn-out",
    "per_head_context_vector": "context",
    "post_attn_hidden_state":   "post-attn-hidden",
    "pre_ffn_norm":             "pre-ffn-norm",
    "ffn_activated_state":      "ffn-activated",
    "ffn_output":               "ffn-output",
    "final_block_output":       "block-output",
}


def sanitize_model(model_name):
    return model_name.replace("/", "_")


def load_run_activations(model_name, run_index_1based):
    """Load all activation parts for a run and concatenate them along the word axis."""
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
    """Return full windows of length window_s at stride_s over the run."""
    windows = []
    start = 0.0
    while start + window_s <= run_duration_s + 1e-9:
        windows.append((start, start + window_s))
        start += stride_s
    # Keep a single partial window for runs shorter than window_s.
    if not windows and run_duration_s > 0:
        windows.append((0.0, run_duration_s))
    return windows


def window_avg_pool_one_layer_state(layer_word_vecs, onsets, win_start, win_end):
    """Average word vectors with onsets in [win_start, win_end), or return None."""
    mask = (onsets >= win_start) & (onsets < win_end)
    if not np.any(mask):
        return None
    return layer_word_vecs[mask].mean(axis=0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--model_key", default="llama-3.2-1b",
                        help="出力ディレクトリ・NPZ ファイル名に使う短いキー")
    parser.add_argument("--series_prefix", default=None,
                        help="series 名の prefix（例: llama, gemma, mistral, qwen, gpt-oss）。"
                             "省略時は model_key の最初のハイフンより前を使う（llama-3.2-1b → llama）。")
    parser.add_argument("--window_s", type=float, default=10.0)
    parser.add_argument("--stride_s", type=float, default=2.0)
    parser.add_argument("--out_root", required=True,
                        help="出力ルート（例 data/speech/speech-emb）")
    parser.add_argument("--states", nargs="*", default=None,
                        help="抽出する MindTransformer 内部 state 名（既定: STATE_RENAME 全キー）")
    args = parser.parse_args()

    series_prefix = args.series_prefix or args.model_key.split("-", 1)[0]

    # Per-run onsets and offsets.
    oo = joblib.load(os.path.join(ACT_DIR, "onsets_offsets_lpp_en.gz"))
    n_runs = len(oo)
    print(f"runs: {n_runs}")

    out_dir = os.path.join(args.out_root, args.model_key)
    os.makedirs(out_dir, exist_ok=True)

    # Use underscores for decimal points in window and stride labels.
    def fmt(x):
        return (f"{x:g}").replace(".", "_")
    win_label = f"window{fmt(args.window_s)}s-stride{fmt(args.stride_s)}s"

    # Accumulate windows across runs for each renamed state and layer.
    # Output series names use kebab-case state names.
    accum = {}  # (new_state, layer) -> (keys list, vecs list)
    states_to_use = args.states  # None selects all STATE_RENAME keys.

    for run_i in range(n_runs):
        run_1 = run_i + 1
        onsets = np.asarray(oo[run_i][0], dtype=float)   # (n_words,)
        offsets = np.asarray(oo[run_i][1], dtype=float)
        run_duration = float(offsets.max())
        acts = load_run_activations(args.model, run_1)
        if states_to_use is None:
            # Use states available in both STATE_RENAME and the activations.
            states_to_use = [k for k in STATE_RENAME.keys() if k in acts]
            skipped = [k for k in acts.keys() if k not in STATE_RENAME]
            if skipped:
                print(f"  (skip: STATE_RENAME に無い state を出力しない) {sorted(skipped)}")
            missing = [k for k in STATE_RENAME.keys() if k not in acts]
            if missing:
                print(f"  (info: activation に無い state) {sorted(missing)}")
        # Check word-count consistency.
        if not states_to_use:
            requested = list(STATE_RENAME) if args.states is None else args.states
            raise ValueError(
                f"run{run_1}: requested states are absent from activation files. "
                f"Requested: {requested}; available: {sorted(acts)}"
            )
        missing_states = [state for state in states_to_use if state not in acts]
        if missing_states:
            raise ValueError(
                f"run{run_1}: requested states are missing from activation files: "
                f"{missing_states}; available: {sorted(acts)}"
            )

        any_state = states_to_use[0]
        n_layers, n_words, _ = np.asarray(acts[any_state]).shape
        if n_words != len(onsets):
            raise ValueError(
                f"run{run_1}: activation 単語数 {n_words} != onsets 数 {len(onsets)}"
            )
        windows = make_windows(run_duration, args.window_s, args.stride_s)
        print(f"  run{run_1}: dur={run_duration:.1f}s, words={n_words}, windows={len(windows)}")

        for state in states_to_use:
            new_state = STATE_RENAME[state]  # Map an internal key to an output state name.
            arr = np.asarray(acts[state])  # (n_layers, n_words, dim)
            for layer in range(arr.shape[0]):
                lw = arr[layer]  # (n_words, dim)
                key0 = (new_state, layer)
                if key0 not in accum:
                    accum[key0] = ([], [])
                for (ws, we) in windows:
                    vec = window_avg_pool_one_layer_state(lw, onsets, ws, we)
                    if vec is None:
                        continue  # Skip windows without words.
                    win_id = f"lppEN_run{run_1}_{ws:.2f}_{we:.2f}"
                    accum[key0][0].append(win_id)
                    accum[key0][1].append(vec.astype(np.float32))

    # Save results.
    n_saved = 0
    for (new_state, layer), (keys, vecs) in accum.items():
        if not keys:
            continue
        series = f"{series_prefix}-layer{layer}-{new_state}"
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
