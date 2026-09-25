"""MindTransformer Mode1 の結果（layer x state ごとの voxel別cc）を、
論文 (ICLR2026, MindTransformer) の Appendix E に厳密準拠して集計・作図する。

再現する図:
  - Figure 3: Weighted Computational Depth vs Cortical Hierarchy
      x軸 = 皮質階層 HG -> PT -> STG -> MTG -> IFG -> AG（Harvard-Oxford atlas）
      y軸 = Weighted Computational Depth D̄_R（式2）。中核8状態subset S（式1, δ=(i-1)/7）。
      全層平均。auditory stream(HG->MTG) に線形フィット (slope/R^2)。
  - Figure 1b: Winning ratio（13状態の勝者割合）を auditory / language / whole-brain 別に。

計算ロジックは本研究の plot_computational_depth_by_roi.py / state_winning_ratio.py と同型
（voxelごとに cc最大の状態を argmax -> 深さ割当 -> ROI平均）。違いは入力が corr.gz、
ROI が Harvard-Oxford、state subset が論文の8状態、layer集計が全層平均である点。

使い方（mindtransformer_env, cwd=MindTransformerディレクトリ）:
  python analyze_mindtransformer_mode1.py \
      --config config_lpp_llama.yaml \
      --model meta-llama/Llama-3.2-1B-Instruct

cd external/MindTransformer  # リポジトリルートから実行
micromamba activate mindtransformer_env   # または下のフルパス
python analyze_mindtransformer_mode1.py \
    --config config_lpp_llama.yaml \
    --model meta-llama/Llama-3.2-1B-Instruct

Harvard-Oxford atlas は初回実行時に Nilearn が取得する（ネットワーク必須）。
`--atlas_data_dir` でキャッシュ先を固定できる。オフライン実行前には、同じキャッシュ先を
指定してネットワーク接続下で一度実行する。
"""
import argparse
import glob
import os
import re

import numpy as np
import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nilearn.input_data import NiftiMasker
from nilearn.datasets import fetch_atlas_harvard_oxford

from config_utils import load_config, make_dir


# --- 論文 Appendix E の中核8状態subset S（順序付き、δ=(i-1)/(|S|-1)） ---
# config の13状態名 -> 論文の表示ラベル
DEPTH_SUBSET_STATES = [
    ("per_head_q", "Per-head Query"),
    ("per_head_q_rope", "Per-head Query w/ RoPE"),
    ("per_head_context_vector", "Per-head Context Vector"),
    ("attn_output", "Combined Attention Output"),
    ("post_attn_hidden_state", "Post-Attention Hidden State"),
    ("pre_ffn_norm", "Pre-FFN Normalized State"),
    ("ffn_activated_state", "FFN Activated State"),
    ("ffn_output", "FFN Output"),
]

# winning ratio 用の全13状態（論文 Fig1b / Appendix の並び。浅い->深い）
ALL_STATES = [
    "input_hidden_state",
    "pre_attn_norm",
    "per_head_q",
    "per_head_k",
    "per_head_q_rope",
    "per_head_k_rope",
    "per_head_v",
    "per_head_context_vector",
    "attn_output",
    "post_attn_hidden_state",
    "pre_ffn_norm",
    "ffn_activated_state",
    "ffn_output",
]
STATE_DISPLAY = {
    "input_hidden_state": "Input Hidden State",
    "pre_attn_norm": "Pre-Attention Normalized State",
    "per_head_q": "Per-Head Query (Q)",
    "per_head_k": "Per-Head Key (K)",
    "per_head_q_rope": "Per-Head Q with RoPE",
    "per_head_k_rope": "Per-Head K with RoPE",
    "per_head_v": "Per-Head Value",
    "per_head_context_vector": "Per-Head Context Vector",
    "attn_output": "Combined Attention Output",
    "post_attn_hidden_state": "Post-Attention Hidden State",
    "pre_ffn_norm": "Pre-FFN Normalized State",
    "ffn_activated_state": "FFN Activated State",
    "ffn_output": "FFN Output",
}

# Harvard-Oxford (cort-maxprob-thr25-1mm) の label index で定義した
# 皮質階層 ROI（論文 Fig3 / Table10 準拠、低次感覚 -> 高次連合の順）
CORTICAL_HIERARCHY = [
    ("HG", [45]),  # Heschl's Gyrus
    ("PT", [46]),  # Planum Temporale
    ("STG", [9, 10]),  # Superior Temporal Gyrus (ant/post)
    ("MTG", [11, 12, 13]),  # Middle Temporal Gyrus (ant/post/temp-occ)
    ("IFG", [5, 6]),  # Inferior Frontal Gyrus (triang/operc)
    ("AG", [21]),  # Angular Gyrus
]
# 線形フィット区間（論文: auditory stream HG->MTG が急勾配）
AUDITORY_STREAM_LABELS = ["HG", "PT", "STG", "MTG"]
# winning ratio の集計対象グループ（whole-brain は全mask voxel）
AUDITORY_GROUP_LABELS = ["HG", "PT", "STG"]
LANGUAGE_GROUP_LABELS = ["MTG", "IFG", "AG"]


def load_cc_by_layer_state(corr_dir, model_name):
    """corr.gz を読み {(layer,state): cc配列(n_voxels,)} を返す。"""
    sanitized = model_name.replace("/", "_")
    pattern = os.path.join(
        corr_dir, f"{sanitized}_*_layer-*_inputs-*_corr.gz"
    )
    data = {}
    for path in sorted(glob.glob(pattern)):
        m = re.search(r"layer-(\d+)_inputs-([a-z0-9_]+)_corr", os.path.basename(path))
        if not m:
            continue
        layer = int(m.group(1))
        state = m.group(2)
        data[(layer, state)] = np.asarray(joblib.load(path), dtype=float)
    return data


def get_layers_states(data):
    layers = sorted({l for l, _ in data})
    states = sorted({s for _, s in data})
    return layers, states


def build_roi_voxel_masks(mask_path, atlas_data_dir=None):
    """Harvard-Oxford atlas を mask 空間へ射影し、ROI -> bool voxel mask を返す。
    mindtransformer.py の parcel masking と同じ流儀（masker.transform(atlas.maps)）。
    atlas_data_dir は Nilearn の atlas キャッシュ先。None なら Nilearn の既定キャッシュを使う。
    """
    masker = NiftiMasker(mask_img=mask_path)
    masker.fit()
    n_voxels = masker.n_elements_
    atlas = fetch_atlas_harvard_oxford(
        "cort-maxprob-thr25-1mm", data_dir=atlas_data_dir
    )
    atlas_1d = masker.transform(atlas.maps).flatten().astype(int)
    roi_masks = {}
    for roi_label, label_indices in CORTICAL_HIERARCHY:
        roi_masks[roi_label] = np.isin(atlas_1d, label_indices)
    return roi_masks, n_voxels, atlas_1d


def compute_weighted_depth(data, layers, roi_voxel_mask):
    """論文 式1/式2。ROI内 voxel について、各layerで subset S の cc最大状態のδを
    voxel平均 -> layer平均（全層平均）した Weighted Computational Depth を返す。

    本研究 compute_roi_depth と同型: layerごとに voxel別 argmax(state) -> δ -> 平均、
    それを全層平均。
    """
    subset_keys = [k for k, _ in DEPTH_SUBSET_STATES]
    n_s = len(subset_keys)
    deltas = np.linspace(0.0, 1.0, n_s)  # δ(s_i) = (i-1)/(|S|-1)
    layer_depths = []
    for layer in layers:
        # subset 8状態が全て揃っている layer のみ対象
        if not all((layer, s) in data for s in subset_keys):
            continue
        # (n_s, n_roi_voxels) の cc 行列
        cc_stack = np.stack(
            [data[(layer, s)][roi_voxel_mask] for s in subset_keys], axis=0
        )
        if cc_stack.shape[1] == 0:
            continue
        winner = np.argmax(cc_stack, axis=0)  # 各 voxel の勝者状態 index
        layer_depths.append(float(np.mean(deltas[winner])))
    if not layer_depths:
        return float("nan"), 0
    return float(np.mean(layer_depths)), len(layer_depths)


def linear_fit(x, y):
    """0〜1正規化した x に対する slope/intercept/R^2。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = x[finite], y[finite]
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    xn = np.linspace(0.0, 1.0, len(x))
    slope, intercept = np.polyfit(xn, y, deg=1)
    y_pred = slope * xn + intercept
    ss_res = float(np.sum((y - y_pred) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float("nan") if ss_tot == 0.0 else 1.0 - ss_res / ss_tot
    return float(slope), float(intercept), r2


def compute_winning_ratio(data, layers, voxel_mask):
    """13状態全部を競わせ、voxelごと cc最大状態を全layer集計した勝者割合(%)。
    voxel_mask: 集計対象 voxel の bool。論文 Fig1b 相当。
    """
    counts = {s: 0 for s in ALL_STATES}
    total = 0
    for layer in layers:
        if not all((layer, s) in data for s in ALL_STATES):
            continue
        cc_stack = np.stack(
            [data[(layer, s)][voxel_mask] for s in ALL_STATES], axis=0
        )
        if cc_stack.shape[1] == 0:
            continue
        winner = np.argmax(cc_stack, axis=0)
        for idx, s in enumerate(ALL_STATES):
            counts[s] += int(np.sum(winner == idx))
        total += cc_stack.shape[1]
    if total == 0:
        return {s: float("nan") for s in ALL_STATES}, 0
    return {s: 100.0 * counts[s] / total for s in ALL_STATES}, total


# ---------------- plotting ----------------
def plot_computational_depth(depth_by_roi, fit, out_path, model_name):
    roi_labels = [r for r, _ in CORTICAL_HIERARCHY]
    y = [depth_by_roi[r] for r in roi_labels]
    x = np.arange(len(roi_labels))
    fig, ax = plt.subplots(figsize=(7, 5))
    # 本研究 plot_computational_depth_by_roi.py に忠実に: 折線は描かず点(scatter)と
    # fit 線だけにする。点は #1f77b4、fit 線は #d62728。
    ax.scatter(x, y, color="#1f77b4", s=58, zorder=3)
    for xi, yi in zip(x, y):
        ax.annotate(f"{yi:.2f}", (xi, yi), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=10)
    # auditory stream の線形フィット線
    fit_x = [i for i, r in enumerate(roi_labels) if r in AUDITORY_STREAM_LABELS]
    if fit is not None and len(fit_x) >= 2:
        slope, intercept, r2 = fit
        xn = np.linspace(0.0, 1.0, len(fit_x))
        fit_y = slope * xn + intercept
        ax.plot(fit_x, fit_y, color="#d62728", linewidth=1.8, zorder=2,
                label=f"auditory fit (HG→MTG)\nslope={slope:.3f}, R²={r2:.3f}")
        ax.legend(loc="upper left", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(roi_labels)
    ax.set_ylim(-0.05, 1.05)
    yt = np.linspace(0.0, 1.0, len(DEPTH_SUBSET_STATES))
    ax.set_yticks(yt)
    ax.set_yticklabels([lbl for _, lbl in DEPTH_SUBSET_STATES], fontsize=9)
    ax.set_xlabel("Cortical Hierarchy", fontsize=12)
    ax.set_ylabel("Weighted Computational Depth", fontsize=12)
    ax.set_title(f"{model_name}\nComputational Depth vs Cortical Hierarchy (avg over layers)",
                 fontsize=11)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_winning_ratio(ratios_by_group, out_path, model_name):
    """本研究 plot_state_winning_ratios.py の作図に忠実に揃える:
    - state バーは Spectral グラデで色分け（全 state を lm 状態として扱う）
    - 上から浅い->深いの順（barh + invert_yaxis）
    """
    groups = list(ratios_by_group.keys())
    fig, axes = plt.subplots(1, len(groups), figsize=(5 * len(groups), 6), sharey=True)
    if len(groups) == 1:
        axes = [axes]
    order = ALL_STATES  # 浅い->深い（上から）
    # 本研究 _get_winning_ratio_bar_colors の lm 配色（Spectral 0.05〜0.95）を踏襲。
    colors = plt.cm.Spectral(np.linspace(0.05, 0.95, len(order)))
    for ax, gname in zip(axes, groups):
        ratios = ratios_by_group[gname]
        vals = [ratios[s] for s in order]
        labels = [STATE_DISPLAY[s] for s in order]
        ypos = np.arange(len(order))
        ax.barh(ypos, vals, color=colors, edgecolor="none")
        ax.invert_yaxis()  # 先頭(浅い)を上に
        for yi, v in zip(ypos, vals):
            ax.text(v + 0.3, yi, f"{v:.1f}%", va="center", fontsize=8)
        ax.set_yticks(ypos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel("Winning ratio (%)", fontsize=11)
        ax.set_title(gname, fontsize=12)
        ax.grid(True, axis="x", alpha=0.3)
    fig.suptitle(f"{model_name} — Winning ratio of each intermediate state", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_lpp_llama.yaml")
    parser.add_argument("--model", default="meta-llama/Llama-3.2-1B-Instruct")
    parser.add_argument("--out_dir", default=None,
                        help="出力先（既定: outputs/lpp_figures）")
    parser.add_argument(
        "--atlas_data_dir", default=None,
        help="Harvard-Oxford atlas の Nilearn キャッシュ先（初回取得時はネットワーク必須）",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    corr_dir = config["paths"]["llms_brain_correlations"]
    mask_dir = config["paths"]["roi_masks"]
    lang = config["experiment"]["language"].lower()
    mask_path = os.path.join(mask_dir, f"mask_lpp_{lang}.nii.gz")
    out_dir = args.out_dir or config["paths"]["figures_folder"]
    make_dir(out_dir)

    print(f"Loading corr.gz from: {corr_dir}")
    data = load_cc_by_layer_state(corr_dir, args.model)
    layers, states = get_layers_states(data)
    print(f"Loaded {len(data)} (layer,state) files. layers={layers}")
    print(f"states present: {states}")

    print(f"Building ROI masks from: {mask_path}")
    roi_masks, n_voxels, atlas_1d = build_roi_voxel_masks(mask_path, args.atlas_data_dir)
    whole_brain_mask = np.ones(n_voxels, dtype=bool)
    for r, m in roi_masks.items():
        print(f"  ROI {r}: {int(m.sum())} voxels")

    # --- Fig3: Weighted Computational Depth ---
    depth_by_roi = {}
    for roi_label, _ in CORTICAL_HIERARCHY:
        depth, n_layers_used = compute_weighted_depth(data, layers, roi_masks[roi_label])
        depth_by_roi[roi_label] = depth
        print(f"  D̄[{roi_label}] = {depth:.4f} (layers used: {n_layers_used})")

    # auditory stream (HG->MTG) 線形フィット
    aud_y = [depth_by_roi[r] for r in AUDITORY_STREAM_LABELS]
    slope, intercept, r2 = linear_fit(range(len(aud_y)), aud_y)
    print(f"  Auditory stream fit (HG→MTG): slope={slope:.4f}, R²={r2:.4f}")

    depth_png = os.path.join(out_dir, f"fig3_computational_depth_{lang}.png")
    plot_computational_depth(depth_by_roi, (slope, intercept, r2), depth_png, args.model)
    print(f"Saved: {depth_png}")

    # --- Fig1b: Winning ratio (whole-brain / auditory / language) ---
    aud_mask = np.zeros(n_voxels, dtype=bool)
    for r in AUDITORY_GROUP_LABELS:
        aud_mask |= roi_masks[r]
    lang_mask = np.zeros(n_voxels, dtype=bool)
    for r in LANGUAGE_GROUP_LABELS:
        lang_mask |= roi_masks[r]

    ratios_by_group = {}
    for gname, vmask in [("Auditory cortex", aud_mask),
                         ("Language network", lang_mask),
                         ("Whole-brain", whole_brain_mask)]:
        ratios, total = compute_winning_ratio(data, layers, vmask)
        ratios_by_group[gname] = ratios
        top = max(ratios.items(), key=lambda kv: (kv[1] if np.isfinite(kv[1]) else -1))
        print(f"  [{gname}] {total} voxel-decisions, top state: {top[0]} ({top[1]:.1f}%)")

    wr_png = os.path.join(out_dir, f"fig1b_winning_ratio_{lang}.png")
    plot_winning_ratio(ratios_by_group, wr_png, args.model)
    print(f"Saved: {wr_png}")

    # CSV も出す
    import csv
    csv_path = os.path.join(out_dir, f"computational_depth_{lang}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["roi", "weighted_depth"])
        for r, _ in CORTICAL_HIERARCHY:
            w.writerow([r, f"{depth_by_roi[r]:.6f}"])
        w.writerow([])
        w.writerow(["auditory_fit_slope", f"{slope:.6f}"])
        w.writerow(["auditory_fit_r2", f"{r2:.6f}"])
    print(f"Saved: {csv_path}")


if __name__ == "__main__":
    main()
