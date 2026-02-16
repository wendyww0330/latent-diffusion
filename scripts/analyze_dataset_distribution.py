#!/usr/bin/env python3
import argparse
from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="dataset_split", help="dataset root containing train/ and test/")
    ap.add_argument("--train", type=str, default="train", help="train folder name under root")
    ap.add_argument("--test", type=str, default="test", help="test folder name under root")
    ap.add_argument("--bins", type=int, default=50, help="histogram bins")
    ap.add_argument("--log1p", action="store_true", help="apply log1p transform before stats/distances/plots")
    ap.add_argument("--outdir", type=str, default="dataset_report", help="output directory")
    ap.add_argument(
        "--cols",
        type=str,
        default="signal_b50,signal_b800,signal_b2000",
        help="comma-separated column names matching values order in txt",
    )
    return ap.parse_args()


def read_sidecars(folder: Path, cols: List[str]) -> pd.DataFrame:
    """Read all .txt files; each contains a single line of comma-separated floats."""
    rows = []
    txts = sorted(folder.glob("*.txt"))
    if not txts:
        raise FileNotFoundError(f"No .txt sidecars found in: {folder}")

    for p in txts:
        s = p.read_text(encoding="utf-8").strip()
        # robust parse: allow commas/spaces
        parts = [x.strip() for x in re.split(r"[,\s]+", s) if x.strip() != ""]
        vals = [float(x) for x in parts]
        if len(vals) != len(cols):
            raise ValueError(f"{p}: expected {len(cols)} values ({cols}) but got {len(vals)} -> {vals}")
        row = {"file_stem": p.stem}
        row.update({c: v for c, v in zip(cols, vals)})
        rows.append(row)

    return pd.DataFrame(rows)


def safe_hist(a: np.ndarray, b: np.ndarray, bins: int):
    a = a[np.isfinite(a)]
    b = b[np.isfinite(b)]
    if a.size == 0 or b.size == 0:
        return np.array([]), np.array([])

    lo = float(min(a.min(), b.min()))
    hi = float(max(a.max(), b.max()))
    if lo == hi:
        hi = lo + 1e-6

    edges = np.linspace(lo, hi, bins + 1)
    ha, _ = np.histogram(a, bins=edges)
    hb, _ = np.histogram(b, bins=edges)

    eps = 1e-12
    p = (ha.astype(np.float64) + eps); p /= p.sum()
    q = (hb.astype(np.float64) + eps); q /= q.sum()
    return p, q


def kl(p: np.ndarray, q: np.ndarray) -> float:
    return float(np.sum(p * np.log(p / q)))


def js(p: np.ndarray, q: np.ndarray) -> float:
    m = 0.5 * (p + q)
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def describe(x: np.ndarray) -> dict:
    x = x[np.isfinite(x)]
    if x.size == 0:
        return dict(n=0, mean=np.nan, std=np.nan, min=np.nan, p25=np.nan, p50=np.nan, p75=np.nan, max=np.nan)
    return dict(
        n=int(x.size),
        mean=float(np.mean(x)),
        std=float(np.std(x, ddof=1)) if x.size > 1 else 0.0,
        min=float(np.min(x)),
        p25=float(np.percentile(x, 25)),
        p50=float(np.percentile(x, 50)),
        p75=float(np.percentile(x, 75)),
        max=float(np.max(x)),
    )


def maybe_log1p(train: np.ndarray, test: np.ndarray):
    tr = train[np.isfinite(train)]
    te = test[np.isfinite(test)]
    if tr.size == 0 or te.size == 0:
        return train, test, 0.0
    minv = float(min(tr.min(), te.min()))
    shift = -minv if minv < 0 else 0.0
    return np.log1p(train + shift), np.log1p(test + shift), shift


def plot_overlay(train: np.ndarray, test: np.ndarray, col: str, outpath: Path, bins: int):
    plt.figure()
    plt.hist(train[np.isfinite(train)], bins=bins, alpha=0.5, label="train", density=True)
    plt.hist(test[np.isfinite(test)], bins=bins, alpha=0.5, label="test", density=True)
    plt.legend()
    plt.title(col)
    plt.tight_layout()
    plt.savefig(outpath, dpi=200)
    plt.close()


def main():
    args = parse_args()
    root = Path(args.root)
    train_dir = root / args.train
    test_dir = root / args.test

    outdir = Path(args.outdir)
    plots_dir = outdir / "plots"
    outdir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    cols = [c.strip() for c in args.cols.split(",") if c.strip()]

    train_df = read_sidecars(train_dir, cols)
    test_df = read_sidecars(test_dir, cols)

    rows = []
    for col in cols:
        tr = train_df[col].to_numpy(dtype=np.float64)
        te = test_df[col].to_numpy(dtype=np.float64)

        # optional log1p (and record shift if negative values exist)
        shift = 0.0
        if args.log1p:
            tr, te, shift = maybe_log1p(tr, te)

        p, q = safe_hist(tr, te, bins=args.bins)
        if p.size == 0:
            kl_v = js_v = np.nan
        else:
            kl_v = kl(p, q)
            js_v = js(p, q)

        tr_stats = describe(tr)
        te_stats = describe(te)

        # plot (use the same transformed arrays so plots match reported distances)
        plot_name = f"{col}{'_log1p' if args.log1p else ''}.png"
        plot_overlay(tr, te, col + (" (log1p)" if args.log1p else ""), plots_dir / plot_name, args.bins)

        rows.append(
            dict(
                column=col,
                log1p=args.log1p,
                shift_before_log1p=shift,
                kl_train_test=kl_v,
                js_train_test=js_v,
                train_n=tr_stats["n"],
                train_mean=tr_stats["mean"],
                train_std=tr_stats["std"],
                train_min=tr_stats["min"],
                train_p25=tr_stats["p25"],
                train_p50=tr_stats["p50"],
                train_p75=tr_stats["p75"],
                train_max=tr_stats["max"],
                test_n=te_stats["n"],
                test_mean=te_stats["mean"],
                test_std=te_stats["std"],
                test_min=te_stats["min"],
                test_p25=te_stats["p25"],
                test_p50=te_stats["p50"],
                test_p75=te_stats["p75"],
                test_max=te_stats["max"],
            )
        )

    report = pd.DataFrame(rows).sort_values("js_train_test", ascending=False)
    report.to_csv(outdir / "report.csv", index=False)

    # 额外输出一个简洁摘要，方便你 copy 到论文/笔记
    summary = report[["column", "kl_train_test", "js_train_test", "train_mean", "test_mean", "train_std", "test_std"]]
    summary.to_csv(outdir / "summary.csv", index=False)

    print(f"[OK] train sidecars: {len(train_df)} | test sidecars: {len(test_df)}")
    print(f"[OK] report: {outdir/'report.csv'}")
    print(f"[OK] summary: {outdir/'summary.csv'}")
    print(f"[OK] plots: {plots_dir}")
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
