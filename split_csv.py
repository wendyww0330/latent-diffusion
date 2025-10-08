# split_csv_robust.py
import argparse, pandas as pd, numpy as np, re
from pathlib import Path

INVISIBLE_WS = "".join([
    "\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u00a0"
])

def read_flexible_csv(path):
    # 1) 自动探测
    try:
        return pd.read_csv(path, sep=None, engine="python", encoding="utf-8-sig")
    except Exception:
        pass
    # 2) 常见分隔符
    for sep in [",",";","\t","|"]:
        try:
            df = pd.read_csv(path, sep=sep, encoding="utf-8-sig")
            if df.shape[1] > 1: return df
        except Exception:
            continue
    # 3) 无表头兜底
    for sep in [",",";","\t","|"]:
        try:
            df = pd.read_csv(path, sep=sep, header=None, encoding="utf-8-sig")
            if df.shape[1] > 1:
                df.columns = [f"col{i}" for i in range(df.shape[1])]
                return df
        except Exception:
            continue
    # 4) 最终兜底
    return pd.read_csv(path, encoding="utf-8-sig")

def normalize_col(s: str) -> str:
    if not isinstance(s, str): s = str(s)
    s = s.replace("\ufeff","")
    s = re.sub(f"[{re.escape(INVISIBLE_WS)}]", "", s)  # 去不可见空白
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)     # 折叠空格
    return s

def build_norm_map(cols):
    norm = {}
    for c in cols:
        norm[normalize_col(c)] = c
    return norm

def pick_col(df, want_name, want_idx):
    cols = list(df.columns)
    norm_map = build_norm_map(cols)
    if want_name:
        key = normalize_col(want_name)
        if key in norm_map:
            return norm_map[key]
    if want_idx is not None and 0 <= want_idx < len(cols):
        return cols[want_idx]
    raise KeyError(f"Cannot find column by name={want_name!r} or idx={want_idx}, available={cols}")

def stratified_split(df, strata_col, train=0.8, val=0.1, test=0.1, seed=42):
    assert abs(train+val+test-1.0) < 1e-6
    rng = np.random.RandomState(seed)
    parts = []
    for _, g in df.groupby(strata_col):
        ix = rng.permutation(len(g))
        n_tr = int(round(len(g)*train))
        n_va = int(round(len(g)*val))
        tr = g.iloc[ix[:n_tr]]
        va = g.iloc[ix[n_tr:n_tr+n_va]]
        te = g.iloc[ix[n_tr+n_va:]]
        parts.append(("train", tr)); parts.append(("val", va)); parts.append(("test", te))
    return (
        pd.concat([p for k,p in parts if k=="train"], axis=0),
        pd.concat([p for k,p in parts if k=="val"],   axis=0),
        pd.concat([p for k,p in parts if k=="test"],  axis=0),
    )

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--img-col-name", dest="img_col_name", default=None)
    ap.add_argument("--txt-col-name", dest="txt_col_name", default=None)
    ap.add_argument("--img-col-idx",  dest="img_col_idx", type=int, default=None)
    ap.add_argument("--txt-col-idx",  dest="txt_col_idx", type=int, default=None)
    ap.add_argument("--bins", type=int, default=5)
    ap.add_argument("--train", type=float, default=0.8)
    ap.add_argument("--val",   type=float, default=0.1)
    ap.add_argument("--test",  type=float, default=0.1)
    ap.add_argument("--seed",  type=int, default=42)
    ap.add_argument("--outdir", default="splits")
    ap.add_argument("--dry-run", action="store_true", help="只检查列，不写文件")
    ap.add_argument("--compose", default=None,
                help="用 Python 格式串把多列拼成文本列，例如: "
                     "'signal_b50={signal_b50:.6g}, signal_b800={signal_b800:.6g}'。"
                     "若提供，则自动生成列名 __text 并作为 txt 列使用。")
    ap.add_argument("--abs-paths", action="store_true",
                    help="将选定的图片列转换为绝对路径")
    args = ap.parse_args()

    df = read_flexible_csv(args.csv)

    # ### NEW: 如果传了 --compose，就用模板把多列拼成一列 __text
    if args.compose:
        # 支持 {col:.6g} 等格式说明
        def row_fmt(r):
            # 允许格式化器取到列名
            return args.compose.format(**{k: r[k] for k in df.columns})
        df["__text"] = df.apply(row_fmt, axis=1)
        # 告诉后续选择列时用 __text
        args.txt_col_name = "__text"

    # 打印标准化前后的列名
    orig_cols = list(df.columns)
    norm_cols = [normalize_col(c) for c in orig_cols]
    print("[INFO] columns:", orig_cols, "shape:", df.shape)
    print("[INFO] normalized:", norm_cols)


    # 选择列（支持按名或按索引）
    img_col = pick_col(df, args.img_col_name, args.img_col_idx if args.img_col_idx is not None else 0)
    txt_col = pick_col(df, args.txt_col_name, args.txt_col_idx if args.txt_col_idx is not None else 1)
    print(f"[OK] picked img_col={img_col!r}, txt_col={txt_col!r}")



    if args.dry_run:
        print("[DRY RUN] Done. No files written.")
        raise SystemExit(0)



    # —— 替换你脚本中 “判定文本列是否为数值 → 决定分层方式” 那一整段 —— 
    # 原地放到 pick_col 之后、写文件之前

    # 统一先拿到文本列并转成字符串
    txt_vals = df[txt_col].astype(str)

    if args.compose:
        # 显式告诉我们这是拼接文本 → 按长度分层
        lens = txt_vals.str.len()
        q = pd.qcut(lens, q=min(args.bins, max(1, lens.nunique())), duplicates="drop")
        df["_strata"] = q.astype(str)
        is_numeric = False
    else:
        # 自动判定是否为数值
        numeric = pd.to_numeric(txt_vals, errors="coerce")
        is_numeric = numeric.notna().mean() > 0.95
        if is_numeric:
            q = pd.qcut(numeric, q=min(args.bins, max(1, numeric.nunique())), duplicates="drop")
        else:
            lens = txt_vals.str.len()
            q = pd.qcut(lens, q=min(args.bins, max(1, lens.nunique())), duplicates="drop")
        df["_strata"] = q.astype(str)


    train_df, val_df, test_df = stratified_split(
        df, "_strata", train=args.train, val=args.val, test=args.test, seed=args.seed
    )

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    keep = [img_col, txt_col] + [c for c in df.columns if c not in [img_col, txt_col, "_strata"]]
    train_df[keep].to_csv(Path(args.outdir)/"train.csv", index=False)
    val_df[keep].to_csv(Path(args.outdir)/"val.csv", index=False)
    test_df[keep].to_csv(Path(args.outdir)/"test.csv", index=False)

    print(f"[OK] numeric={is_numeric}")
    print(f"[OK] sizes: train={len(train_df)}, val={len(val_df)}, test={len(test_df)} -> {args.outdir}/")
