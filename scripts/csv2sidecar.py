# csv2sidecar.py  (fixed: use underscores, safer IO)
import csv, os, pathlib, shutil, argparse, hashlib
from PIL import Image

def safe_name(p):
    h = hashlib.md5(p.encode("utf-8")).hexdigest()[:8]
    stem = pathlib.Path(p).stem
    ext = pathlib.Path(p).suffix
    return f"{stem}-{h}{ext}"

def process_image(src, dst, force_rgb=False, resize=None, center_crop=False):
    if force_rgb or resize or center_crop:
        im = Image.open(src)
        if force_rgb:
            im = im.convert("RGB")
        if resize:
            w, h = im.size
            scale = resize / min(w, h)
            im = im.resize((int(w*scale), int(h*scale)), Image.BICUBIC)
            w, h = im.size
            if center_crop:
                s = min(w, h)
                left = (w - s) // 2
                top  = (h - s) // 2
                im = im.crop((left, top, left+s, top+s))
        im.save(dst)
    else:
        shutil.copy2(src, dst)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="包含图片路径与文本/数值的 CSV")
    ap.add_argument("--out", required=True, help="输出目录（图片 + 同名 .txt）")
    ap.add_argument("--img-col-name", dest="img_col_name", default=None, help="图片列名；留空则用索引")
    ap.add_argument("--txt-col-name", dest="txt_col_name", default=None, help="文本/数值列名；留空则用索引")
    ap.add_argument("--img-col-idx",  dest="img_col_idx", type=int, default=0, help="图片列索引（从0开始）")
    ap.add_argument("--txt-col-idx",  dest="txt_col_idx", type=int, default=1, help="文本列索引（从0开始）")
    ap.add_argument("--template", default=None, help='如 "signal_b50={value}"，将第2列包装成文本')
    ap.add_argument("--force-rgb", action="store_true")
    ap.add_argument("--resize", type=int, default=None)
    ap.add_argument("--center-crop", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        header = next(reader)

        # 选列：优先用列名，否则用索引
        if args.img_col_name and args.img_col_name in header:
            img_idx = header.index(args.img_col_name)
        else:
            img_idx = args.img_col_idx

        if args.txt_col_name and args.txt_col_name in header:
            txt_idx = header.index(args.txt_col_name)
        else:
            txt_idx = args.txt_col_idx

        print(f"[INFO] header={header}")
        print(f"[INFO] using img_idx={img_idx}, txt_idx={txt_idx}")

        cnt_ok, cnt_missing = 0, 0
        for row in reader:
            if not row or len(row) <= max(img_idx, txt_idx):
                continue
            img_src = row[img_idx].strip()
            raw_txt = row[txt_idx].strip()

            cap = args.template.format(value=raw_txt) if args.template else str(raw_txt)

            if not os.path.isfile(img_src):
                print(f"[WARN] not found: {img_src}")
                cnt_missing += 1
                continue

            img_name = safe_name(img_src)
            img_dst = os.path.join(args.out, img_name)

            try:
                process_image(
                    img_src, img_dst,
                    force_rgb=args.force_rgb,
                    resize=args.resize,
                    center_crop=args.center_crop
                )
            except Exception as e:
                print(f"[WARN] PIL failed on {img_src}: {e}")
                continue

            txt_dst = os.path.splitext(img_dst)[0] + ".txt"
            with open(txt_dst, "w", encoding="utf-8") as w:
                w.write(cap.strip() + "\n")

            cnt_ok += 1

    print(f"[OK] wrote {cnt_ok} pairs, {cnt_missing} missing. Out: {args.out}")

if __name__ == "__main__":
    main()
