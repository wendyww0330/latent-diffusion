import os
import glob
import re
import numpy as np
import argparse
from tqdm import tqdm

# ==========================================
# 1. 简单的 Dataset 逻辑 (只读 txt)
# ==========================================
class TextOnlyDataset:
    def __init__(self, root):
        self.txts = sorted(glob.glob(os.path.join(root, "*.txt")))
        
        # 正则表达式 (和你的训练代码一致)
        _NUM_RE = r"([-+]?\d[\d,]*(?:\.\d+)?(?:[eE][-+]?\d+)?)"
        self.re_b50  = re.compile(rf"signal_b50\s*=\s*{_NUM_RE}\s*[,;]?")
        self.re_b800 = re.compile(rf"signal_b800\s*=\s*{_NUM_RE}\s*[,;]?")

    def __len__(self):
        return len(self.txts)

    def _parse_pair(self, text: str):
        m1, m2 = self.re_b50.search(text), self.re_b800.search(text)
        if not (m1 and m2): return None
        def _to_float(m): return float(m.group(1).rstrip(",;").replace(",", ""))
        try: return [_to_float(m1), _to_float(m2)]
        except: return None

    def get_all_conds(self):
        conds = []
        print(f"正在读取 {len(self.txts)} 个文本文件...")
        for tpath in tqdm(self.txts):
            with open(tpath, "r", encoding="utf-8") as f:
                s = f.read().strip()
            pair = self._parse_pair(s)
            if pair is not None:
                conds.append(pair)
        return np.array(conds) # [N, 2]

# ==========================================
# 2. 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dir", required=True, help="包含 .txt 文件的训练数据文件夹路径")
    args = parser.parse_args()

    ds = TextOnlyDataset(args.train_dir)
    if len(ds) == 0:
        print("未找到txt文件")
        return

    all_conds = ds.get_all_conds()
    
    # 计算统计值
    # axis=0 表示跨样本计算
    mean = all_conds.mean(0)
    std = all_conds.std(0)
    
    print("\n" + "="*40)
    print("【统计结果】请把以下两行代码复制到 inference.py 中：")
    print("="*40)
    print(f"COND_MEAN = np.array({mean.tolist()})")
    print(f"COND_STD  = np.array({std.tolist()})")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()