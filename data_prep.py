# -*- coding: utf-8 -*-
"""
data_prep.py — 生成 WITHOUT 数据（文献子集）
=============================================
主项目 `prepare_fixed_split` 需要两个数据目录：
  - data/BasicData     （WITH：2332 条文献 + 24 条实验 = 2356 行）
  - data/BasicData_wo  （WITHOUT：仅 2332 条文献，用于划分身份断言与文献行数）

为避免仓库重复存放两份 ~317MB 描述符矩阵，WITHOUT 由本脚本从 WITH 的前
N_LIT=2332 行切片生成（实验数据固定追加在末尾，因此切片即文献部分）。

用法：
    python data_prep.py
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
WITH = os.path.join(BASE, "data", "BasicData")
WITHOUT = os.path.join(BASE, "data", "BasicData_wo")
N_LIT = 2332  # 文献行数（当前数据版本事实；实验行 = 总行数 - N_LIT，位于末尾）


def read_csv_enc(fp):
    for enc in ["utf-8", "gbk", "latin1"]:
        try:
            return pd.read_csv(fp, encoding=enc, low_memory=False)
        except UnicodeDecodeError:
            continue
    return pd.read_csv(fp, encoding="utf-8", errors="replace")


def main():
    os.makedirs(WITHOUT, exist_ok=True)
    files = [
        "EP_Des.csv", "FR_Des.csv", "Curing_Des.csv",
        "Other_Material_1_Des.csv", "Other_Material_2_Des.csv",
        "Other_Material_3_Des.csv", "Curing_Strategy.csv", "Target.csv",
    ]
    for fn in files:
        df = read_csv_enc(os.path.join(WITH, fn))
        assert len(df) > N_LIT, f"{fn} 行数不足: {len(df)}"
        df.iloc[:N_LIT].to_csv(os.path.join(WITHOUT, fn),
                               index=False, encoding="utf-8-sig")
        print(f"  {fn}: {len(df)} -> {N_LIT} 行")
    # Dataset_with_SMILES.xlsx
    ds = pd.read_excel(os.path.join(WITH, "Dataset_with_SMILES.xlsx"))
    ds.iloc[:N_LIT].to_excel(os.path.join(WITHOUT, "Dataset_with_SMILES.xlsx"),
                             index=False)
    print(f"  Dataset_with_SMILES.xlsx: {len(ds)} -> {N_LIT} 行")
    print(f"\n[DONE] WITHOUT 数据已生成于 {WITHOUT}")


if __name__ == "__main__":
    main()
