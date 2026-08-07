# -*- coding: utf-8 -*-
"""
data_prep.py — Generate WITHOUT data (literature subset)
=============================================
The main `prepare_fixed_split` requires two data directories:
  - data/BasicData     (WITH: 2332 literature + 24 experiment = 2356 rows)
  - data/BasicData_wo  (WITHOUT: 2332 literature only, used for split identity assertion and literature count)

To avoid storing two ~317MB descriptor matrices in the repository, WITHOUT is
sliced from the first N_LIT=2332 rows of WITH (experimental rows are appended
at the end, so the slice is exactly the literature part).

Usage:
    python data_prep.py
"""
import os
import pandas as pd

BASE = os.path.dirname(os.path.abspath(__file__))
WITH = os.path.join(BASE, "data", "BasicData")
WITHOUT = os.path.join(BASE, "data", "BasicData_wo")
N_LIT = 2332  # number of literature rows (current data-version fact; experiment rows = total - N_LIT, at the end)


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
        assert len(df) > N_LIT, f"{fn} has too few rows: {len(df)}"
        df.iloc[:N_LIT].to_csv(os.path.join(WITHOUT, fn),
                               index=False, encoding="utf-8-sig")
        print(f"  {fn}: {len(df)} -> {N_LIT} rows")
    # Dataset_with_SMILES.xlsx
    ds = pd.read_excel(os.path.join(WITH, "Dataset_with_SMILES.xlsx"))
    ds.iloc[:N_LIT].to_excel(os.path.join(WITHOUT, "Dataset_with_SMILES.xlsx"),
                             index=False)
    print(f"  Dataset_with_SMILES.xlsx: {len(ds)} -> {N_LIT} rows")
    print(f"\n[DONE] WITHOUT data generated at {WITHOUT}")


if __name__ == "__main__":
    main()