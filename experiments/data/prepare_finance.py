"""Download and parse the Ken French 48-industry daily value-weighted return series used by fin_exp.py.

Source: Kenneth French's data library,
https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
(48_Industry_Portfolios daily file). The file is refreshed periodically by French's group; this script
always downloads the current version, so the exact end date and file size can drift over time.

Usage: python data/prepare_finance.py
Output: data/ff48_vw_daily.pkl  (pandas DataFrame, dates x 48 industries, % daily returns, NaN for
        industries not yet listed on a given date, matching the paper's zero-mean Gaussian convention
        once demeaned).
"""
import io
import os
import zipfile

import numpy as np
import pandas as pd
import urllib.request

URL = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/48_Industry_Portfolios_daily_CSV.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
ZIP_PATH = os.path.join(HERE, "ff48.zip")
CSV_PATH = os.path.join(HERE, "48_Industry_Portfolios_Daily.csv")
OUT_PATH = os.path.join(HERE, "ff48_vw_daily.pkl")


def download():
    if not os.path.exists(ZIP_PATH):
        print(f"Downloading {URL} ...")
        urllib.request.urlretrieve(URL, ZIP_PATH)
    with zipfile.ZipFile(ZIP_PATH) as z:
        # the archive contains one CSV whose exact filename/case can vary by release
        name = next(n for n in z.namelist() if n.lower().endswith(".csv"))
        with z.open(name) as f, open(CSV_PATH, "wb") as out:
            out.write(f.read())


def parse():
    """The file stacks several blocks (value-weighted, equal-weighted, ... returns) separated by
    blank/header lines. We extract the "Average Value Weighted Returns -- Daily" block only."""
    raw = open(CSV_PATH, encoding="latin-1").read().split("\n")
    start = next(i for i, l in enumerate(raw) if "Average Value Weighted Returns -- Daily" in l)
    end = next(i for i, l in enumerate(raw) if "Average Equal Weighted Returns -- Daily" in l)
    block = "\n".join(raw[start + 1:end]).strip()
    df = pd.read_csv(io.StringIO(block), index_col=0)
    df.index = pd.to_datetime(df.index.astype(str), format="%Y%m%d")
    df.columns = [c.strip() for c in df.columns]
    df = df.replace([-99.99, -999], np.nan)  # French's missing-data codes
    return df


if __name__ == "__main__":
    download()
    df = parse()
    df.to_pickle(OUT_PATH)
    print(f"Saved {OUT_PATH}: {df.shape[0]} days x {df.shape[1]} industries, "
          f"{df.index[0].date()} to {df.index[-1].date()}")
