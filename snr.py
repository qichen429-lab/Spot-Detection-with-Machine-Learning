"""
Write has_strong_crossing to metadata (SNR > 5). Not used in training.

  python snr.py
"""

import pandas as pd
import numpy as np
from scipy.interpolate import UnivariateSpline
from operator import itemgetter
from itertools import groupby
from tqdm import tqdm

STRONG_THRESHOLD = 5.0


def compute_snr(data, noise_ppm):
    masked_data = data["mask"] == 1
    oot = ~masked_data
    spline = UnivariateSpline(
        data[oot]["time_days"].values, data[oot]["flux_clean"].values, k=2, s=0
    )
    flux_baseline = spline(data["time_days"].values)
    noise = noise_ppm * 1e-6
    snrs = []
    for _, g in groupby(enumerate(np.where(masked_data)[0]), lambda x: x[0] - x[1]):
        group = list(map(itemgetter(1), g))
        bump = data["flux_clean"].iloc[group].values - flux_baseline[group]
        snrs.append((bump.max() - bump.min()) / noise)
    return snrs


def has_strong_crossing(snrs) -> int:
    if snrs is None or (isinstance(snrs, float) and np.isnan(snrs)):
        return 0
    return int(any(s > STRONG_THRESHOLD for s in snrs))


if __name__ == "__main__":
    metadata = pd.read_csv("dataset/metadata.csv")
    snr_column = []
    for _, row in tqdm(metadata.iterrows(), total=len(metadata)):
        if row["has_crossing"] == 0:
            snr_column.append(np.nan)
        else:
            lc = pd.read_csv(f"dataset/lightcurves/{row['name']}.csv")
            snr_column.append(compute_snr(lc, row["noise_ppm"]))
    metadata["approx_snr"] = snr_column
    metadata["has_strong_crossing"] = metadata["approx_snr"].apply(has_strong_crossing)
    metadata.to_csv("dataset/metadata.csv", index=False)
    print(
        f"Saved dataset/metadata.csv "
        f"(has_strong_crossing: any crossing SNR > {STRONG_THRESHOLD})"
    )
