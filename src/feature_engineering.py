"""
feature_engineering.py
-----------------------
Creates temporal windows (3h, 6h, 12h, 24h, 48h) from raw ICU event
tables and aggregates them into statistical features (mean, max, min,
std, slope) for each patient.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

TIME_WINDOWS = [3, 6, 12, 24, 48]  # hours post-ICU admission


# =============================================================================
# TEMPORAL WINDOWING
# =============================================================================

def create_temporal_windows(
    df: pd.DataFrame,
    cohort: pd.DataFrame,
    time_col: str,
    stay_col: str,
    time_windows: list = TIME_WINDOWS,
) -> dict:
    """
    Slice an event table into multiple time-window sub-tables.

    Parameters
    ----------
    df           : labs or vitals event DataFrame
    cohort       : cohort DataFrame (must contain stay_col + 'intime')
    time_col     : name of the timestamp column in df
    stay_col     : join key — 'hadm_id' for labs, 'stay_id' for vitals
    time_windows : list of cutoff hours

    Returns
    -------
    dict  {f'window_{N}h': filtered_DataFrame}
    """
    print(f"\nCreating temporal windows for {stay_col} events...")

    merge_cols = [stay_col, "intime"] if stay_col == "hadm_id" else [stay_col, "intime"]
    df_temp = df.merge(cohort[[stay_col, "intime"]], on=stay_col, how="left")
    df_temp["hours_since_admission"] = (
        (df_temp[time_col] - df_temp["intime"]).dt.total_seconds() / 3600
    )

    windows = {}
    for window in time_windows:
        print(f"  Processing {window}h window...")
        windows[f"window_{window}h"] = df_temp[
            (df_temp["hours_since_admission"] >= 0) &
            (df_temp["hours_since_admission"] <= window)
        ].copy()

    print(f"  Created {len(windows)} windows.")
    return windows


# =============================================================================
# STATISTICAL AGGREGATION
# =============================================================================

def _calculate_slope(group: pd.DataFrame, value_col: str) -> float:
    """Compute linear regression slope across multiple measurements."""
    if len(group) < 2:
        return np.nan
    x = np.arange(len(group))
    y = group[value_col].values
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    return float(np.polyfit(x[mask], y[mask], 1)[0])


def aggregate_measurements(
    df_window: pd.DataFrame,
    value_col: str,
    group_col: str,
    prefix: str,
) -> pd.DataFrame:
    """
    Aggregate a single measurement type into statistical features.

    Produces: mean, max, min, std, count, slope  — all prefixed with `prefix`.

    Parameters
    ----------
    df_window  : event rows already filtered to a time window
    value_col  : numeric column to aggregate
    group_col  : patient ID column to group by
    prefix     : column name prefix (e.g. 'window_6h_lab_Lactate')
    """
    features = df_window.groupby(group_col)[value_col].agg(
        ["mean", "max", "min", "std", "count"]
    )
    features.columns = [f"{prefix}_{c}" for c in features.columns]

    # Slope (trend over time — more informative than a single snapshot)
    slopes = (
        df_window.groupby(group_col)
        .apply(lambda g: _calculate_slope(g, value_col))
    )
    features[f"{prefix}_slope"] = slopes

    return features


# =============================================================================
# BUILD FULL FEATURE MATRIX
# =============================================================================

def create_all_features(
    lab_windows: dict,
    vital_windows: dict,
    cohort: pd.DataFrame,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Iterate over every time-window × measurement-type combination and
    build a wide feature DataFrame (one row per patient).

    Parameters
    ----------
    lab_windows   : dict from create_temporal_windows() for labs
    vital_windows : dict from create_temporal_windows() for vitals
    cohort        : cohort DataFrame (unused directly; kept for clarity)
    top_n         : number of most-common measurement types to keep per table

    Returns
    -------
    pd.DataFrame  indexed by stay_id (or hadm_id for lab features)
    """
    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING: AGGREGATIONS")
    print("=" * 70)

    all_parts = []

    # ---- Lab features -------------------------------------------------------
    for window_name, lab_df in lab_windows.items():
        print(f"\n  [Labs] {window_name}...")
        lab_parts = {}
        for lab_name in lab_df["lab_name"].value_counts().head(top_n).index:
            subset    = lab_df[lab_df["lab_name"] == lab_name]
            safe_name = lab_name.replace(" ", "_").replace("/", "_")[:20]
            prefix    = f"{window_name}_lab_{safe_name}"
            feat      = aggregate_measurements(subset, "valuenum", "hadm_id", prefix)
            lab_parts[lab_name] = feat
        if lab_parts:
            all_parts.append(pd.concat(lab_parts.values(), axis=1))

    # ---- Vital features -----------------------------------------------------
    for window_name, vital_df in vital_windows.items():
        print(f"  [Vitals] {window_name}...")
        vital_parts = {}
        for vital_name in vital_df["vital_name"].value_counts().head(top_n).index:
            subset    = vital_df[vital_df["vital_name"] == vital_name]
            safe_name = vital_name.replace(" ", "_").replace("/", "_")[:20]
            prefix    = f"{window_name}_vital_{safe_name}"
            feat      = aggregate_measurements(subset, "valuenum", "stay_id", prefix)
            vital_parts[vital_name] = feat
        if vital_parts:
            all_parts.append(pd.concat(vital_parts.values(), axis=1))

    features_df = pd.concat(all_parts, axis=1)

    print(f"\nCreated {len(features_df.columns)} features from {len(features_df)} patients.")
    print("Feature engineering complete.")
    return features_df


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    from preprocessing import load_raw_data, convert_timestamps

    data          = load_raw_data()
    data          = convert_timestamps(data)

    lab_windows   = create_temporal_windows(data["labs"],   data["cohort"], "charttime", "hadm_id")
    vital_windows = create_temporal_windows(data["vitals"], data["cohort"], "charttime", "stay_id")
    features_df   = create_all_features(lab_windows, vital_windows, data["cohort"])

    print(features_df.head())
