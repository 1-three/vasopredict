"""
preprocessing.py
----------------
Loads raw MIMIC-IV CSVs, checks data quality, converts timestamps,
handles missing values, and builds the final train/val/test split.
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# LOAD RAW DATA
# =============================================================================

def load_raw_data(data_dir: str = "data/raw") -> dict:
    """
    Load all five MIMIC-IV sample CSVs from data/raw/.

    Returns
    -------
    dict with keys: cohort, labs, vitals, inputs, outputs
    """
    print("=" * 70)
    print("LOADING MIMIC-IV SAMPLE DATA")
    print("=" * 70)

    cohort  = pd.read_csv(f"{data_dir}/cohort_sample.csv")
    labs    = pd.read_csv(f"{data_dir}/labevents_sample.csv")
    vitals  = pd.read_csv(f"{data_dir}/chartevents_sample.csv")
    inputs  = pd.read_csv(f"{data_dir}/inputevents_sample.csv")
    outputs = pd.read_csv(f"{data_dir}/outputevents_sample.csv")

    datasets = {
        "Cohort":                cohort,
        "Lab Events":            labs,
        "Chart Events (Vitals)": vitals,
        "Input Events":          inputs,
        "Output Events":         outputs,
    }

    for name, df in datasets.items():
        print(f"\n{name}:")
        print(f"  Rows:    {len(df):,}")
        print(f"  Columns: {len(df.columns)}")
        print(f"  Memory:  {df.memory_usage().sum() / 1024**2:.2f} MB")
        print(f"  Missing: {df.isnull().sum().sum():,} values")

    print(f"\nCohort Distribution:")
    print(f"  Total patients: {len(cohort):,}")
    print(f"  Sepsis cases:   {cohort['sepsis'].sum():,}  ({cohort['sepsis'].mean():.1%})")
    print(f"  Non-sepsis:     {(cohort['sepsis']==0).sum():,}  ({(cohort['sepsis']==0).mean():.1%})")

    return {"cohort": cohort, "labs": labs, "vitals": vitals,
            "inputs": inputs, "outputs": outputs}


# =============================================================================
# TIMESTAMP CONVERSION
# =============================================================================

def convert_timestamps(data: dict) -> dict:
    """Parse all datetime columns to pandas Timestamps in-place."""
    for df in [data["labs"], data["vitals"], data["inputs"], data["outputs"]]:
        if "charttime" in df.columns:
            df["charttime"] = pd.to_datetime(df["charttime"], errors="coerce", format="mixed")
        if "starttime" in df.columns:
            df["starttime"] = pd.to_datetime(df["starttime"], errors="coerce", format="mixed")

    data["cohort"]["intime"]  = pd.to_datetime(data["cohort"]["intime"],  errors="coerce", format="mixed")
    data["cohort"]["outtime"] = pd.to_datetime(data["cohort"]["outtime"], errors="coerce", format="mixed")

    print("\nTimestamps converted successfully.")
    return data


# =============================================================================
# MISSING VALUE HANDLING
# =============================================================================

def handle_missing(df: pd.DataFrame, drop_threshold: float = 0.80) -> pd.DataFrame:
    """
    1. Drop columns where missing fraction > drop_threshold.
    2. Median-impute remaining numeric columns (excluding 'sepsis').
    """
    missing_pct = df.isnull().sum() / len(df)

    print(f"\nMissing value stats:")
    print(f"  Features with >50% missing: {(missing_pct > 0.50).sum()}")
    print(f"  Features with >80% missing: {(missing_pct > 0.80).sum()}")

    df = df[missing_pct[missing_pct <= drop_threshold].index]

    for col in df.select_dtypes(include=[np.number]).columns:
        if col != "sepsis":
            df[col].fillna(df[col].median(), inplace=True)

    print(f"After cleaning: {df.shape[1]} features, {df.shape[0]} patients.")
    return df


# =============================================================================
# BUILD FINAL DATASET
# =============================================================================

def build_dataset(features_df: pd.DataFrame, cohort: pd.DataFrame):
    """
    Merge engineered features with cohort demographics and return
    clean X / y plus scaled train/val/test splits.

    Parameters
    ----------
    features_df : output of feature_engineering.create_all_features()
    cohort      : raw cohort DataFrame

    Returns
    -------
    X, y,
    X_train, X_val, X_test,
    y_train, y_val, y_test,
    X_train_sc, X_val_sc, X_test_sc,
    scaler
    """
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    # Build base with demographics
    final_df = cohort[["stay_id", "hadm_id", "subject_id",
                        "anchor_age", "gender", "sepsis"]].copy()
    final_df["age"]           = final_df["anchor_age"]
    final_df["gender_female"] = (final_df["gender"] == "F").astype(int)
    final_df = final_df.set_index("stay_id").join(features_df, how="left")

    # Clean
    final_df = handle_missing(final_df)

    X = final_df.drop(columns=["sepsis", "subject_id", "hadm_id", "gender"], errors="ignore")
    y = final_df["sepsis"]

    print(f"\nFinal dataset: {X.shape[1]} features, {X.shape[0]} samples")
    print(f"Sepsis prevalence: {y.mean():.1%}")

    # 60 / 20 / 20 split
    X_tmp,   X_test,  y_tmp,   y_test  = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)
    X_train, X_val,   y_train, y_val   = train_test_split(X_tmp, y_tmp, test_size=0.25, random_state=42, stratify=y_tmp)

    print(f"\nData splits:")
    print(f"  Train: {len(X_train)}  ({y_train.mean():.1%} sepsis)")
    print(f"  Val:   {len(X_val)}   ({y_val.mean():.1%} sepsis)")
    print(f"  Test:  {len(X_test)}  ({y_test.mean():.1%} sepsis)")

    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    print("\nPreprocessing complete.")
    return (X, y,
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            X_train_sc, X_val_sc, X_test_sc,
            scaler)


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import feature_engineering as fe

    data = load_raw_data()
    data = convert_timestamps(data)

    lab_windows   = fe.create_temporal_windows(data["labs"],   data["cohort"], "charttime", "hadm_id")
    vital_windows = fe.create_temporal_windows(data["vitals"], data["cohort"], "charttime", "stay_id")
    features_df   = fe.create_all_features(lab_windows, vital_windows, data["cohort"])

    results = build_dataset(features_df, data["cohort"])
    print("\nAll preprocessing steps complete.")
