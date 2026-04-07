# feature_prep.py
import pandas as pd
import numpy as np
import pickle
import os

# ── CRITICAL FEATURES ─────────────────────────────────────────────────────────
# These 6 columns MUST exist. If any are missing → Case 2 error → results page.
CRITICAL_FEATURES = [
    'duration',
    'protocol_type',
    'service',
    'flag',
    'src_bytes',
    'dst_bytes',
]

# ── NSL-KDD RELEVANCE SIGNATURE ───────────────────────────────────────────────
# At least ONE of these must exist for the dataset to be considered
# network-traffic related at all. If NONE exist → Case 1 (irrelevant dataset).
RELEVANCE_SIGNATURE = [
    'duration', 'protocol_type', 'service', 'flag',
    'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'logged_in', 'count', 'srv_count',
    'serror_rate', 'rerror_rate', 'dst_host_count',
]

# Minimum fraction of signature columns that must exist to be "network data"
RELEVANCE_THRESHOLD = 0.25   # at least 25% of signature columns must be present

# ── NON-CRITICAL FEATURE DEFAULTS ────────────────────────────────────────────
# Domain-appropriate defaults — NOT blindly 0 for everything.
NON_CRITICAL_DEFAULTS = {
    'land':                        0,
    'wrong_fragment':              0,
    'urgent':                      0,
    'hot':                         0,
    'num_failed_logins':           0,
    'logged_in':                   0,
    'num_compromised':             0,
    'root_shell':                  0,
    'su_attempted':                0,
    'num_root':                    0,
    'num_file_creations':          0,
    'num_shells':                  0,
    'num_access_files':            0,
    'num_outbound_cmds':           0,
    'is_host_login':               0,
    'is_guest_login':              0,
    'count':                       1,      # at least 1 connection seen
    'srv_count':                   1,
    'serror_rate':                 0.0,
    'srv_serror_rate':             0.0,
    'rerror_rate':                 0.0,
    'srv_rerror_rate':             0.0,
    'same_srv_rate':               1.0,    # assume same-service traffic
    'diff_srv_rate':               0.0,
    'srv_diff_host_rate':          0.0,
    'dst_host_count':              1,
    'dst_host_srv_count':          1,
    'dst_host_same_srv_rate':      1.0,
    'dst_host_diff_srv_rate':      0.0,
    'dst_host_same_src_port_rate': 0.0,
    'dst_host_srv_diff_host_rate': 0.0,
    'dst_host_serror_rate':        0.0,
    'dst_host_srv_serror_rate':    0.0,
    'dst_host_rerror_rate':        0.0,
    'dst_host_srv_rerror_rate':    0.0,
}

# ── NUMERIC COLUMNS ───────────────────────────────────────────────────────────
NUMERIC_FEATURES = [
    'duration', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent',
    'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
    'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login',
    'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate',
    'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate',
    'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate',
]


# ─────────────────────────────────────────────────────────────────────────────
def load_feature_names(features_path='features.pkl'):
    """Load the ordered list of 117 feature names the trained model expects."""
    if not os.path.exists(features_path):
        raise FileNotFoundError(
            f"features.pkl not found at '{features_path}'. "
            f"Ensure it is in your project root folder."
        )
    with open(features_path, 'rb') as f:
        feature_names = pickle.load(f)
    return feature_names


# ─────────────────────────────────────────────────────────────────────────────
# CASE 1 CHECK
# ─────────────────────────────────────────────────────────────────────────────
def check_dataset_relevance(df):
    """
    CASE 1 — Is this dataset network-traffic related at all?

    Checks what fraction of the NSL-KDD relevance signature columns exist.
    If below threshold → completely irrelevant dataset → return error.

    Returns:
        (True,  None)           → dataset looks like network traffic
        (False, error_message)  → dataset is unrelated to network traffic
    """
    cols_lower = [c.lower().strip() for c in df.columns]
    sig_lower  = [c.lower() for c in RELEVANCE_SIGNATURE]

    matched = sum(1 for col in sig_lower if col in cols_lower)
    ratio   = matched / len(RELEVANCE_SIGNATURE)

    print(f"[INFO] Relevance check: {matched}/{len(RELEVANCE_SIGNATURE)} "
          f"signature columns found ({ratio:.0%})")

    if ratio < RELEVANCE_THRESHOLD:
        return False, (
            "Invalid dataset. The uploaded file does not appear to contain "
            "network traffic data. Please upload a network traffic dataset "
            "in NSL-KDD format."
        )
    return True, None


# ─────────────────────────────────────────────────────────────────────────────
# CASE 2 CHECK
# ─────────────────────────────────────────────────────────────────────────────
def check_critical_features(df):
    """
    CASE 2 — Are all critical features present?

    If any are missing, attempt simple derivation first.
    If derivation is not possible → return error listing missing columns.

    Returns:
        (df, None)              → all critical features now present
        (None, error_message)   → critical features unrecoverable
    """
    missing = [col for col in CRITICAL_FEATURES if col not in df.columns]

    if not missing:
        return df, None     # Case 3: all present — proceed normally

    print(f"[WARN] Missing critical features: {missing}")

    # ── Attempt simple derivation for known derivable columns ──────────────
    # src_bytes derivable from total_bytes - dst_bytes
    if 'src_bytes' in missing and 'total_bytes' in df.columns and 'dst_bytes' in df.columns:
        df['src_bytes'] = df['total_bytes'] - df['dst_bytes']
        missing.remove('src_bytes')
        print(f"[INFO] Derived src_bytes from total_bytes - dst_bytes")

    # dst_bytes derivable from total_bytes - src_bytes
    if 'dst_bytes' in missing and 'total_bytes' in df.columns and 'src_bytes' in df.columns:
        df['dst_bytes'] = df['total_bytes'] - df['src_bytes']
        missing.remove('dst_bytes')
        print(f"[INFO] Derived dst_bytes from total_bytes - src_bytes")

    # duration: if 'time' or 'connection_time' column exists
    if 'duration' in missing:
        for alt in ['time', 'connection_time', 'conn_duration']:
            if alt in df.columns:
                df['duration'] = df[alt]
                missing.remove('duration')
                print(f"[INFO] Mapped '{alt}' → duration")
                break

    # If still missing after derivation attempts → unrecoverable
    if missing:
        return None, (
            f"Missing required features: {', '.join(missing)}. "
            f"These columns are critical for intrusion detection and "
            f"cannot be derived from the available data. "
            f"Please ensure your dataset contains: "
            f"duration, protocol_type, service, flag, src_bytes, dst_bytes."
        )

    return df, None


# ─────────────────────────────────────────────────────────────────────────────
def fill_non_critical_features(df):
    """
    CASE 2/3 — Fill non-critical missing features with domain defaults.
    Only fills columns that are actually missing.
    """
    added = []
    for col, default_val in NON_CRITICAL_DEFAULTS.items():
        if col not in df.columns:
            df[col] = default_val
            added.append(col)

    if added:
        print(f"[INFO] Filled {len(added)} non-critical missing column(s) "
              f"with domain defaults.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
def enforce_numeric_types(df):
    """Force-convert numeric columns. Non-convertible values → NaN."""
    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    return df


# ─────────────────────────────────────────────────────────────────────────────
def clean_nulls(df):
    """
    Fill remaining NaN values with 0.
    Runs AFTER domain defaults — only catches values that failed
    numeric conversion, not legitimately missing features.
    """
    null_count = df.isnull().sum().sum()
    if null_count > 0:
        print(f"[INFO] Cleaning {null_count} NaN value(s) left after conversion.")
        df = df.fillna(0)
    return df


# ─────────────────────────────────────────────────────────────────────────────
def construct_derived_features(df):
    """Build engineered features. No encoding — numeric only."""

    if 'src_bytes' in df.columns and 'dst_bytes' in df.columns:
        total = df['src_bytes'] + df['dst_bytes']
        df['bytes_ratio'] = df['src_bytes'] / total.replace(0, 1)
        df['total_bytes'] = total

    if 'serror_rate' in df.columns and 'rerror_rate' in df.columns:
        df['error_rate_diff'] = df['serror_rate'] - df['rerror_rate']

    if 'srv_serror_rate' in df.columns and 'srv_rerror_rate' in df.columns:
        df['srv_error_rate_diff'] = df['srv_serror_rate'] - df['srv_rerror_rate']

    if 'dst_host_srv_count' in df.columns and 'dst_host_count' in df.columns:
        df['host_srv_ratio'] = (
            df['dst_host_srv_count'] / df['dst_host_count'].replace(0, 1)
        )

    if 'logged_in' in df.columns and 'root_shell' in df.columns:
        df['is_logged_root'] = (
            (df['logged_in'] == 1) & (df['root_shell'] == 1)
        ).astype(int)

    if 'num_failed_logins' in df.columns:
        df['failed_login_flag'] = (df['num_failed_logins'] > 0).astype(int)

    if 'duration' in df.columns:
        df['duration_log'] = np.log1p(df['duration'])

    return df


# ─────────────────────────────────────────────────────────────────────────────
def handle_label_column(df):
    """
    Handles the label/classnum column situation.

    During training, 'classnum' (numeric class ID) was accidentally
    included as a feature. The uploaded CSV has this same column
    but named 'label'. We rename it so the model accepts it.

    Any other non-feature columns are dropped.
    """
    # Rename 'label' → 'classnum' if label contains numeric values
    # (this satisfies the model's expectation of classnum)
    if 'label' in df.columns:
        if pd.api.types.is_numeric_dtype(df['label']):
            df = df.rename(columns={'label': 'classnum'})
            print(f"[INFO] Renamed 'label' → 'classnum' (numeric class IDs).")
        else:
            # label has string values like 'neptune', 'normal' — drop it
            df = df.drop(columns=['label'])
            print(f"[INFO] Dropped string 'label' column.")

    # Drop any other known non-feature columns
    other_drop = ['class', 'target', 'attack_type', 'class_label',
                  'outcome', 'category', 'attack']
    existing = [col for col in other_drop if col in df.columns]
    if existing:
        df = df.drop(columns=existing)
        print(f"[INFO] Dropped extra non-feature columns: {existing}")

    return df


def align_to_model_features(df, feature_names):
    """
    Final alignment:
    1. Add any still-missing model columns with 0
    2. Reorder to exactly match training column order
    """
    still_missing = [col for col in feature_names if col not in df.columns]
    if still_missing:
        print(f"[INFO] Alignment: padding {len(still_missing)} column(s) with 0.")
        for col in still_missing:
            df[col] = 0
    return df[feature_names]


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def prepare_features(df, features_path='features.pkl'):
    """
    Intelligent 3-case Feature Preparation Pipeline.

    CASE 1: Dataset not network-related     → (None, error)
    CASE 2: Network data, missing criticals → (None, error) or attempt recovery
    CASE 3: Valid dataset                   → (prepared_df, None)

    Returns:
        (prepared_df, None)      → success, ready for pipeline.predict()
        (None, error_message)    → failure, render on results page
    """
    print(f"\n{'='*55}")
    print(f"[FEATURE PREP] Started — input shape: {df.shape}")
    print(f"{'='*55}")

    # ── CASE 1: Relevance check ───────────────────────────────────────────
    is_relevant, error = check_dataset_relevance(df)
    if not is_relevant:
        print(f"[CASE 1] Dataset rejected — not network traffic.")
        return None, error
    print(f"[OK] Dataset is network-traffic related.")

    # ── CASE 2: Critical feature check + recovery attempt ─────────────────
    df, error = check_critical_features(df)
    if error:
        print(f"[CASE 2] Critical features missing and unrecoverable.")
        return None, error
    print(f"[OK] All critical features present.")

    # ── CASE 3: Full processing ───────────────────────────────────────────
    # Drop label/target columns before any processing
    df = handle_label_column(df)

    df = fill_non_critical_features(df)
    df = enforce_numeric_types(df)
    print(f"[OK] Numeric types enforced.")

    df = construct_derived_features(df)
    print(f"[OK] Derived features built.")

    df = clean_nulls(df)
    print(f"[OK] Null values cleaned.")

    try:
        feature_names = load_feature_names(features_path)
        print(f"[OK] Loaded {len(feature_names)} model feature names.")
    except FileNotFoundError as e:
        return None, str(e)

    df = align_to_model_features(df, feature_names)
    print(f"[OK] Aligned to model feature order.")
    print(f"[FEATURE PREP COMPLETE] Final shape: {df.shape}")

    return df, None
