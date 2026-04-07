# predictor.py
import joblib
import os
import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
def load_pipeline(pipeline_path='model/pipeline.pkl'):
    """
    Load the trained pipeline from disk.
    Call this ONCE at Flask startup — not on every request.
    """
    if not os.path.exists(pipeline_path):
        raise FileNotFoundError(
            f"pipeline.pkl not found at '{pipeline_path}'. "
            f"Ensure the model/ folder exists and contains pipeline.pkl."
        )
    pipeline = joblib.load(pipeline_path)
    print(f"[OK] Pipeline loaded from '{pipeline_path}'.")
    return pipeline


# ─────────────────────────────────────────────────────────────────────────────
def get_pipeline_feature_names(pipeline):
    """
    Extract the exact feature names the pipeline was trained on.
    Works by inspecting the first step of the pipeline.
    Returns a list of feature names, or None if not determinable.
    """
    try:
        # Try getting feature names from the pipeline's first step
        first_step = pipeline.steps[0][1]
        if hasattr(first_step, 'feature_names_in_'):
            return list(first_step.feature_names_in_)
    except Exception:
        pass

    try:
        # Try getting from the pipeline itself
        if hasattr(pipeline, 'feature_names_in_'):
            return list(pipeline.feature_names_in_)
    except Exception:
        pass

    return None


# ─────────────────────────────────────────────────────────────────────────────
def align_to_pipeline(df, pipeline):
    """
    Align the DataFrame to exactly what the pipeline expects.

    This is the FINAL safety net — inspects the pipeline's expected
    feature names and adds any missing columns (like 'classnum') with 0.

    This handles the case where 'classnum' was accidentally included
    during model training (data leakage) but is absent from new data.
    """
    expected_features = get_pipeline_feature_names(pipeline)

    if expected_features is None:
        print("[WARN] Could not determine pipeline feature names. "
              "Proceeding without alignment check.")
        return df

    print(f"[INFO] Pipeline expects {len(expected_features)} features.")
    print(f"[INFO] DataFrame has {len(df.columns)} columns.")

    # Find what is missing
    missing = [f for f in expected_features if f not in df.columns]
    extra   = [c for c in df.columns if c not in expected_features]

    if missing:
        print(f"[INFO] Adding {len(missing)} missing feature(s) with 0: {missing}")
        for col in missing:
            df[col] = 0

    if extra:
        print(f"[INFO] Dropping {len(extra)} extra column(s): {extra}")
        df = df.drop(columns=extra)

    # Reorder to exactly match pipeline's expected order
    df = df[expected_features]
    print(f"[OK]  DataFrame aligned to pipeline. Final shape: {df.shape}")

    return df


# ─────────────────────────────────────────────────────────────────────────────
def normalise_label(pred):
    """Convert model output (0/1 or string) to 'normal' or 'attack'."""
    val = str(pred).strip().lower()
    if val in ('0', 'normal', 'norm'):
        return 'normal'
    return 'attack'


# ─────────────────────────────────────────────────────────────────────────────
def run_prediction(pipeline, prepared_df, attack_labels=None):
    """
    Run pipeline.predict() and build results with real attack type names.
    """

    if prepared_df is None or prepared_df.empty:
        return None, "Prepared data is empty. Cannot run prediction."

    print(f"\n[PREDICTION] Running on {len(prepared_df)} records...")

    # Final alignment
    try:
        prepared_df = align_to_pipeline(prepared_df.copy(), pipeline)
    except Exception as e:
        return None, f"Failed to align data to model format. Error: {str(e)}"

    # Predict
    try:
        predictions = pipeline.predict(prepared_df)
    except Exception as e:
        return None, (
            f"Prediction failed. The model could not process the uploaded data. "
            f"Error: {str(e)}"
        )

    print(f"[OK]  {len(predictions)} predictions generated.")

    normalised    = [normalise_label(p) for p in predictions]
    total_records = len(normalised)
    normal_count  = normalised.count('normal')
    attack_count  = normalised.count('attack')

    if total_records == 0:
        return None, "No records were returned after prediction."

    normal_pct = round((normal_count / total_records) * 100, 1)
    attack_pct = round((attack_count / total_records) * 100, 1)

    print(f"[INFO] Total   : {total_records}")
    print(f"[INFO] Normal  : {normal_count}  ({normal_pct}%)")
    print(f"[INFO] Attacks : {attack_count}  ({attack_pct}%)")

    # ── Build per-record list with real attack type names ─────────────────
    # Attack type mapping — clean up raw label names for display
    ATTACK_CATEGORY = {
        'normal':         '—',
        'neptune':        'DoS',
        'smurf':          'DoS',
        'pod':            'DoS',
        'teardrop':       'DoS',
        'land':           'DoS',
        'back':           'DoS',
        'apache2':        'DoS',
        'udpstorm':       'DoS',
        'processtable':   'DoS',
        'mailbomb':       'DoS',
        'ipsweep':        'Probe',
        'portsweep':      'Probe',
        'nmap':           'Probe',
        'satan':          'Probe',
        'saint':          'Probe',
        'mscan':          'Probe',
        'guess_passwd':   'R2L',
        'ftp_write':      'R2L',
        'imap':           'R2L',
        'phf':            'R2L',
        'multihop':       'R2L',
        'warezmaster':    'R2L',
        'warezclient':    'R2L',
        'spy':            'R2L',
        'xlock':          'R2L',
        'xsnoop':         'R2L',
        'snmpgetattack':  'R2L',
        'snmpguess':      'R2L',
        'httptunnel':     'R2L',
        'sendmail':       'R2L',
        'named':          'R2L',
        'buffer_overflow':'U2R',
        'loadmodule':     'U2R',
        'perl':           'U2R',
        'rootkit':        'U2R',
        'xterm':          'U2R',
        'ps':             'U2R',
        'sqlattack':      'U2R',
        'worm':           'U2R',
    }

    records = []
    for i, label in enumerate(normalised):

        # Get raw attack name from CSV labels if available
        raw_label = ''
        if attack_labels and i < len(attack_labels):
            raw_label = str(attack_labels[i]).strip().lower()

        # Determine attack type to display
        if label == 'normal':
            attack_type = '—'
        elif raw_label and raw_label != 'normal':
            # Map raw label to category, fallback to capitalised raw label
            attack_type = ATTACK_CATEGORY.get(
                raw_label,
                raw_label.replace('_', ' ').title()
            )
        else:
            attack_type = 'Unknown'

        records.append({
            'id':          f"REC-{str(i + 1).zfill(4)}",
            'prediction':  label,
            'attack_type': attack_type,
            'raw_label':   raw_label if raw_label else '—',
        })

    return {
        'total_records': total_records,
        'normal_count':  normal_count,
        'attack_count':  attack_count,
        'normal_pct':    normal_pct,
        'attack_pct':    attack_pct,
        'records':       records,
        'error':         None,
    }, None