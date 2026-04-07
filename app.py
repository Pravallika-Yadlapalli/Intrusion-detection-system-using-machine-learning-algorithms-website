# app.py
import os
import json
import pandas as pd
from functools import wraps
from flask import (Flask, render_template, request,
                   redirect, url_for, session)

from database     import (init_db, create_user, verify_login,
                           save_result, get_user_results,
                           get_user_stats, get_user_by_id)
from feature_prep import prepare_features
from predictor    import load_pipeline, run_prediction

app = Flask(__name__)
app.secret_key = 'ids_project_2024_xK9mP'

UPLOAD_FOLDER      = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}
RECORDS_FILE       = os.path.join(UPLOAD_FOLDER, '_last_records.json')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ── Initialise database on startup ────────────────────────────────────────────
init_db()

# ── Load pipeline ONCE at startup ─────────────────────────────────────────────
try:
    PIPELINE = load_pipeline('model/pipeline.pkl')
except Exception as e:
    print(f"[WARNING] Pipeline not loaded: {e}")
    PIPELINE = None


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def allowed_file(filename):
    return (
        '.' in filename and
        filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def login_required(f):
    """
    Decorator — redirects to login page if user is not logged in.
    Apply to any route that requires authentication.
    """
    @wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated


def push_error(filename, message):
    """Store error state in session → always redirects to results page."""
    session['filename']      = filename
    session['has_error']     = True
    session['error_message'] = message
    session['total_records'] = 0
    session['normal_count']  = 0
    session['attack_count']  = 0
    session['normal_pct']    = 0
    session['attack_pct']    = 0


def push_results(filename, results):
    """Store prediction results in session + write records to JSON file."""
    session['filename']        = filename
    session['has_error']       = False
    session['error_message']   = ''
    session['total_records']   = results['total_records']
    session['normal_count']    = results['normal_count']
    session['attack_count']    = results['attack_count']
    session['normal_pct']      = results['normal_pct']
    session['attack_pct']      = results['attack_pct']
    session['low_conf_count']  = results.get('low_conf_count', 0)
    session['category_counts'] = results.get('category_counts', {})

    try:
        with open(RECORDS_FILE, 'w') as f:
            json.dump(results['records'], f)
    except Exception as e:
        print(f"[WARN] Could not write records file: {e}")


def extract_attack_labels(df):
    """Extract raw attack names from the CSV label column before processing."""
    for col in ['label', 'attack_type', 'class', 'category']:
        if col in df.columns:
            labels = df[col].astype(str).str.strip().tolist()
            print(f"[INFO] Extracted {len(labels)} labels from '{col}'.")
            return labels
    return []


def fix_nslkdd_columns(df, filepath):
    """
    Fix misaligned NSL-KDD CSV — reassigns correct column names ONLY when:
      1. First column is non-numeric (string values like 'private', 'tcp')
      2. The CSV has exactly 41 columns (matching NSL-KDD format)

    This prevents irrelevant CSVs (e.g. tea_vs_coffee with 34 cols, string
    first col) from being falsely treated as misaligned NSL-KDD files.

    Returns (fixed_df, attack_labels_list).
    """
    labels    = []
    first_col = df.iloc[:, 0]
    num_cols  = df.shape[1]

    # NSL-KDD misaligned format has EXACTLY 41 columns (39 features + label + classnum)
    # AND the first column contains service names (non-numeric strings)
    NSL_KDD_COL_COUNT = 41

    is_nslkdd_format = (
        not pd.api.types.is_numeric_dtype(first_col) and
        num_cols == NSL_KDD_COL_COUNT
    )

    if is_nslkdd_format:
        print("[INFO] Detected misaligned NSL-KDD CSV (41 cols, "
              "non-numeric first col) — applying column fix.")

        CORRECT_COLS = [
            'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
            'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
            'num_compromised', 'root_shell', 'su_attempted', 'num_root',
            'num_file_creations', 'num_shells', 'num_access_files',
            'num_outbound_cmds', 'is_host_login', 'is_guest_login',
            'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
            'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
            'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
            'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
            'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
            'dst_host_serror_rate', 'dst_host_srv_serror_rate',
            'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
            'label', 'classnum'
        ]

        df = pd.read_csv(filepath, header=None, skiprows=1,
                         names=CORRECT_COLS, low_memory=False)

        # Extract real attack names BEFORE dropping label
        if 'label' in df.columns:
            labels = df['label'].astype(str).str.strip().tolist()
            print(f"[INFO] Extracted {len(labels)} real attack labels.")
            df = df.drop(columns=['label'])

        if 'duration' not in df.columns:
            df['duration'] = 0

        if 'protocol_type' not in df.columns:
            df['protocol_type'] = 'tcp'

        print(f"[INFO] CSV fixed — shape: {df.shape}")
    else:
        if num_cols != NSL_KDD_COL_COUNT:
            print(f"[INFO] CSV has {num_cols} columns (not 41) — "
                  f"treating as standard format, no column remapping.")

    return df, labels


# ─────────────────────────────────────────────────────────────────────────────
# PUBLIC ROUTES (no login required)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    # Already logged in → go to upload
    if 'user_id' in session:
        return redirect(url_for('upload'))

    error = None

    if request.method == 'POST':
        email    = request.form.get('email',    '').strip()
        password = request.form.get('password', '').strip()

        if not email or not password:
            error = 'Please fill in both email and password.'
        else:
            success, result = verify_login(email, password)
            if success:
                # Store user info in session
                session['user_id']   = result['id']
                session['username']  = result['username']
                session['email']     = result['email']
                print(f"[AUTH] Login: user_id={result['id']}, "
                      f"email={result['email']}")
                return redirect(url_for('upload'))
            else:
                error = result   # error message string

    return render_template('login.html', error=error)


@app.route('/signup', methods=['GET', 'POST'])
def signup():
    # Already logged in → go to upload
    if 'user_id' in session:
        return redirect(url_for('upload'))

    error = None

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        email    = request.form.get('email',    '').strip()
        password = request.form.get('password', '').strip()
        confirm  = request.form.get('confirm_password', '').strip()

        # Basic validation
        if not username or not email or not password:
            error = 'Please fill in all fields.'
        elif password != confirm:
            error = 'Passwords do not match.'
        elif len(password) < 6:
            error = 'Password must be at least 6 characters.'
        else:
            success, result = create_user(username, email, password)
            if success:
                # Auto-login after signup
                session['user_id']  = result
                session['username'] = username
                session['email']    = email
                print(f"[AUTH] Signup + auto-login: user_id={result}")
                return redirect(url_for('upload'))
            else:
                error = result   # error message string

    return render_template('signup.html', error=error)


@app.route('/logout')
def logout():
    user_id = session.get('user_id', 'unknown')
    session.clear()
    print(f"[AUTH] Logout: user_id={user_id}")
    return redirect(url_for('home'))


# ─────────────────────────────────────────────────────────────────────────────
# PROTECTED ROUTES (login required)
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    if request.method == 'POST':

        filename = 'unknown.csv'

        # ── Pipeline check ─────────────────────────────────────────────────
        if PIPELINE is None:
            push_error(filename,
                'ML model is not loaded. '
                'Ensure model/pipeline.pkl exists.')
            return redirect(url_for('results'))

        # ── File check ─────────────────────────────────────────────────────
        if 'file' not in request.files or \
                request.files['file'].filename == '':
            push_error(filename,
                'No file selected. Please choose a CSV file.')
            return redirect(url_for('results'))

        file     = request.files['file']
        filename = file.filename

        if not allowed_file(filename):
            push_error(filename,
                f'Invalid file type. Only CSV files are accepted.')
            return redirect(url_for('results'))

        # ── Save file ──────────────────────────────────────────────────────
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # ── Read CSV ───────────────────────────────────────────────────────
        try:
            df = pd.read_csv(filepath, low_memory=False)
        except Exception as e:
            push_error(filename,
                f'Could not read the CSV file. Error: {str(e)}')
            return redirect(url_for('results'))

        print(f"\n{'='*55}")
        print(f"[UPLOAD] {filename} — "
              f"{df.shape[0]} rows × {df.shape[1]} cols | "
              f"user_id={session['user_id']}")
        print(f"{'='*55}")

        # ── Fix column misalignment + extract real attack labels ─────────────
        # fix_nslkdd_columns extracts labels BEFORE dropping the label column
        # so we get real names like 'neptune', 'normal', 'mscan'
        df, attack_labels = fix_nslkdd_columns(df, filepath)

        # If no labels from fix (normal CSV format), try extract from df columns
        if not attack_labels:
            attack_labels = extract_attack_labels(df)

        # ── Feature preparation ────────────────────────────────────────────
        prepared_df, error = prepare_features(
            df, features_path='model/features.pkl'
        )
        if error:
            push_error(filename, error)
            return redirect(url_for('results'))

        # ── Prediction ─────────────────────────────────────────────────────
        results, error = run_prediction(
            PIPELINE, prepared_df, attack_labels
        )
        if error:
            push_error(filename, error)
            return redirect(url_for('results'))

        # ── Save result to database linked to this user ────────────────────
        user_id = session['user_id']
        saved, db_result = save_result(
            user_id       = user_id,
            filename      = filename,
            total_records = results['total_records'],
            normal_count  = results['normal_count'],
            attack_count  = results['attack_count'],
            normal_pct    = results['normal_pct'],
            attack_pct    = results['attack_pct'],
        )
        if not saved:
            print(f"[WARN] Result not saved to DB: {db_result}")

        # ── Store in session and redirect ──────────────────────────────────
        push_results(filename, results)
        print(f"[DONE] Prediction complete for user_id={user_id}.")
        return redirect(url_for('results'))

    return render_template('upload.html')


@app.route('/results')
@login_required
def results():
    """Always renders results.html — shows prediction or error."""
    has_error     = session.get('has_error',      False)
    error_message = session.get('error_message',  '')
    filename      = session.get('filename',       'Unknown')
    total_records = session.get('total_records',  0)
    normal_count  = session.get('normal_count',   0)
    attack_count  = session.get('attack_count',   0)
    normal_pct    = session.get('normal_pct',     0)
    attack_pct    = session.get('attack_pct',     0)

    records = []
    if not has_error and os.path.exists(RECORDS_FILE):
        try:
            with open(RECORDS_FILE, 'r') as f:
                records = json.load(f)
        except Exception:
            records = []

    low_conf_count  = session.get('low_conf_count',  0)
    category_counts = session.get('category_counts', {})

    return render_template(
        'results.html',
        has_error       = has_error,
        error_message   = error_message,
        filename        = filename,
        total_records   = total_records,
        normal_count    = normal_count,
        attack_count    = attack_count,
        normal_pct      = normal_pct,
        attack_pct      = attack_pct,
        low_conf_count  = low_conf_count,
        category_counts = category_counts,
        records         = records,
    )


@app.route('/history')
@login_required
def history():
    """Show only the logged-in user's scan history from the database."""
    user_id = session['user_id']
    rows    = get_user_results(user_id)
    stats   = get_user_stats(user_id)

    # Convert sqlite3.Row objects to plain dicts for Jinja
    history_data = []
    for row in rows:
        history_data.append({
            'id':            row['id'],
            'filename':      row['filename'],
            'total_records': row['total_records'],
            'normal_count':  row['normal_count'],
            'attack_count':  row['attack_count'],
            'normal_pct':    row['normal_pct'],
            'attack_pct':    row['attack_pct'],
            'scanned_at':    row['scanned_at'],
            'status':        'clean' if row['attack_count'] == 0
                             else ('warning' if row['attack_count'] / row['total_records'] < 0.15
                             else 'threat'),
        })

    return render_template(
        'history.html',
        history       = history_data,
        total_scans   = stats['total_scans'],
        clean_sessions= stats['clean_sessions'],
        threat_sessions=stats['threat_sessions'],
        total_records = stats['total_records'],
        username      = session.get('username', 'User'),
    )


if __name__ == '__main__':
    app.run(debug=True)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)