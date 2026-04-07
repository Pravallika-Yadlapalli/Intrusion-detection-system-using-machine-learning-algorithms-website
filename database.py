# database.py
import sqlite3
import os

DB_PATH = 'ids_database.db'


# ─────────────────────────────────────────────────────────────────────────────
def get_db():
    """
    Open and return a database connection.
    Row factory set so rows behave like dicts: row['column_name']
    Always call close() on the connection when done.
    """
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ─────────────────────────────────────────────────────────────────────────────
def init_db():
    """
    Create all tables if they don't exist.
    Safe to call on every startup — won't overwrite existing data.
    """
    conn = get_db()
    cursor = conn.cursor()

    # ── USERS TABLE ───────────────────────────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            username  TEXT    NOT NULL,
            email     TEXT    NOT NULL UNIQUE,
            password  TEXT    NOT NULL,
            created_at DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # ── RESULTS TABLE ─────────────────────────────────────────────────────
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id       INTEGER NOT NULL,
            filename      TEXT    NOT NULL,
            total_records INTEGER NOT NULL,
            normal_count  INTEGER NOT NULL,
            attack_count  INTEGER NOT NULL,
            normal_pct    REAL    NOT NULL,
            attack_pct    REAL    NOT NULL,
            scanned_at    DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id)
        )
    ''')

    conn.commit()
    conn.close()
    print("[OK] Database initialised.")


# ─────────────────────────────────────────────────────────────────────────────
# USER OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def create_user(username, email, password):
    """
    Insert a new user into the users table.

    Returns:
        (True,  user_id)        → success
        (False, error_message)  → email already exists or other error
    """
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute(
            'INSERT INTO users (username, email, password) VALUES (?, ?, ?)',
            (username.strip(), email.strip().lower(), password)
        )
        conn.commit()
        user_id = cursor.lastrowid
        print(f"[DB] New user created: id={user_id}, email={email}")
        return True, user_id
    except sqlite3.IntegrityError:
        # UNIQUE constraint on email failed
        return False, "An account with this email already exists."
    except Exception as e:
        return False, f"Registration failed: {str(e)}"
    finally:
        conn.close()


def get_user_by_email(email):
    """
    Fetch a user row by email.
    Returns the row dict or None if not found.
    """
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute(
            'SELECT * FROM users WHERE email = ?',
            (email.strip().lower(),)
        )
        return cursor.fetchone()
    finally:
        conn.close()


def get_user_by_id(user_id):
    """
    Fetch a user row by ID.
    Returns the row dict or None if not found.
    """
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute('SELECT * FROM users WHERE id = ?', (user_id,))
        return cursor.fetchone()
    finally:
        conn.close()


def verify_login(email, password):
    """
    Check email + password against the database.

    Returns:
        (True,  user_row)       → valid credentials
        (False, error_message)  → invalid credentials
    """
    user = get_user_by_email(email)

    if user is None:
        return False, "No account found with this email address."

    if user['password'] != password:
        return False, "Incorrect password. Please try again."

    return True, user


# ─────────────────────────────────────────────────────────────────────────────
# RESULTS OPERATIONS
# ─────────────────────────────────────────────────────────────────────────────

def save_result(user_id, filename, total_records,
                normal_count, attack_count, normal_pct, attack_pct):
    """
    Save a prediction result linked to the logged-in user.

    Returns:
        (True,  result_id)      → saved successfully
        (False, error_message)  → failed to save
    """
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO results
                (user_id, filename, total_records, normal_count,
                 attack_count, normal_pct, attack_pct)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (user_id, filename, total_records,
              normal_count, attack_count, normal_pct, attack_pct))
        conn.commit()
        result_id = cursor.lastrowid
        print(f"[DB] Result saved: id={result_id}, user_id={user_id}, "
              f"file={filename}")
        return True, result_id
    except Exception as e:
        return False, f"Failed to save result: {str(e)}"
    finally:
        conn.close()


def get_user_results(user_id):
    """
    Fetch all results for a specific user, newest first.
    Returns a list of row dicts.
    """
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT * FROM results
            WHERE user_id = ?
            ORDER BY scanned_at DESC
        ''', (user_id,))
        return cursor.fetchall()
    finally:
        conn.close()


def get_user_stats(user_id):
    """
    Aggregate statistics for a user's history page header.
    Returns a dict with total_scans, clean_sessions, threat_sessions,
    total_records.
    """
    conn = get_db()
    try:
        cursor = conn.cursor()
        cursor.execute('''
            SELECT
                COUNT(*)                                    AS total_scans,
                SUM(CASE WHEN attack_count = 0 THEN 1 ELSE 0 END) AS clean_sessions,
                SUM(CASE WHEN attack_count > 0 THEN 1 ELSE 0 END) AS threat_sessions,
                SUM(total_records)                          AS total_records
            FROM results
            WHERE user_id = ?
        ''', (user_id,))
        row = cursor.fetchone()
        return {
            'total_scans':     row['total_scans']     or 0,
            'clean_sessions':  row['clean_sessions']  or 0,
            'threat_sessions': row['threat_sessions'] or 0,
            'total_records':   row['total_records']   or 0,
        }
    finally:
        conn.close()
