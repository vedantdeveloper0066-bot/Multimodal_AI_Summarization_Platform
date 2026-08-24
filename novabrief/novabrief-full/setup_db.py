import sys, os, subprocess

DB_HOST     = os.environ.get("MYSQL_HOST", "localhost")
DB_PORT     = int(os.environ.get("MYSQL_PORT", "3306"))
DB_USER     = os.environ.get("MYSQL_USER", "root")
DB_PASSWORD = os.environ.get("MYSQL_PASSWORD", "change-me")
DB_NAME     = os.environ.get("MYSQL_DATABASE", "NovaAiSummary")

print("=" * 55)
print("  NovaBrief — MySQL Setup & Test")
print("=" * 55)
print(f"\n  Host     : {DB_HOST}")
print(f"  Port     : {DB_PORT}")
print(f"  User     : {DB_USER}")
print(f"  Database : {DB_NAME}\n")

try:
    import mysql.connector
    print("[INFO] mysql-connector-python already installed.")
except ImportError:
    print("[INFO] Installing mysql-connector-python ...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'mysql-connector-python', '-q'])
    import mysql.connector
    print("[INFO] Installed successfully.")

print("\n[1/4] Testing connection to MySQL server ...")
try:
    conn = mysql.connector.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD)
    print("      OK Connected to MySQL server")
    conn.close()
except mysql.connector.Error as e:
    print(f"      FAIL Connection failed: {e}")
    print("\n  Common fixes:")
    print("  Windows : net start mysql")
    print("  Linux   : sudo systemctl start mysql")
    print("  macOS   : brew services start mysql")
    sys.exit(1)

print(f"\n[2/4] Creating database '{DB_NAME}' ...")
try:
    conn = mysql.connector.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD)
    cur = conn.cursor()
    cur.execute(f"CREATE DATABASE IF NOT EXISTS `{DB_NAME}` CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci")
    conn.commit(); cur.close(); conn.close()
    print(f"      OK Database '{DB_NAME}' ready")
except mysql.connector.Error as e:
    print(f"      FAIL {e}"); sys.exit(1)

print("\n[3/4] Creating tables ...")
try:
    conn = mysql.connector.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME, charset="utf8mb4")
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id            INT          AUTO_INCREMENT PRIMARY KEY,
            name          VARCHAR(255) NOT NULL,
            email         VARCHAR(255) UNIQUE NOT NULL,
            password_hash VARCHAR(255) NOT NULL,
            created_at    TIMESTAMP    DEFAULT CURRENT_TIMESTAMP
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """)
    print("      OK Table: users")
    cur.execute("""
        CREATE TABLE IF NOT EXISTS summaries (
            id                  INT          AUTO_INCREMENT PRIMARY KEY,
            user_id             INT          NOT NULL,
            title               VARCHAR(500),
            source_type         VARCHAR(50)  NOT NULL,
            source_info         TEXT,
            original_language   VARCHAR(20)  DEFAULT 'en',
            original_word_count INT          DEFAULT 0,
            summary_text        MEDIUMTEXT,
            summary_language    VARCHAR(20)  DEFAULT 'en',
            method              VARCHAR(150),
            audio_filename      VARCHAR(255),
            created_at          TIMESTAMP    DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE CASCADE
        ) CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci
    """)
    print("      OK Table: summaries")
    try:
        cur.execute("CREATE INDEX idx_summaries_user_created ON summaries (user_id, created_at DESC)")
        print("      OK Index: idx_summaries_user_created")
    except mysql.connector.Error as e:
        if e.errno == 1061:
            print("      OK Index: idx_summaries_user_created (already existed)")
        else:
            raise
    conn.commit(); cur.close(); conn.close()
except mysql.connector.Error as e:
    print(f"      FAIL {e}"); sys.exit(1)

print("\n[4/4] Verifying ...")
try:
    conn = mysql.connector.connect(host=DB_HOST, port=DB_PORT, user=DB_USER, password=DB_PASSWORD, database=DB_NAME)
    cur = conn.cursor()
    cur.execute("SHOW TABLES")
    tables = [r[0] for r in cur.fetchall()]
    cur.close(); conn.close()
    print(f"      Tables found: {', '.join(tables)}")
    print("\n" + "=" * 55)
    print("  Database setup complete!")
    print(f"  Database : {DB_NAME}")
    print(f"  Tables   : {', '.join(tables)}")
    print("\n  Now open config.py and set:")
    print(f"    DB_NAME     = '{DB_NAME}'")
    print(f"    DB_PASSWORD = '{DB_PASSWORD}'")
    print("\n  Then run: python app.py")
    print("=" * 55 + "\n")
except mysql.connector.Error as e:
    print(f"      FAIL {e}"); sys.exit(1)
