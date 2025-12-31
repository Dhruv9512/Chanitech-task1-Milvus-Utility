import pandas as pd
import mysql.connector
import numpy as np
import json
import ast
import re
import io
import math
from dotenv import load_dotenv

load_dotenv()

# ⚙️ CONFIGURATION
MYSQL_CONFIG = {
    "host": "localhost",
    "user": "root",
    "password": "Mrdhruv123",
    "database": "Task1_Milvus",
    "port": "3306"
}

# =========================================================
# 🛠️ HELPER: Robust Line-by-Line CSV Fixer
# =========================================================
def fix_csv_line(line):
    """
    Fixes a single CSV line by escaping quotes inside JSON fields.
    """
    pattern = re.compile(r'(,|^)"(\[.*?\]|\{.*?\})"(?=,|$|\n)', re.DOTALL)

    def replacer(m):
        prefix = m.group(1) if m.group(1) else ""
        body = m.group(2)
        fixed_body = body.replace('"', '""')
        return prefix + '"' + fixed_body + '"'

    return pattern.sub(replacer, line)

# =========================================================
# 🧹 CLEANERS
# =========================================================
def clean_json_generic(val):
    if val is None:
        return None

    val_str = str(val).strip()

    try:
        json.loads(val_str)
        return val_str
    except Exception:
        pass

    if val_str.startswith(("{", "[")):
        try:
            parsed = ast.literal_eval(val_str)
            return json.dumps(parsed)
        except Exception:
            pass

    return json.dumps(val_str)

def clean_boolean(val):
    if val is None:
        return None
    s = str(val).lower().strip()
    return 1 if s in ['true', '1', 't', 'yes'] else 0

# =========================================================
# 🚀 MAIN IMPORT FUNCTION
# =========================================================
def insert_data_complete(csv_path: str, table_name: str, batch_size: int = 1000):

    print(f"\n🚀 Starting Import for: `{table_name}`")

    # ---------------- DB CONNECT ----------------
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
    except Exception as e:
        print(f"❌ DB Connection Error: {e}")
        return

    # ---------------- SCHEMA READ ----------------
    cursor.execute(f"""
        SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
        FROM INFORMATION_SCHEMA.COLUMNS
        WHERE TABLE_SCHEMA = '{MYSQL_CONFIG['database']}'
        AND TABLE_NAME = '{table_name}';
    """)

    schema_info = cursor.fetchall()
    if not schema_info:
        print(f"❌ Table `{table_name}` not found!")
        return

    db_schema = {r[0]: r[1] for r in schema_info}
    col_lengths = {r[0]: r[2] for r in schema_info if r[2]}
    col_nullable = {r[0]: (r[3] == 'YES') for r in schema_info}

    # 🔥 Identify datetime columns
    datetime_columns = [
        col for col, dtype in db_schema.items()
        if dtype in ['datetime', 'timestamp', 'date']
    ]

    if datetime_columns:
        print(f"🗑️ Dropping datetime columns from insert: {datetime_columns}")

    # ---------------- FILE READ ----------------
    print(f"📂 Reading file: {csv_path}")
    valid_rows_df = []

    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    print(f"ℹ️  Total Lines in CSV: {len(lines)}")

    header_line = lines[0]

    for line in lines[1:]:
        try:
            fixed_line = fix_csv_line(line)
            mini_csv = io.StringIO(header_line + fixed_line)
            row_df = pd.read_csv(mini_csv)
            valid_rows_df.append(row_df)
        except Exception:
            pass

    if not valid_rows_df:
        print("❌ No valid data found.")
        return

    # ---------------- MERGE ----------------
    df = pd.concat(valid_rows_df, ignore_index=True)
    print(f"✅ Parsed {len(df)} rows. Cleaning...")

    # ---------------- DROP GARBAGE ROWS ----------------
    if 'id' in df.columns:
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
        before = len(df)
        df = df.dropna(subset=['id'])
        dropped = before - len(df)
        if dropped:
            print(f"⚠️ Dropped {dropped} junk rows")

    # ---------------- DROP ID ----------------
    if 'id' in df.columns:
        print("🔧 Dropping 'id' column to allow auto-increment")
        df = df.drop(columns=['id'])

    # ---------------- ALIGN COLUMNS ----------------
    common_cols = [
        c for c in df.columns
        if c in db_schema and c not in datetime_columns
    ]
    df = df[common_cols]

    # ---------------- TYPE ENFORCEMENT ----------------
    print("🧹 Enforcing schema constraints...")

    for col in common_cols:
        dtype = db_schema[col]
        max_len = col_lengths.get(col)
        nullable = col_nullable.get(col, True)

        if dtype in ['int', 'bigint', 'smallint', 'mediumint', 'tinyint']:
            if dtype == 'tinyint' or col.startswith(('is_', 'has_')):
                df[col] = df[col].apply(clean_boolean)
            else:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            if not nullable:
                df[col] = df[col].fillna(0)

        elif dtype in ['char', 'varchar', 'text']:
            def truncate(val):
                if pd.isna(val):
                    return None if nullable else ""
                s = str(val)
                return s[:max_len] if max_len and len(s) > max_len else s

            df[col] = df[col].apply(truncate)
            if not nullable:
                df[col] = df[col].fillna("")

        elif dtype == 'json':
            df[col] = df[col].apply(clean_json_generic)
            if not nullable:
                df[col] = df[col].fillna("{}")

    # ---------------- SANITIZE ----------------
    cleaned_data = [
        tuple(None if pd.isna(x) else x for x in row)
        for row in df.values.tolist()
    ]

    # ---------------- INSERT ----------------
    cols_sql = ", ".join(f"`{c}`" for c in common_cols)
    placeholders = ", ".join(["%s"] * len(common_cols))
    query = f"INSERT INTO {table_name} ({cols_sql}) VALUES ({placeholders})"

    total_rows = len(cleaned_data)
    total_batches = math.ceil(total_rows / batch_size)

    print(f"🔄 Inserting {total_rows} valid rows in {total_batches} batches...")

    try:
        for i in range(0, total_rows, batch_size):
            batch = cleaned_data[i:i + batch_size]
            cursor.executemany(query, batch)
            conn.commit()
            print(f"   🔹 Batch {i // batch_size + 1}/{total_batches} inserted")

        print("✅ SUCCESS! All rows inserted safely.")

    except mysql.connector.Error as err:
        print(f"❌ SQL Insert Error: {err}")

    finally:
        cursor.close()
        conn.close()

# =========================================================
# ▶ RUN
# =========================================================
insert_data_complete(
    r"C:\Users\dhruv sharma\Downloads\GFG LLM DATA 2.csv",
    "link_scrapper_app_htmltomarkdownconversion",
    batch_size=500
)
