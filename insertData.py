import pandas as pd
import mysql.connector
import numpy as np
import json
import ast
import re
import io
import math
import sys
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
    if val is None: return None
    val_str = str(val).strip()
    try:
        json.loads(val_str)
        return val_str
    except (json.JSONDecodeError, TypeError):
        pass
    if val_str.startswith(("{", "[")):
        try:
            parsed = ast.literal_eval(val_str)
            return json.dumps(parsed)
        except (ValueError, SyntaxError):
            pass
    return json.dumps(val_str)

def clean_boolean(val):
    if val is None: return None
    s = str(val).lower().strip()
    return 1 if s in ['true', '1', 't', 'yes'] else 0

# =========================================================
# 🚀 MAIN IMPORT FUNCTION
# =========================================================
def insert_data_complete(csv_path: str, table_name: str, batch_size: int = 1000):
    conn = None
    cursor = None
    
    print(f"\n🚀 Starting Import for: `{table_name}`")

    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        
        # 1️⃣ Get Schema Information
        cursor.execute(f"""
            SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, IS_NULLABLE
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_SCHEMA = '{MYSQL_CONFIG['database']}' 
            AND TABLE_NAME = '{table_name}';
        """)
        
        schema_info = cursor.fetchall()
        db_schema = {row[0]: row[1] for row in schema_info}
        col_lengths = {row[0]: row[2] for row in schema_info if row[2] is not None}
        col_nullable = {row[0]: (row[3] == 'YES') for row in schema_info}
        
        if not db_schema:
            print(f"❌ Table `{table_name}` not found!")
            return

    except Exception as e:
        print(f"❌ DB Connection Error: {e}")
        return

    # 2️⃣ Read File (Robust Mode)
    print(f"📂 Reading file: {csv_path}")
    valid_rows_df = []
    
    try:
        with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
            lines = f.readlines()
            
        print(f"ℹ️  Total Lines in CSV: {len(lines)}")
        if len(lines) < 1500:
            print(f"⚠️  NOTE: File only contains {len(lines)} lines (including header). Can only insert {len(lines)-1} rows.")

        header_line = lines[0]
        
        for i, line in enumerate(lines[1:], start=2):
            try:
                fixed_line = fix_csv_line(line)
                mini_csv = io.StringIO(header_line + fixed_line)
                row_df = pd.read_csv(mini_csv)
                valid_rows_df.append(row_df)
            except:
                try:
                    mini_csv = io.StringIO(header_line + line)
                    row_df = pd.read_csv(mini_csv, on_bad_lines='skip')
                    if not row_df.empty:
                        valid_rows_df.append(row_df)
                except:
                    pass

    except Exception as e:
        print(f"❌ File Read Error: {e}")
        return

    if not valid_rows_df:
        print("❌ No valid data found.")
        return

    # 3️⃣ Merge Data
    print(f"✅ Parsed {len(valid_rows_df)} rows. Cleaning...")
    df = pd.concat(valid_rows_df, ignore_index=True)
    
    # ---------------------------------------------------------
    # 🛠️ STEP A: Filter Garbage Rows using 'id'
    # ---------------------------------------------------------
    if 'id' in df.columns:
        # If 'id' is "Create a Quiz...", convert to NaN
        df['id'] = pd.to_numeric(df['id'], errors='coerce')
        initial_len = len(df)
        df = df.dropna(subset=['id'])
        dropped = initial_len - len(df)
        if dropped > 0:
            print(f"⚠️  Dropped {dropped} junk rows (Fixed misaligned data).")
    
    # ---------------------------------------------------------
    # 🛠️ STEP B: Drop 'id' to fix Duplicate Key Error
    # ---------------------------------------------------------
    # We remove 'id' so MySQL auto-generates a NEW, UNIQUE primary key
    if 'id' in df.columns:
        print("🔧 Dropping 'id' column from CSV to allow auto-increment (Fixes Duplicate Entry error).")
        df = df.drop(columns=['id'])

    # Align columns again (now without 'id')
    common_cols = [c for c in df.columns if c in db_schema]
    df = df[common_cols]

    # ---------------------------------------------------------
    # 🛠️ STEP C: Enforce Types & Defaults
    # ---------------------------------------------------------
    print("🧹 Enforcing schema constraints...")
    
    for col in common_cols:
        dtype = db_schema[col]
        max_len = col_lengths.get(col)
        is_nullable = col_nullable.get(col, True)

        # Integers
        if dtype in ['int', 'bigint', 'smallint', 'mediumint', 'tinyint']:
            # Special case: Tinyint often used for Boolean
            if dtype == 'tinyint' or col.startswith('is_') or col.startswith('has_'):
                 df[col] = df[col].apply(clean_boolean)
            else:
                 df[col] = pd.to_numeric(df[col], errors='coerce')
            
            if not is_nullable: df[col] = df[col].fillna(0)

        # Strings
        elif dtype in ['char', 'varchar', 'text']:
            if max_len:
                def truncate_or_null(val):
                    if pd.isna(val) or val is None: return "" if not is_nullable else None
                    s = str(val)
                    if len(s) > max_len:
                        if 'uuid' in col: return None 
                        return s[:max_len]
                    return s
                df[col] = df[col].apply(truncate_or_null)
            if not is_nullable: df[col] = df[col].fillna("")

        # JSON
        elif dtype == 'json':
            df[col] = df[col].apply(clean_json_generic)
            if not is_nullable: df[col] = df[col].fillna('{}')

    # ---------------------------------------------------------
    # 🛠️ STEP D: Sanitize (NaN -> NULL)
    # ---------------------------------------------------------
    print("🧹 Sanitizing data for MySQL...")
    
    raw_data = df.values.tolist()
    cleaned_data = []
    
    for row in raw_data:
        new_row = [None if pd.isna(x) else x for x in row]
        cleaned_data.append(tuple(new_row))

    # 4️⃣ Batch Insert
    cols_sql = ", ".join([f"`{c}`" for c in common_cols])
    placeholders = ", ".join(["%s"] * len(common_cols))
    query = f"INSERT INTO {table_name} ({cols_sql}) VALUES ({placeholders})"
    
    total_rows = len(cleaned_data)
    total_batches = math.ceil(total_rows / batch_size)
    
    print(f"🔄 Inserting {total_rows} valid rows in {total_batches} batches...")
    
    try:
        for i in range(0, total_rows, batch_size):
            batch_data = cleaned_data[i : i + batch_size]
            cursor.executemany(query, batch_data)
            conn.commit()
            print(f"   🔹 Batch {i//batch_size + 1}/{total_batches}: Inserted {len(batch_data)} rows")
            
        print("✅ SUCCESS! All valid rows inserted.")
        
    except mysql.connector.Error as err:
        print(f"❌ SQL Insert Error: {err}")
    finally:
        cursor.close()
        conn.close()

# ==========================================
# 👇 RUN IT
# ==========================================
insert_data_complete(
    r"C:\Users\dhruv sharma\Downloads\1599.csv", 
    "link_scrapper_app_htmltomarkdownconversion",
    batch_size=500
)