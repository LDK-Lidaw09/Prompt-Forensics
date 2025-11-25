# adapters/dna_store.py
import sqlite3
import os
import json
from datetime import datetime

DB = os.getenv("DB_PATH", "./dna_store.sqlite")

def init_db(db_path: str = None):
    dbp = db_path or DB
    conn = sqlite3.connect(dbp)
    cur = conn.cursor()
    cur.execute("""
    CREATE TABLE IF NOT EXISTS dna_records (
        dna_hash TEXT PRIMARY KEY,
        payload TEXT,
        signature TEXT,
        pubkey TEXT,
        created_at TEXT
    )
    """)
    cur.execute("""
    CREATE TABLE IF NOT EXISTS chat_logs (
        dna_hash TEXT PRIMARY KEY,
        prompt_text TEXT,
        output_text TEXT,
        user_id TEXT,
        attack_flag INTEGER DEFAULT 0,
        attack_info TEXT,
        created_at TEXT
    )
    """)
    conn.commit()
    conn.close()

def store_dna_record(dna_hash: str, payload: dict, signature: str, pubkey_pem: str, db_path: str = None):
    dbp = db_path or DB
    conn = sqlite3.connect(dbp)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO dna_records (dna_hash,payload,signature,pubkey,created_at) VALUES (?,?,?,?,?)",
        (dna_hash, json.dumps(payload, ensure_ascii=False), signature, pubkey_pem, datetime.utcnow().isoformat() + "Z")
    )
    conn.commit()
    conn.close()

def store_chat_log(dna_hash: str, prompt_text: str, output_text: str, user_id: str, attack_flag: bool, attack_info: dict=None, db_path: str=None):
    dbp = db_path or DB
    conn = sqlite3.connect(dbp)
    cur = conn.cursor()
    cur.execute(
        "INSERT OR REPLACE INTO chat_logs (dna_hash,prompt_text,output_text,user_id,attack_flag,attack_info,created_at) VALUES (?,?,?,?,?,?,?)",
        (dna_hash, prompt_text, output_text, user_id or "", int(bool(attack_flag)), json.dumps(attack_info, ensure_ascii=False) if attack_info else None, datetime.utcnow().isoformat() + "Z")
    )
    conn.commit()
    conn.close()
