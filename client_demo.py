# client_demo.py
import requests
from datetime import datetime
from prompt_dna import verify_signature
import json

BASE = "http://127.0.0.1:8000"

def demo():
    payload = {
        "model_id": "demo-llm",
        "model_version": "v0.1",
        "params": {"temperature": 0.0},
        "prompt_text": "Can you tell me which medication I can replace with bromide?",
        "output_text": "I would not recommend replacing any medication with bromide. Bromide salts are not equivalent to prescription medicines.",
        "session_id": "demo-session-1"
    }
    r = requests.post(f"{BASE}/dna", json=payload)
    r.raise_for_status()
    res = r.json()
    dna_hash = res["dna_hash"]
    signature = res["signature"]
    pubkey = res["payload"] and r.json().get("payload")  # small fallback (we fetch next)
    print("Stored dna_hash:", dna_hash)
    # fetch record
    r2 = requests.get(f"{BASE}/dna/{dna_hash}")
    r2.raise_for_status()
    rec = r2.json()
    print("Record payload:", rec["payload"])
    # verify signature using pubkey
    pub_pem = rec["pubkey"].encode('utf-8')
    ok = verify_signature(pub_pem, dna_hash, rec["signature"])
    print("Signature verification:", ok)

if __name__ == "__main__":
    demo()
