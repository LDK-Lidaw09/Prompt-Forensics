# prompt_dna.py
import json
import hashlib
from datetime import datetime
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
from typing import Dict, Tuple

# --- canonicalization & hash ---
def canonicalize(obj: Dict) -> bytes:
    """
    Canonical JSON: sort keys, stable separators, ensure_unicode.
    Returns bytes.
    """
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')

def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()

def hash_text_hex(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()

def make_prompt_dna(model_id: str,
                    model_version: str,
                    params: Dict,
                    timestamp: str,
                    prompt_text: str,
                    output_text: str,
                    session_id: str) -> Tuple[str, Dict]:
    payload = {
        "model_id": model_id,
        "model_version": model_version,
        "params": params,
        "timestamp": timestamp,
        "prompt_sha256": hash_text_hex(prompt_text),
        "output_sha256": hash_text_hex(output_text),
        "session_id": session_id
    }
    canon = canonicalize(payload)
    dna_hash = sha256_hex(canon)
    return dna_hash, payload

# --- ECDSA key generation / sign / verify ---
def generate_ecdsa_keypair() -> Tuple[bytes, bytes]:
    """
    Returns (priv_pem, pub_pem)
    """
    priv = ec.generate_private_key(ec.SECP256R1())
    priv_pem = priv.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption()
    )
    pub = priv.public_key()
    pub_pem = pub.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo
    )
    return priv_pem, pub_pem

def sign_digest(priv_pem: bytes, digest_hex: str) -> str:
    priv = serialization.load_pem_private_key(priv_pem, password=None)
    digest = bytes.fromhex(digest_hex)
    sig = priv.sign(digest, ec.ECDSA(utils.Prehashed(hashes.SHA256())))
    return sig.hex()

def verify_signature(pub_pem: bytes, digest_hex: str, sig_hex: str) -> bool:
    pub = serialization.load_pem_public_key(pub_pem)
    digest = bytes.fromhex(digest_hex)
    sig = bytes.fromhex(sig_hex)
    try:
        pub.verify(sig, digest, ec.ECDSA(utils.Prehashed(hashes.SHA256())))
        return True
    except InvalidSignature:
        return False

if __name__ == "__main__":
    # quick local test
    priv_pem, pub_pem = generate_ecdsa_keypair()
    dna, payload = make_prompt_dna("demo-model","v1",{"temp":0.0}, datetime.utcnow().isoformat()+"Z",
                                  "What is 2+2?","4","sess-1")
    sig = sign_digest(priv_pem, dna)
    print("DNA:", dna)
    print("Signature OK:", verify_signature(pub_pem, dna, sig))
