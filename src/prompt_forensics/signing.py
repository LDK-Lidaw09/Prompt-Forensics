"""ECDSA P-256 signatures for evidence records (thesis section 4.2.2).

A SHA-256 hash alone shows that a record changed only if the original hash was
kept somewhere safe: anyone who can edit the database can also recompute the
hash. Signing each record digest with a private key that is kept outside the
database closes that gap: a modified record no longer verifies against the
public key.

The private key is stored as a PEM file (``SIGNING_KEY_PATH``) and created on
first use, with the public key next to it (``*.pub.pem``). Keep the private key
out of version control (``keys/`` is ignored by Git) and archive the public key
with the results.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec, utils


@dataclass(frozen=True)
class SigningKey:
    """A loaded P-256 key pair."""

    private_key: ec.EllipticCurvePrivateKey

    @property
    def public_key_pem(self) -> str:
        return self.private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode("ascii")

    def sign_digest(self, digest_hex: str) -> str:
        """Sign a SHA-256 digest given as hex. Returns the DER signature as hex."""
        signature = self.private_key.sign(
            bytes.fromhex(digest_hex), ec.ECDSA(utils.Prehashed(hashes.SHA256()))
        )
        return signature.hex()


def generate_key() -> SigningKey:
    """Create a new random P-256 key pair."""
    return SigningKey(ec.generate_private_key(ec.SECP256R1()))


def load_or_create_key(path: str | os.PathLike[str]) -> SigningKey:
    """Load the private key at ``path``, creating it (mode 0600) if missing."""
    path = Path(path)
    if path.exists():
        key = serialization.load_pem_private_key(path.read_bytes(), password=None)
        if not isinstance(key, ec.EllipticCurvePrivateKey):
            raise ValueError(f"{path} does not contain an EC private key")
        return SigningKey(key)

    signing_key = generate_key()
    path.parent.mkdir(parents=True, exist_ok=True)
    pem = signing_key.private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )
    # Create the file with owner-only permissions before writing the key.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "wb") as handle:
        handle.write(pem)
    # The public key is written next to it, to be archived with the results.
    public_key_path(path).write_text(signing_key.public_key_pem, encoding="ascii")
    return signing_key


def public_key_path(private_key_path: str | os.PathLike[str]) -> Path:
    """``keys/psi_signing_key.pem`` -> ``keys/psi_signing_key.pub.pem``."""
    path = Path(private_key_path)
    return path.with_name(path.stem + ".pub.pem")


def verify_signature(public_key_pem: str, digest_hex: str, signature_hex: str) -> bool:
    """Return True if ``signature_hex`` is a valid signature of ``digest_hex``."""
    public_key = serialization.load_pem_public_key(public_key_pem.encode("ascii"))
    if not isinstance(public_key, ec.EllipticCurvePublicKey):
        return False
    try:
        public_key.verify(
            bytes.fromhex(signature_hex),
            bytes.fromhex(digest_hex),
            ec.ECDSA(utils.Prehashed(hashes.SHA256())),
        )
        return True
    except (InvalidSignature, ValueError):
        return False
