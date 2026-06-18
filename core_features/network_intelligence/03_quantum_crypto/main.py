# main.py
# ─────────────────────────────────────────────────────────────────────────────
# Quantum-resistant encryption for sensitive data (PAN, Aadhaar, documents).
#
# This is the ONLY module other parts of TrustNet need to import from.
#
# Usage:
#   from quantum_crypto.main import generate_keypair, encrypt_text, decrypt_text
#
#   public_key, private_key = generate_keypair()
#   encrypted = encrypt_text("ABCDE1234F", public_key)   # e.g. a PAN number
#   plaintext = decrypt_text(encrypted, private_key)
#
# HOW THIS IS QUANTUM-RESISTANT:
#   RSA and ECC (the cryptography underlying almost all of today's internet)
#   can be broken by a sufficiently large quantum computer running Shor's
#   algorithm. This module instead uses ML-KEM-768 (FIPS 203, the NIST-
#   standardized algorithm formerly known as CRYSTALS-Kyber) to establish a
#   shared secret — a lattice-based scheme with no known efficient quantum
#   attack. That shared secret then becomes a one-time AES-256-GCM key used
#   to actually encrypt the data. This "KEM + symmetric cipher" pattern is
#   the same hybrid approach used in real post-quantum TLS deployments —
#   AES-256 itself is already considered quantum-safe (Grover's algorithm
#   only halves its effective strength, leaving ~128-bit security).
# ─────────────────────────────────────────────────────────────────────────────

import base64
import os
from typing import Tuple, Dict

from kyber_py.ml_kem import ML_KEM_768
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.exceptions import InvalidTag

ALGORITHM_LABEL = "ML-KEM-768 + AES-256-GCM"


def generate_keypair() -> Tuple[bytes, bytes]:
    """
    Generates a new ML-KEM-768 keypair.

    Returns:
        (public_key, private_key) — both raw bytes.
            public_key  (1184 bytes) — safe to share, used to ENCRYPT data
            private_key (2400 bytes) — must stay secret, used to DECRYPT data

    In production: generate this once per tenant/bank, store private_key in
    a secrets manager (AWS KMS, HashiCorp Vault, Azure Key Vault) — never in
    source control, a config file, or anywhere that ends up in a git commit.
    """
    public_key, private_key = ML_KEM_768.keygen()
    return public_key, private_key


# ── Persistent key storage ──────────────────────────────────────────────────
# generate_keypair() makes a NEW random keypair every call — fine for tests,
# but in production you must generate the keypair ONCE and reuse the same
# public key for every applicant. Regenerating it would make every
# previously-encrypted PAN/Aadhaar/document permanently undecryptable.

_KEYS_DIR         = os.path.join(os.path.dirname(os.path.abspath(__file__)), "keys")
_PUBLIC_KEY_PATH  = os.path.join(_KEYS_DIR, "public_key.bin")
_PRIVATE_KEY_PATH = os.path.join(_KEYS_DIR, "private_key.bin")


def load_or_create_keypair() -> Tuple[bytes, bytes]:
    """
    Loads the persisted keypair from keys/, generating and saving one the
    first time this is ever called. Call this once at app startup and reuse
    the returned keys for every request — do NOT call generate_keypair()
    directly per-request in real code.

    In production, store private_key.bin in a secrets manager (AWS KMS,
    HashiCorp Vault, Azure Key Vault) instead of a local file — this local
    file approach is for development/demo only. Make sure keys/ is in
    .gitignore so the private key never gets committed.
    """
    os.makedirs(_KEYS_DIR, exist_ok=True)

    if os.path.exists(_PUBLIC_KEY_PATH) and os.path.exists(_PRIVATE_KEY_PATH):
        with open(_PUBLIC_KEY_PATH, "rb") as f:
            public_key = f.read()
        with open(_PRIVATE_KEY_PATH, "rb") as f:
            private_key = f.read()
    else:
        public_key, private_key = generate_keypair()
        with open(_PUBLIC_KEY_PATH, "wb") as f:
            f.write(public_key)
        with open(_PRIVATE_KEY_PATH, "wb") as f:
            f.write(private_key)
        print(f"Generated new keypair and saved to {_KEYS_DIR}/")

    return public_key, private_key


def encrypt_data(plaintext: bytes, public_key: bytes) -> Dict[str, str]:
    """
    Encrypts raw bytes using post-quantum hybrid encryption.

    Args:
        plaintext:  raw bytes to encrypt (a PAN string encoded to bytes, an
                    Aadhaar number, or an entire uploaded document's bytes)
        public_key: recipient's ML-KEM-768 public key (from generate_keypair)

    Returns:
        A JSON-serialisable dict with everything needed to decrypt later:
        {
            "kem_ciphertext": str (base64),  # wraps this message's shared secret
            "nonce":          str (base64),  # AES-GCM nonce, unique per message
            "ciphertext":     str (base64),  # encrypted data + auth tag
            "algorithm":      "ML-KEM-768 + AES-256-GCM",
        }

    A fresh shared secret (and therefore a fresh AES key) is generated for
    every call — encrypting the same plaintext twice produces two
    completely different outputs, which is the correct, secure behaviour.
    """
    shared_secret, kem_ciphertext = ML_KEM_768.encaps(public_key)

    aesgcm = AESGCM(shared_secret)           # 32-byte shared secret = AES-256 key
    nonce = os.urandom(12)                   # 96-bit nonce, standard for GCM
    ciphertext = aesgcm.encrypt(nonce, plaintext, None)

    return {
        "kem_ciphertext": base64.b64encode(kem_ciphertext).decode("ascii"),
        "nonce":          base64.b64encode(nonce).decode("ascii"),
        "ciphertext":     base64.b64encode(ciphertext).decode("ascii"),
        "algorithm":      ALGORITHM_LABEL,
    }


def decrypt_data(encrypted_package: Dict[str, str], private_key: bytes) -> bytes:
    """
    Decrypts a package produced by encrypt_data().

    Args:
        encrypted_package: the dict returned by encrypt_data()
        private_key:       the matching ML-KEM-768 private key

    Returns:
        The original plaintext bytes.

    Raises:
        cryptography.exceptions.InvalidTag — if the ciphertext was tampered
        with, corrupted, or the wrong private_key was used. AES-GCM is
        authenticated, so this fails loudly instead of silently returning
        garbage data, which matters a lot when the data is a PAN or Aadhaar
        number.
    """
    kem_ciphertext = base64.b64decode(encrypted_package["kem_ciphertext"])
    nonce          = base64.b64decode(encrypted_package["nonce"])
    ciphertext     = base64.b64decode(encrypted_package["ciphertext"])

    shared_secret = ML_KEM_768.decaps(private_key, kem_ciphertext)

    aesgcm = AESGCM(shared_secret)
    return aesgcm.decrypt(nonce, ciphertext, None)


# ── Convenience wrappers for TrustNet's common data types ──────────────────

def encrypt_text(text: str, public_key: bytes) -> Dict[str, str]:
    """Encrypts a string — e.g. a PAN ('ABCDE1234F') or Aadhaar number."""
    return encrypt_data(text.encode("utf-8"), public_key)


def decrypt_text(encrypted_package: Dict[str, str], private_key: bytes) -> str:
    """Decrypts back to the original string."""
    return decrypt_data(encrypted_package, private_key).decode("utf-8")


def encrypt_file(file_path: str, public_key: bytes) -> Dict[str, str]:
    """Encrypts an entire file's contents — e.g. an uploaded document."""
    with open(file_path, "rb") as f:
        return encrypt_data(f.read(), public_key)


def decrypt_file(encrypted_package: Dict[str, str], private_key: bytes, output_path: str) -> None:
    """Decrypts a package and writes the recovered bytes to output_path."""
    plaintext = decrypt_data(encrypted_package, private_key)
    with open(output_path, "wb") as f:
        f.write(plaintext)


if __name__ == "__main__":
    # ── Quick demo ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  TrustNet — Quantum-Resistant Encryption Demo")
    print(f"  ({ALGORITHM_LABEL})")
    print("=" * 60)

    print("\nGenerating ML-KEM-768 keypair...")
    public_key, private_key = generate_keypair()
    print(f"  Public key  : {len(public_key)} bytes")
    print(f"  Private key : {len(private_key)} bytes")

    test_cases = [
        ("PAN Number",     "ABCDE1234F"),
        ("Aadhaar Number", "234812345678"),
    ]

    for label, value in test_cases:
        print(f"\n── {label}: '{value}' ──")
        encrypted = encrypt_text(value, public_key)
        print(f"  Encrypted (kem_ciphertext, truncated): {encrypted['kem_ciphertext'][:40]}...")
        print(f"  Encrypted (ciphertext, truncated)    : {encrypted['ciphertext'][:40]}...")
        decrypted = decrypt_text(encrypted, private_key)
        match = "✓" if decrypted == value else "✗ MISMATCH"
        print(f"  Decrypted: '{decrypted}'  {match}")

    print("\n── Tamper Detection Check ──")
    encrypted = encrypt_text("ABCDE1234F", public_key)
    tampered = dict(encrypted)
    tampered["ciphertext"] = tampered["ciphertext"][:-4] + "AAAA"
    try:
        decrypt_text(tampered, private_key)
        print("  ✗ FAILED — tampered ciphertext was not detected!")
    except InvalidTag:
        print("  ✓ Correctly rejected tampered ciphertext (InvalidTag)")

    print("\n── Wrong-Key Check ──")
    _, wrong_private_key = generate_keypair()
    encrypted = encrypt_text("ABCDE1234F", public_key)
    try:
        decrypt_text(encrypted, wrong_private_key)
        print("  ✗ FAILED — decrypted successfully with the wrong key!")
    except Exception as e:
        print(f"  ✓ Correctly rejected wrong private key ({type(e).__name__})")

    print("\n" + "=" * 60 + "\n")