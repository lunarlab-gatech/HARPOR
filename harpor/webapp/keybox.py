"""Sealed API keys: the worker's side of how a user's key travels from the public page.

Each worker start makes a new ECDH P-256 key pair in memory (KeyBox) and puts the public half in its
workers row: base64 of the 65-byte uncompressed point, 88 characters. The private half is never
written anywhere, so a key sealed to an earlier start can never be opened. The page seals the key
with WebCrypto (supa.js seal()): a one-time P-256 key pair, ECDH with the worker's public key,
HKDF-SHA256 with the one-time public point as salt and the AAD as info, and AES-256-GCM with a random
12-byte IV and the same AAD. It sends epk (the one-time point), iv and ct (the ciphertext with the
16-byte tag at its end, as WebCrypto writes it), each in base64. The AAD is

    harpor-key-v1|<job>|<user>|<agent>|<worker id>|<ref>

where ref is a new uuid per request. The worker builds it again from the request row, its own id
and the agent that server.py will run (aad), never from the secret row. So a key sealed for another
job, user, agent, worker or request does not open (Invalid). No error quotes the key.

Needs the cryptography package in envs/web. Only the worker imports this module, so the tunnel
mode runs without it.
"""

import base64
import binascii
import re

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

from harpor.webapp import steps

VERSION = "harpor-key-v1"
ID = re.compile(r"[A-Za-z0-9-]{1,64}")  # job, user, worker and ref are uuids: never "|", so the AAD reads one way
MAX_KEY = 512  # jobs.KEY: at most 512 characters, so ct has at most 512 + 16 bytes


class Invalid(Exception):
    """A sealed key that does not open, or is not well formed. The message never quotes the key."""


def aad(job, user, agent, worker, ref):
    """The additional data that ties a sealed key to one job, user, agent, worker start and request."""
    if agent not in steps.PROVIDER:  # the agents that take a key
        raise Invalid(f"keys are sealed for {' or '.join(steps.PROVIDER)}, not {agent!r}")
    if not all(isinstance(p, str) and ID.fullmatch(p) for p in (job, user, worker, ref)):
        raise Invalid("the job, user, worker and ref of a sealed key are uuids")
    return f"{VERSION}|{job}|{user}|{agent}|{worker}|{ref}".encode()


def b64(data):
    return base64.b64encode(data).decode()


def unb64(text, what):
    try:
        return base64.b64decode(text, validate=True)
    except (binascii.Error, TypeError, ValueError):
        raise Invalid(f"the sealed key's {what} is not base64") from None


def point(key):
    """The 65-byte uncompressed point of a key pair's public half."""
    return key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)


def peer(data, what):
    try:
        return ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), data)
    except ValueError:
        raise Invalid(f"the {what} is not a P-256 point") from None


def aes(shared, epk, info):
    """WebCrypto's deriveKey(HKDF SHA-256, salt epk, info aad) to a 256-bit AES-GCM key."""
    return AESGCM(HKDF(hashes.SHA256(), 32, salt=epk, info=info).derive(shared))


class KeyBox:
    """One worker start's ECDH P-256 key pair. public_b64 goes into the workers row, and open() opens the
    keys that the page sealed to it."""

    def __init__(self, private=None):
        self._key = private or ec.generate_private_key(ec.SECP256R1())  # private: for a fixed test vector
        self.public_b64 = b64(point(self._key))

    def __repr__(self):
        return f"KeyBox({self.public_b64[:12]}...)"

    def open(self, epk, iv, ct, aad):
        """The API key sealed in epk, iv and ct (base64) under aad, else Invalid."""
        epk, iv, ct = unb64(epk, "epk"), unb64(iv, "iv"), unb64(ct, "ct")
        if len(epk) != 65 or len(iv) != 12 or not 17 <= len(ct) <= MAX_KEY + 16:
            raise Invalid("the sealed key is malformed")
        shared = self._key.exchange(ec.ECDH(), peer(epk, "sealed key's epk"))
        try:
            key = aes(shared, epk, aad).decrypt(iv, ct, aad)
        except InvalidTag:
            raise Invalid("the sealed key does not open: it was sealed for another job, agent or worker, "
                          "or it was changed") from None
        try:
            return key.decode("ascii")
        except UnicodeDecodeError:
            raise Invalid("the sealed key is not ASCII") from None

