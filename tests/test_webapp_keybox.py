import base64
import hashlib
import json
import shutil
import subprocess

import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from fake_supabase import seal, sealed

from harpor.webapp import keybox

JOB, USER = "6f1c0d2e-0000-4000-8000-000000000001", "9a7e0d2e-0000-4000-8000-000000000002"
WORKER, REF = "3b2a1c0d-0000-4000-8000-000000000003", "c4d5e6f7-0000-4000-8000-000000000004"
KEY = "sk-ant-fake-harpor-keybox-vector"
# Sealed once by node 20's WebCrypto (the code of supa.js seal(), below) to the fixed key pair of fixed_box().
VECTOR = {"epk": "BFWouWeZELOANEWAlsP4+eBbCWvfLcP1Hsxsd4UMrlTor+mO6bmgSEIrrO6h8GTAVsZzUsi/688we4NXEIvUNCM=",
          "iv": "Mbu66KFw233DVqVv", "ct": "JTQMEK7z05uJlWVieHsBenNHUBgvvcUXh8f+wxU70Me1UvINwcjlRIX36G5FFw0H"}
VECTOR_PUBLIC = "BO0vEiE2FNrk5I3YJfE41MtVCTomRxlOor1cwIr9/Jhw0J3LdAz0ugD88AnyZMtAUJH0jyUUPUeRUzHJaQzYm6k="
# The page's side, as supa.js seal() does it in the browser; node 20 has the same globals (crypto, btoa, atob).
NODE = r"""
const subtle = globalThis.crypto.subtle, enc = new TextEncoder(), dec = new TextDecoder();
const b64 = (u8) => btoa(String.fromCharCode(...u8)), unb64 = (s) => Uint8Array.from(atob(s), (c) => c.charCodeAt(0));
const curve = { name: "ECDH", namedCurve: "P-256" };
async function seal(apiKey, job, uid, agent, w) {
  const ref = crypto.randomUUID();
  const wk = await subtle.importKey("raw", unb64(w.public_key), curve, false, []);
  const eph = await subtle.generateKey(curve, false, ["deriveBits"]);
  const epk = new Uint8Array(await subtle.exportKey("raw", eph.publicKey));
  const shared = await subtle.deriveBits({ name: "ECDH", public: wk }, eph.privateKey, 256);
  const aad = enc.encode(`harpor-key-v1|${job}|${uid}|${agent}|${w.id}|${ref}`);
  const hk = await subtle.importKey("raw", shared, "HKDF", false, ["deriveKey"]);
  const aes = await subtle.deriveKey({ name: "HKDF", hash: "SHA-256", salt: epk, info: aad }, hk,
    { name: "AES-GCM", length: 256 }, false, ["encrypt"]);
  const iv = crypto.getRandomValues(new Uint8Array(12));
  const ct = new Uint8Array(await subtle.encrypt({ name: "AES-GCM", iv, additionalData: aad }, aes,
    enc.encode(apiKey)));
  return { ref, worker_id: w.id, agent, epk: b64(epk), iv: b64(iv), ct: b64(ct) };
}
async function pair() {           // a worker's key pair in WebCrypto, to open what Python seals
  const k = await subtle.generateKey(curve, true, ["deriveBits"]);
  return { public_key: b64(new Uint8Array(await subtle.exportKey("raw", k.publicKey))),
           jwk: await subtle.exportKey("jwk", k.privateKey) };
}
async function open(jwk, s, aadText) {
  const priv = await subtle.importKey("jwk", jwk, curve, false, ["deriveBits"]);
  const epk = unb64(s.epk), aad = enc.encode(aadText);
  const pub = await subtle.importKey("raw", epk, curve, false, []);
  const shared = await subtle.deriveBits({ name: "ECDH", public: pub }, priv, 256);
  const hk = await subtle.importKey("raw", shared, "HKDF", false, ["deriveKey"]);
  const aes = await subtle.deriveKey({ name: "HKDF", hash: "SHA-256", salt: epk, info: aad }, hk,
    { name: "AES-GCM", length: 256 }, false, ["decrypt"]);
  return dec.decode(await subtle.decrypt({ name: "AES-GCM", iv: unb64(s.iv), additionalData: aad }, aes, unb64(s.ct)));
}
const a = JSON.parse(process.argv[2]);
(async () => {
  const out = a.op === "seal" ? await seal(a.key, a.job, a.user, a.agent, a.worker)
            : a.op === "pair" ? await pair() : await open(a.jwk, a.sealed, a.aad);
  console.log(JSON.stringify(out));
})().catch((e) => { console.error(e.name); process.exit(1); });
"""

needs_node = pytest.mark.skipif(shutil.which("node") is None, reason="node is not installed")


def node(tmp_path, **args):
    script = tmp_path / "seal.js"
    script.write_text(NODE)
    out = subprocess.run(["node", str(script), json.dumps(args)], capture_output=True, text=True, timeout=60)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def fixed_box():
    d = int.from_bytes(hashlib.sha256(b"harpor keybox test vector").digest(), "big")
    return keybox.KeyBox(ec.derive_private_key(d, ec.SECP256R1()))


def test_round_trip_and_the_public_key_form():
    box = keybox.KeyBox()
    raw = base64.b64decode(box.public_b64)
    assert len(box.public_b64) == 88 and len(raw) == 65 and raw[0] == 4  # the workers row's check: 88 characters
    s = seal(box.public_b64, KEY, keybox.aad(JOB, USER, "opus", WORKER, REF))
    assert [len(s["epk"]), len(s["iv"])] == [88, 16] and 24 <= len(s["ct"]) <= 704  # job_secrets' checks
    assert box.open(s["epk"], s["iv"], s["ct"], keybox.aad(JOB, USER, "opus", WORKER, REF)) == KEY
    longest = "sk-" + "a" * 509
    s = seal(box.public_b64, longest, keybox.aad(JOB, USER, "astra", WORKER, REF))
    assert len(s["ct"]) == 704
    assert box.open(s["epk"], s["iv"], s["ct"], keybox.aad(JOB, USER, "astra", WORKER, REF)) == longest
    assert box.public_b64[:12] in repr(box) and "_key" not in repr(box)


def test_a_changed_aad_or_ciphertext_does_not_open():
    box = keybox.KeyBox()
    s = seal(box.public_b64, KEY, keybox.aad(JOB, USER, "opus", WORKER, REF))
    other = "00000000-0000-4000-8000-0000000000ff"
    for aad in (keybox.aad(other, USER, "opus", WORKER, REF), keybox.aad(JOB, other, "opus", WORKER, REF),
                keybox.aad(JOB, USER, "astra", WORKER, REF), keybox.aad(JOB, USER, "opus", other, REF),
                keybox.aad(JOB, USER, "opus", WORKER, other)):
        with pytest.raises(keybox.Invalid, match="does not open") as e:
            box.open(s["epk"], s["iv"], s["ct"], aad)
        assert KEY not in str(e.value)
    aad = keybox.aad(JOB, USER, "opus", WORKER, REF)
    ct = bytearray(base64.b64decode(s["ct"]))
    ct[3] ^= 1
    iv = bytearray(base64.b64decode(s["iv"]))
    iv[0] ^= 1
    for epk, iv_, ct_ in ((s["epk"], s["iv"], keybox.b64(ct)), (s["epk"], keybox.b64(iv), s["ct"]),
                          (seal(box.public_b64, KEY, aad)["epk"], s["iv"], s["ct"])):
        with pytest.raises(keybox.Invalid, match="does not open"):
            box.open(epk, iv_, ct_, aad)
    with pytest.raises(keybox.Invalid, match="does not open"):  # a new worker start cannot open it
        keybox.KeyBox().open(s["epk"], s["iv"], s["ct"], aad)


def test_malformed_parts_are_refused_with_fixed_messages():
    box = keybox.KeyBox()
    aad = keybox.aad(JOB, USER, "opus", WORKER, REF)
    s = seal(box.public_b64, KEY, aad)
    epk = base64.b64decode(s["epk"])
    bad = [({**s, "epk": "not base64!"}, "epk is not base64"), ({**s, "iv": None}, "iv is not base64"),
           ({**s, "epk": keybox.b64(epk[:64])}, "malformed"), ({**s, "iv": keybox.b64(bytes(16))}, "malformed"),
           ({**s, "ct": keybox.b64(bytes(16))}, "malformed"), ({**s, "ct": keybox.b64(bytes(529))}, "malformed"),
           ({**s, "epk": keybox.b64(b"\x04" + bytes(64))}, "not a P-256 point"),
           ({**s, "epk": keybox.b64(b"\x05" + epk[1:])}, "not a P-256 point")]
    for parts, why in bad:
        with pytest.raises(keybox.Invalid, match=why):
            box.open(parts["epk"], parts["iv"], parts["ct"], aad)
    s = seal(box.public_b64, "sk-ant-été", aad)  # the page encodes UTF-8; keys are ASCII
    with pytest.raises(keybox.Invalid, match="not ASCII"):
        box.open(s["epk"], s["iv"], s["ct"], aad)


def test_aad_form_and_its_checks():
    assert keybox.aad(JOB, USER, "astra", WORKER, REF) == f"harpor-key-v1|{JOB}|{USER}|astra|{WORKER}|{REF}".encode()
    for agent in ("none", False, None, "Opus"):
        with pytest.raises(keybox.Invalid, match="sealed for astra or opus"):
            keybox.aad(JOB, USER, agent, WORKER, REF)
    for parts in ((JOB + "|x", USER, WORKER, REF), (JOB, None, WORKER, REF), (JOB, USER, "", REF),
                  (JOB, USER, WORKER, "a" * 65), (JOB, USER, WORKER, "ref with space")):
        with pytest.raises(keybox.Invalid, match="are uuids"):
            keybox.aad(parts[0], parts[1], "opus", parts[2], parts[3])


def test_sealed_matches_what_relay_takes():
    box = keybox.KeyBox()
    s = sealed({"id": WORKER, "public_key": box.public_b64}, KEY, JOB, USER, "opus")
    assert set(s) == {"ref", "worker_id", "agent", "epk", "iv", "ct"} and s["worker_id"] == WORKER
    assert box.open(s["epk"], s["iv"], s["ct"], keybox.aad(JOB, USER, "opus", WORKER, s["ref"])) == KEY
    assert sealed({"id": WORKER, "public_key": box.public_b64}, KEY, JOB, USER, "opus", ref=REF)["ref"] == REF


def test_a_key_that_node_sealed_once_still_opens():
    box = fixed_box()
    assert box.public_b64 == VECTOR_PUBLIC
    aad = keybox.aad(JOB, USER, "opus", WORKER, REF)
    assert box.open(VECTOR["epk"], VECTOR["iv"], VECTOR["ct"], aad) == KEY
    with pytest.raises(keybox.Invalid):
        box.open(VECTOR["epk"], VECTOR["iv"], VECTOR["ct"], keybox.aad(JOB, USER, "astra", WORKER, REF))


@needs_node
def test_node_seals_and_python_opens(tmp_path):
    box = keybox.KeyBox()
    s = node(tmp_path, op="seal", key=KEY, job=JOB, user=USER, agent="opus",
             worker={"id": WORKER, "public_key": box.public_b64})
    assert set(s) == {"ref", "worker_id", "agent", "epk", "iv", "ct"}
    assert box.open(s["epk"], s["iv"], s["ct"], keybox.aad(JOB, USER, "opus", s["worker_id"], s["ref"])) == KEY
    with pytest.raises(keybox.Invalid):  # sealed for opus: an astra request does not open it
        box.open(s["epk"], s["iv"], s["ct"], keybox.aad(JOB, USER, "astra", s["worker_id"], s["ref"]))


@needs_node
def test_python_seals_and_node_opens(tmp_path):
    p = node(tmp_path, op="pair")
    aad = keybox.aad(JOB, USER, "astra", WORKER, REF)
    s = seal(p["public_key"], "sk-fake-1234", aad)
    assert node(tmp_path, op="open", jwk=p["jwk"], sealed=s, aad=aad.decode()) == "sk-fake-1234"

