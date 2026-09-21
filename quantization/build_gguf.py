"""Build the GGUF quantization ladder for one merged model.

Phase 9 / QUANT-001. Every stage is idempotent: a build is done when its .gguf
exists, so re-running skips finished work and continues from wherever it
stopped - same contract as training/run_matrix.py.

Reference hierarchy (why there are three levels, not one)
---------------------------------------------------------
    L0  merged HF model, bf16      isolates ADAPTER MERGE error
    L1  BF16 GGUF                  isolates HF->GGUF CONVERSION / template error
    L2  quantized builds           isolates QUANTIZATION error, measured vs L1

Collapsing these would sum three different errors into one number and make a
regression impossible to localise. L1 is BF16, not F16: llama.cpp treats them
as distinct GGUF types, and bf16 is what both arms actually trained in. A
bf16 -> f16 downcast is a lossy re-rounding, so if an F16 build is ever needed
it is produced separately and the delta is measured, never assumed free.

Disk
----
The binding constraint on this machine. Every stage pre-flights, because a
truncated 14 GB file is worse than a refusal, and Windows degrades badly under
~5 GB free.

Usage:
    python quantization/build_gguf.py --merged vjlm-lora-llama-3.2-3b-bf16 \
        --name vjlm-llama3b
    python quantization/build_gguf.py --merged ... --name ... --imatrix path.gguf
"""
import argparse, json, os, shutil, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
LLAMA = os.path.join(ROOT, "runtime", "llama")
CONVERT = os.path.join(LLAMA, "convert")
BUILDS = os.path.join(ROOT, "quantization", "builds")
MERGED = os.path.join(BUILDS, "merged")
GGUF = os.path.join(BUILDS, "gguf")
MANIFEST = os.path.join(ROOT, "quantization", "builds.json")
PY = os.path.join(ROOT, ".venv310", "Scripts", "python.exe")

MIN_FREE_GB = 6.0

# The required ladder. "extra" carries llama-quantize flags beyond the type.
#
# Q4_K_M is built as a CURVE POINT so the degradation shape is visible, not as
# a candidate: per the standing instruction it ships only if it clears exactly
# the same gates as everything else.
LADDER = [
    ("Q8_0",   "q8_0",   [], "required"),
    ("Q6_K",   "q6_k",   [], "required"),
    ("Q5_K_M", "q5_k_m", [], "required"),
    ("Q4_K_M", "q4_k_m", [], "curve point - not a candidate unless it passes every gate"),
]

# Llama-only. Llama-3.2-3B's 128,256-token vocabulary makes token_embd and
# output a large share of a 3B model, so holding them at q8_0 while the rest
# goes to q5_k_m is a sensible hypothesis - TESTED here, not assumed to help.
EMBD_VARIANT = ("Q5_K_M-embd8", "q5_k_m",
                ["--token-embedding-type", "q8_0", "--output-tensor-type", "q8_0"],
                "big-vocab hypothesis")


def tag():
    for line in open(os.path.join(LLAMA, "PINNED_TAG.txt"), encoding="utf-8"):
        if line.startswith("tag="):
            return line.strip().split("=", 1)[1]
    return "unknown"


def commit():
    for line in open(os.path.join(LLAMA, "PINNED_TAG.txt"), encoding="utf-8"):
        if line.startswith("commit="):
            return line.strip().split("=", 1)[1]
    return "unknown"


def free_gb():
    return shutil.disk_usage(ROOT).free / 2**30


def preflight(need, what):
    f = free_gb()
    if f < need + MIN_FREE_GB:
        sys.exit("ABORT: %s needs ~%.1f GB, only %.1f GB free (keeping %.1f GB "
                 "headroom). Free space or drop an earlier build first."
                 % (what, need, f, MIN_FREE_GB))
    return f


def sh(cmd, what):
    print("  $ %s" % " ".join(str(c) for c in cmd[-6:]), flush=True)
    t0 = time.time()
    p = subprocess.run(cmd, cwd=ROOT)
    if p.returncode != 0:
        print("  FAILED %s (rc=%d)" % (what, p.returncode))
        return None
    return time.time() - t0


def convert(merged_dir, out):
    """L1: merged HF bf16 -> BF16 GGUF."""
    if os.path.exists(out):
        print("SKIP  convert - %s exists" % os.path.basename(out))
        return True
    src_gb = sum(os.path.getsize(os.path.join(merged_dir, f))
                 for f in os.listdir(merged_dir)) / 2**30
    preflight(src_gb * 1.1, "BF16 GGUF conversion")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.path.join(CONVERT, "gguf-py") + os.pathsep + CONVERT
    print("convert -> %s" % os.path.basename(out))
    t0 = time.time()
    p = subprocess.run([PY, "-u", os.path.join(CONVERT, "convert_hf_to_gguf.py"),
                        merged_dir, "--outfile", out, "--outtype", "bf16"],
                       cwd=ROOT, env=env, stdout=subprocess.DEVNULL)
    if p.returncode != 0:
        sys.exit("ABORT: conversion failed (rc=%d)" % p.returncode)
    print("  done in %.1f min" % ((time.time() - t0) / 60))
    return True


def quantize(src, out, qtype, extra, imatrix):
    if os.path.exists(out):
        print("SKIP  %-14s exists" % os.path.basename(out))
        return None
    # A k-quant is roughly bits/16 of the bf16 source; 0.6 is a safe over-estimate.
    preflight(os.path.getsize(src) / 2**30 * 0.6, os.path.basename(out))
    cmd = [os.path.join(LLAMA, "cpu", "llama-quantize.exe")]
    if imatrix:
        cmd += ["--imatrix", imatrix]
    cmd += list(extra) + [src, out, qtype]
    return sh(cmd, qtype)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, help="dir name under quantization/builds/merged/")
    ap.add_argument("--name", required=True, help="output prefix, e.g. vjlm-llama3b")
    ap.add_argument("--imatrix", default=None, help="imatrix .gguf; builds the imatrix variants")
    ap.add_argument("--embd-variant", action="store_true",
                    help="also build Q5_K_M with q8_0 embeddings/output (large-vocab models)")
    ap.add_argument("--only", default=None, help="comma-separated labels to build")
    a = ap.parse_args()

    merged_dir = os.path.join(MERGED, a.merged)
    ref_exists = os.path.exists(os.path.join(GGUF, "%s-BF16.gguf" % a.name))
    # The merged HF weights are only needed to CREATE the BF16 GGUF. Once that
    # exists, quantizing reads the GGUF alone - so requiring the merged dir here
    # wrongly blocked later builds after it had been deleted to reclaim disk.
    if not os.path.isdir(merged_dir) and not ref_exists:
        sys.exit("ABORT: no merged model at %s and no BF16 reference yet. "
                 "Run training/merge_adapter.py first." % merged_dir)
    os.makedirs(GGUF, exist_ok=True)

    ref = os.path.join(GGUF, "%s-BF16.gguf" % a.name)
    print("=" * 68)
    print("llama.cpp %s (%s)  |  disk free %.1f GB" % (tag(), commit(), free_gb()))
    print("=" * 68)
    convert(merged_dir, ref)

    ladder = list(LADDER)
    if a.embd_variant:
        ladder.append(EMBD_VARIANT)
    if a.imatrix:
        # imatrix concentrates its benefit at low bit-width; Q8_0 is effectively
        # unaffected. Q5_K_M both ways is the measurement that answers whether
        # it is worth anything at this precision for VJLM.
        ladder.append(("Q5_K_M-imat", "q5_k_m", [], "imatrix A/B against plain Q5_K_M"))
        # Conditional, per the pre-registered narrow-miss rule: plain Q6_K came
        # in at mean KLD 0.010682 against a 0.01 gate - over by 6.8%, well
        # inside the 20% band that says "try the imatrix before stepping up a
        # size class". Built because a rule fired, not because the number was
        # close to something we wanted.
        ladder.append(("Q6_K-imat", "q6_k", [], "imatrix retry after a narrow Q6_K miss"))
    if a.only:
        want = {s.strip() for s in a.only.split(",")}
        ladder = [x for x in ladder if x[0] in want]

    rows = []
    for label, qtype, extra, why in ladder:
        out = os.path.join(GGUF, "%s-%s.gguf" % (a.name, label))
        use_imat = a.imatrix if label.endswith("-imat") else None
        secs = quantize(ref, out, qtype, extra, use_imat)
        if not os.path.exists(out):
            print("  MISSING after build: %s" % out)
            continue
        size = os.path.getsize(out)
        rows.append({"name": a.name, "label": label, "quant_type": qtype,
                     "file": os.path.basename(out), "size_bytes": size,
                     "size_gb": round(size / 2**30, 3),
                     "imatrix": bool(use_imat),
                     "extra_flags": extra, "role": why,
                     "build_seconds": round(secs, 1) if secs else None,
                     "llama_cpp_tag": tag(), "llama_cpp_commit": commit(),
                     "source": os.path.basename(ref),
                     "merged_from": a.merged,
                     "ts": time.strftime("%Y-%m-%d %H:%M")})

    ref_size = os.path.getsize(ref)
    print("\n%-16s %8s %8s  %s" % ("build", "GB", "vs BF16", "role"))
    print("%-16s %8.3f %8s  %s" % ("BF16 (L1 ref)", ref_size / 2**30, "1.00x",
                                   "reference - every delta is measured from this"))
    for r in rows:
        print("%-16s %8.3f %7.2fx  %s" % (r["label"], r["size_gb"],
                                          r["size_bytes"] / ref_size, r["role"]))

    man = json.load(open(MANIFEST, encoding="utf-8")) if os.path.exists(MANIFEST) else []
    have = {(m["name"], m["label"]) for m in man}
    man += [r for r in rows if (r["name"], r["label"]) not in have]
    json.dump(man, open(MANIFEST, "w", encoding="utf-8"), indent=1)
    print("\nmanifest -> %s" % MANIFEST)
    print("disk free %.1f GB" % free_gb())
    print("\nSizes alone decide nothing. Next: KLD vs the BF16 reference, then the\n"
          "behavioural gates. A build ships only if it passes them.")


if __name__ == "__main__":
    main()
