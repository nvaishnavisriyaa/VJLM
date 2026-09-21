"""Measure what quantization actually cost, against the BF16 reference.

Why this exists rather than just re-running the behavioural eval
---------------------------------------------------------------
The Phase 8 test set has 87 scorable items and SLM-001 pre-registered a noise
floor of ~0.03 F1. Quantization damage at Q5_K_M and above is normally far
smaller than that, so the behavioural eval CANNOT detect it - it can only
confirm nothing large broke. Treating "F1 didn't move" as "quantization was
free" would be reading a null result off an instrument too blunt to see the
effect.

KL-divergence per token is the sensitive instrument. Every token in the
calibration text is a paired observation between the reference and the quant,
so ~16k tokens gives ~16k paired measurements instead of 87.

What the numbers mean
---------------------
  mean KLD          0 = identical output distributions. Small is good.
  top-1 flips       share of tokens where the argmax changed at all.
  delta-p quantiles llama.cpp prints percentiles of the change in the correct
                    token's probability. SYMMETRIC percentiles mean the quant
                    is adding noise; a heavy NEGATIVE tail means it is actually
                    making the model worse. This distinction is the whole point
                    of looking at the distribution and not just the mean.

Disk
----
The reference logit file is large - roughly vocab x ctx x 1 byte per chunk, so
~70-80 MB/chunk for Llama's 128,256-token vocabulary against ~17 MB/chunk for
Mistral's 32,768. --chunks bounds it. The file is deleted after a model's runs.

Usage:
    python quantization/eval_quant.py --capture --ref <BF16.gguf> --text c.txt
    python quantization/eval_quant.py --compare --ref <BF16.gguf> --text c.txt
"""
import argparse, json, os, re, subprocess, shutil, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
LLAMA = os.path.join(ROOT, "runtime", "llama")
GGUF = os.path.join(ROOT, "quantization", "builds", "gguf")
OUT = os.path.join(ROOT, "experiments", "quant_results.jsonl")

PATTERNS = {
    # llama.cpp pads these labels with multiple spaces and appends a "+- err"
    # term, e.g. "Mean    KLD:   0.001647 +-   0.000044". An earlier version of
    # this file assumed single spaces and silently matched nothing, so every
    # row was dropped as unparseable - hence \s+ and the tolerant tails.
    "mean_kld": r"Mean\s+KLD:\s*(-?[0-9.eE+-]+)",
    "median_kld": r"Median\s+KLD:\s*(-?[0-9.eE+-]+)",
    "kld_99": r"99\.0%\s+KLD:\s*(-?[0-9.eE+-]+)",
    "max_kld": r"Maximum\s+KLD:\s*(-?[0-9.eE+-]+)",
    "mean_dp_pct": r"Mean\s+Δp:\s*(-?[0-9.]+)",
    "median_dp_pct": r"Median\s+Δp:\s*(-?[0-9.]+)",
    "p95_dp_pct": r"95\.0%\s+Δp:\s*(-?[0-9.]+)",
    "p05_dp_pct": r"\s5\.0%\s+Δp:\s*(-?[0-9.]+)",
    "rms_dp_pct": r"RMS\s+Δp\s*:\s*(-?[0-9.]+)",
    "same_top_pct": r"Same top p:\s*([0-9.]+)",
    "ppl_q": r"Mean PPL\(Q\)\s*:\s*([0-9.]+)",
    "ppl_base": r"Mean PPL\(base\)\s*:\s*([0-9.]+)",
    "ppl_ratio": r"Mean PPL\(Q\)/PPL\(base\)\s*:\s*([0-9.]+)",
    "ppl_diff": r"Mean PPL\(Q\)-PPL\(base\)\s*:\s*(-?[0-9.]+)",
    "ppl_corr_pct": r"Cor\(ln\(PPL\(Q\)\), ln\(PPL\(base\)\)\):\s*([0-9.]+)",
}


def pin(key):
    for line in open(os.path.join(LLAMA, "PINNED_TAG.txt"), encoding="utf-8"):
        if line.startswith(key + "="):
            return line.strip().split("=", 1)[1]
    return "unknown"


def free_gb():
    return shutil.disk_usage(ROOT).free / 2**30


def run(cmd, log_path):
    print("  running (output -> %s)" % os.path.basename(log_path), flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8", errors="replace") as f:
        p = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT)
    return p.returncode, time.time() - t0


def parse(log_path):
    txt = open(log_path, encoding="utf-8", errors="replace").read()
    out = {}
    for k, pat in PATTERNS.items():
        m = re.search(pat, txt, re.S)
        if m:
            try:
                out[k] = float(m.group(1))
            except ValueError:
                pass
    return out, txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="BF16 reference .gguf")
    ap.add_argument("--text", required=True, help="calibration text")
    ap.add_argument("--base-file", default=None, help="logits file path")
    ap.add_argument("--chunks", type=int, default=32)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--threads", type=int, default=20)
    ap.add_argument("--ngl", type=int, default=0,
                    help=">0 uses the CUDA build; build-time only, never a deployment number")
    ap.add_argument("--capture", action="store_true")
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--reparse", action="store_true",
                    help="re-read saved .kld.log files without re-running the "
                         "models; the logs are the raw evidence, so a parser "
                         "fix does not need hours of recomputation")
    ap.add_argument("--quants", default=None, help="comma-separated .gguf paths")
    a = ap.parse_args()

    build = "cuda" if a.ngl > 0 else "cpu"
    exe = os.path.join(LLAMA, build, "llama-perplexity.exe")
    logs = os.path.join(ROOT, "quantization", "builds", "logs")
    os.makedirs(logs, exist_ok=True)
    base_file = a.base_file or os.path.join(
        ROOT, "quantization", "builds",
        os.path.basename(a.ref).replace(".gguf", "") + ".kld")

    common = ["-m", a.ref, "-f", a.text, "-c", str(a.ctx),
              "--chunks", str(a.chunks), "-t", str(a.threads)]
    if a.ngl:
        common += ["-ngl", str(a.ngl)]

    if a.capture:
        # ~vocab x ctx bytes per chunk; over-estimate so a half-written file
        # cannot fill the disk.
        est = a.chunks * a.ctx * 128256 * 1.2 / 2**30
        print("capture reference logits: %d chunks x %d ctx" % (a.chunks, a.ctx))
        print("  estimated logits file <= %.1f GB, %.1f GB free" % (est, free_gb()))
        if free_gb() < est + 3:
            sys.exit("ABORT: not enough disk for the logits file. Lower --chunks.")
        log = os.path.join(logs, "capture.log")
        rc, secs = run([exe] + common + ["--kl-divergence-base", base_file], log)
        if rc != 0:
            sys.exit("ABORT: capture failed (rc=%d), see %s" % (rc, log))
        print("  captured in %.1f min -> %s (%.2f GB)"
              % (secs / 60, base_file, os.path.getsize(base_file) / 2**30))
        return

    if not (a.compare or a.reparse):
        sys.exit("pass --capture, --compare or --reparse")
    if not a.reparse and not os.path.exists(base_file):
        sys.exit("ABORT: no reference logits at %s. Run --capture first." % base_file)

    quants = ([q.strip() for q in a.quants.split(",")] if a.quants else
              sorted(os.path.join(GGUF, f) for f in os.listdir(GGUF)
                     if f.endswith(".gguf") and "BF16" not in f))
    rows = []
    for q in quants:
        label = os.path.basename(q).replace(".gguf", "")
        print("\n%s" % label)
        log = os.path.join(logs, label + ".kld.log")
        cmd = [exe, "-m", q, "-f", a.text, "-c", str(a.ctx), "--chunks",
               str(a.chunks), "-t", str(a.threads),
               "--kl-divergence-base", base_file, "--kl-divergence"]
        if a.ngl:
            cmd += ["-ngl", str(a.ngl)]
        if a.reparse:
            if not os.path.exists(log):
                print("  no saved log - skipping"); continue
            rc, secs = 0, 0.0
        else:
            rc, secs = run(cmd, log)
        if rc != 0:
            print("  FAILED rc=%d - row NOT recorded" % rc)
            continue
        stats, _ = parse(log)
        if "mean_kld" not in stats:
            print("  WARNING: could not parse Mean KLD; leaving the row out "
                  "rather than recording a half-parsed result. See %s" % log)
            continue
        row = dict(stats)
        row.update({"label": label, "gguf": os.path.basename(q),
                    "size_bytes": os.path.getsize(q),
                    "size_gb": round(os.path.getsize(q) / 2**30, 3),
                    "reference": os.path.basename(a.ref),
                    "ref_size_gb": round(os.path.getsize(a.ref) / 2**30, 3),
                    "chunks": a.chunks, "ctx": a.ctx,
                    "n_tokens": a.chunks * a.ctx,
                    "calibration_text": os.path.basename(a.text),
                    "kld_seconds": round(secs, 1),
                    "llama_cpp_tag": pin("tag"), "llama_cpp_commit": pin("commit"),
                    "ts": time.strftime("%Y-%m-%d %H:%M")})
        rows.append(row)
        print("  mean KLD %.6f | same top-1 %.2f%% | RMS dp %s"
              % (row["mean_kld"], row.get("same_top_p", float("nan")),
                 row.get("rms_dp")))

    with open(OUT, "a", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    if rows:
        print("\n%-22s %8s %10s %10s %10s" % ("build", "GB", "meanKLD", "sameTop1%", "RMSdp"))
        for r in sorted(rows, key=lambda x: x["size_gb"]):
            print("%-22s %8.3f %10.6f %10s %10s"
                  % (r["label"], r["size_gb"], r["mean_kld"],
                     r.get("same_top_pct", "-"), r.get("rms_dp_pct", "-")))
        print("\nappended %d rows -> %s" % (len(rows), OUT))
        print("KLD ranks the builds. It does NOT decide: the hard behavioural\n"
              "gates (abstention, false_abstain, citation_rate) still apply.")


if __name__ == "__main__":
    main()
