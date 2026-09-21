"""Drive the whole Phase 9 pipeline to completion, idempotently.

Same contract as training/run_matrix.py: every stage is skippable, so
re-running continues from wherever it stopped instead of redoing work.

Why this file exists rather than a shell loop
---------------------------------------------
The Llama behavioural sweep was launched from a session-owned shell and died
mid-request when that session ended - 85 minutes of work lost with an empty log
and exit code 4. FT-004 records the same failure from training and the same
fix: run detached, make every stage resumable, and never assume a long job will
outlive the thing that started it. Launch this with WMI Win32_Process.Create so
it is parented to WmiPrvSE and sits outside any job object a session teardown
would close.

Stages
------
  llama-l2      finish the behavioural sweep over the quantized builds
  llama-bench   CPU tokens/sec, TTFT and peak RSS - the DEPLOYMENT numbers,
                measured with the CPU build and nothing else running
  llama-prune   drop artifacts that are no longer needed, to make room
  mistral-*     the same pipeline for Mistral-7B

Disk is the binding constraint: Mistral's merged bf16 plus its BF16 GGUF is
~29 GB on their own, so Llama has to be finished and pruned first.
"""
import json, os, shutil, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PY = os.path.join(ROOT, ".venv310", "Scripts", "python.exe")
L = os.path.join(ROOT, "runtime", "llama")
B = os.path.join(ROOT, "quantization", "builds")
G = os.path.join(B, "gguf")
LOGS = os.path.join(B, "logs")
RES = os.path.join(ROOT, "experiments", "generation_results.jsonl")
STATUS = os.path.join(ROOT, "quantization", "quant_status.json")
LOG = os.path.join(ROOT, "quantization", "quant_matrix.log")
PORT = 8080

LLAMA_BUILDS = ["Q8_0", "Q6_K-imat", "Q6_K", "Q5_K_M-imat", "Q5_K_M",
                "Q5_K_M-embd8", "Q4_K_M"]
MISTRAL_BUILDS = ["Q8_0", "Q6_K-imat", "Q6_K", "Q5_K_M-imat", "Q5_K_M", "Q4_K_M"]

# After every Llama gate is scored the BF16 reference has done its job: its
# numbers live in experiments/*.jsonl and it is re-creatable from the adapter
# in ~6 min. Mistral needs the space more than we need the file. Keep the two
# KLD survivors - the leader, and Q8_0 as the conservative fallback.
LLAMA_KEEP = ["vjlm-llama3b-Q6_K-imat.gguf", "vjlm-llama3b-Q8_0.gguf"]


def log(msg):
    line = "%s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def status(**kw):
    d = {}
    if os.path.exists(STATUS):
        try:
            d = json.load(open(STATUS, encoding="utf-8"))
        except Exception:
            d = {}
    d.update(kw)
    d["heartbeat"] = time.strftime("%Y-%m-%d %H:%M:%S")
    d["pid"] = os.getpid()
    json.dump(d, open(STATUS, "w", encoding="utf-8"), indent=1)


def free_gb():
    return shutil.disk_usage(ROOT).free / 2**30


def rows():
    out = []
    if os.path.exists(RES):
        for l in open(RES, encoding="utf-8"):
            l = l.strip()
            if l:
                try:
                    out.append(json.loads(l))
                except Exception:
                    pass
    return out


def scored(arm, role):
    return any(r.get("arm") == arm and r.get("role") == role for r in rows())


def health():
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=2):
            return True
    except Exception:
        return False


def serve_and_score(gguf, arm, role, label, backend="cuda", ngl=99, threads=20):
    if scored(arm, role):
        log("SKIP  %s/%s - already recorded" % (arm, role))
        return True
    if not os.path.exists(gguf):
        log("SKIP  %s - file absent" % os.path.basename(gguf))
        return True
    slog = open(os.path.join(LOGS, "server-%s.log" % role), "w",
                encoding="utf-8", errors="replace")
    cmd = [os.path.join(L, backend, "llama-server.exe"), "-m", gguf,
           "-c", "4096", "-t", str(threads), "--host", "127.0.0.1",
           "--port", str(PORT)]
    if ngl:
        cmd += ["-ngl", str(ngl)]
    log("START score %s (%s)" % (role, os.path.basename(gguf)))
    srv = subprocess.Popen(cmd, cwd=ROOT, stdout=slog, stderr=subprocess.STDOUT)
    try:
        ok = False
        for _ in range(180):
            if health():
                ok = True
                break
            if srv.poll() is not None:
                break
            time.sleep(2)
        if not ok:
            log("FAIL  server never became healthy for %s" % role)
            return False
        t0 = time.time()
        with open(os.path.join(LOGS, "eval-%s.log" % role), "w",
                  encoding="utf-8", errors="replace") as f:
            rc = subprocess.run(
                [PY, "-u", "evaluation/eval_generation.py", "--backend", "http",
                 "--arm", arm, "--role", role, "--label", label,
                 "--method", "lora", "--seed", "42"],
                cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        log("%s  score %s after %.1f min" % ("DONE " if rc == 0 else "FAIL ",
                                             role, (time.time() - t0) / 60))
        return rc == 0
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except Exception:
            srv.kill()
        slog.close()
        time.sleep(3)


def bench_cpu(name, builds):
    """Deployment numbers: CPU build, CPU only. Never the CUDA build.

    llama-bench reports prompt-processing (pp) and token-generation (tg)
    throughput. tg is the one that matters for interactive answering, and on
    CPU it is memory-bandwidth bound - which is why smaller quants are usually
    FASTER, the opposite of the intuition that more precision costs speed.
    """
    out = os.path.join(ROOT, "experiments", "quant_bench.jsonl")
    done = set()
    if os.path.exists(out):
        for l in open(out, encoding="utf-8"):
            try:
                done.add(json.loads(l)["build"])
            except Exception:
                pass
    for b in builds:
        gguf = os.path.join(G, "%s-%s.gguf" % (name, b))
        key = "%s-%s" % (name, b)
        if key in done:
            log("SKIP  bench %s" % key)
            continue
        if not os.path.exists(gguf):
            continue
        log("START bench %s (CPU)" % key)
        blog = os.path.join(LOGS, "bench-%s.log" % key)
        with open(blog, "w", encoding="utf-8", errors="replace") as f:
            rc = subprocess.run(
                [os.path.join(L, "cpu", "llama-bench.exe"), "-m", gguf,
                 "-t", "20", "-p", "512", "-n", "128", "-r", "3", "-o", "json"],
                cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        rec = {"build": key, "gguf": os.path.basename(gguf),
               "size_gb": round(os.path.getsize(gguf) / 2**30, 3),
               "threads": 20, "backend": "cpu", "rc": rc,
               "ts": time.strftime("%Y-%m-%d %H:%M")}
        try:
            txt = open(blog, encoding="utf-8", errors="replace").read()
            data = json.loads(txt[txt.index("["):txt.rindex("]") + 1])
            for d in data:
                tag = "pp" if d.get("n_prompt", 0) else "tg"
                rec[tag + "_tps"] = round(d.get("avg_ts", 0), 2)
                rec[tag + "_stddev"] = round(d.get("stddev_ts", 0), 2)
        except Exception as e:
            rec["parse_error"] = str(e)[:120]
        with open(out, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")
        log("DONE  bench %s  tg=%s t/s  pp=%s t/s"
            % (key, rec.get("tg_tps"), rec.get("pp_tps")))


def prune(keep, where=G):
    freed = 0.0
    for f in sorted(os.listdir(where)):
        if f.endswith(".gguf") and f not in keep:
            p = os.path.join(where, f)
            freed += os.path.getsize(p) / 2**30
            os.remove(p)
            log("  pruned %s" % f)
    log("pruned %.1f GB; %.1f GB free" % (freed, free_gb()))


def main():
    os.makedirs(LOGS, exist_ok=True)
    log("=" * 70)
    log("quant matrix start  pid=%d  disk free %.1f GB" % (os.getpid(), free_gb()))
    status(state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"))

    # ---- Llama: finish the behavioural sweep -------------------------------
    status(stage="llama-l2")
    for b in LLAMA_BUILDS:
        serve_and_score(os.path.join(G, "vjlm-llama3b-%s.gguf" % b),
                        "A", "l2-" + b, "llama3b-" + b)

    # ---- Llama: CPU deployment benchmarks ----------------------------------
    status(stage="llama-bench")
    bench_cpu("vjlm-llama3b", LLAMA_BUILDS)

    # ---- Llama: prune, so Mistral has room ---------------------------------
    status(stage="llama-prune")
    log("pruning Llama builds; keeping %s" % ", ".join(LLAMA_KEEP))
    prune(LLAMA_KEEP)

    log("-" * 70)
    log("LLAMA COMPLETE. %.1f GB free." % free_gb())
    status(state="llama-done", stage="llama-complete",
           finished_llama=time.strftime("%Y-%m-%d %H:%M:%S"))


MIS = "vjlm-mistral7b"
MIS_ADAPTER = "vjlm-qlora-mistral-7b-v0.3"
MIS_MERGED = MIS_ADAPTER + "-bf16"


def sh(cmd, what, env=None):
    log("START %s" % what)
    t0 = time.time()
    lg = os.path.join(LOGS, what.replace(" ", "-") + ".log")
    with open(lg, "w", encoding="utf-8", errors="replace") as f:
        rc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=f,
                            stderr=subprocess.STDOUT).returncode
    log("%s %s after %.1f min (rc=%d)" % ("DONE " if rc == 0 else "FAIL ",
                                          what, (time.time() - t0) / 60, rc))
    if rc != 0:
        tail = open(lg, encoding="utf-8", errors="replace").read()[-400:]
        log("  tail: " + tail.replace(chr(10), " | "))
    return rc == 0


def mis_path(q):
    return os.path.join(G, "%s-%s.gguf" % (MIS, q))


def mistral():
    """Mistral-7B, built in waves because the disk will not hold it all at once.

    Merged bf16 (14.5 GB) plus its BF16 GGUF (14.5 GB) is ~29 GB by itself, and
    the full quant ladder is another ~34 GB. So: merge, convert, drop the merged
    HF immediately, score L1 and compute the imatrix while the reference still
    exists, build the REQUIRED quants, then only the conditional imatrix
    variants that a narrow miss actually justifies.
    """
    status(state="running", stage="mistral-merge")
    log("=" * 70)
    log("MISTRAL start  disk free %.1f GB" % free_gb())
    merged_dir = os.path.join(B, "merged", MIS_MERGED)
    ref = os.path.join(G, "%s-BF16.gguf" % MIS)

    if not os.path.exists(ref):
        if not os.path.isdir(merged_dir):
            if free_gb() < 24:
                log("ABORT: %.1f GB free; merge plus convert needs ~29 GB."
                    % free_gb())
                status(state="blocked", stage="mistral-disk")
                return
            if not sh([PY, "-u", "training/merge_adapter.py",
                       "--adapter", MIS_ADAPTER], "mistral merge"):
                status(state="failed", stage="mistral-merge")
                return
        status(stage="mistral-convert")
        env = dict(os.environ)
        env["PYTHONPATH"] = (os.path.join(L, "convert", "gguf-py") + os.pathsep
                             + os.path.join(L, "convert"))
        if not sh([PY, "-u", os.path.join(L, "convert", "convert_hf_to_gguf.py"),
                   merged_dir, "--outfile", ref, "--outtype", "bf16"],
                  "mistral convert", env):
            status(state="failed", stage="mistral-convert")
            return

    # The merged HF weights exist only to produce the GGUF. Drop them now:
    # 14.5 GB is the difference between the ladder fitting and not.
    if os.path.isdir(merged_dir):
        shutil.rmtree(merged_dir)
        log("dropped merged HF (re-creatable in ~2 min); %.1f GB free" % free_gb())

    status(stage="mistral-l1")
    serve_and_score(ref, "B", "l1-gguf", "mistral7b-bf16-gguf")

    status(stage="mistral-imatrix")
    corpus = os.path.join(B, "imatrix-mistral.txt")
    if not os.path.exists(corpus):
        sh([PY, "-u", "quantization/make_imatrix_corpus.py", "--model", "mistral",
            "--out", corpus], "mistral imatrix corpus")
    imat = os.path.join(B, "imatrix-mistral.gguf")
    if not os.path.exists(imat):
        sh([os.path.join(L, "cuda", "llama-imatrix.exe"), "-m", ref, "-f", corpus,
            "-o", imat, "-ngl", "99", "--parse-special"], "mistral imatrix")

    # KLD logits cost ~17 MB/chunk at Mistral 32,768 vocab - a quarter of what
    # Llama 128,256 cost.
    status(stage="mistral-kld-capture")
    kldtext = os.path.join(B, "kld-mistral.txt")
    if not os.path.exists(kldtext):
        sh([PY, "-u", "quantization/make_imatrix_corpus.py", "--model", "mistral",
            "--out", kldtext, "--chunks", "80", "--dialogs", "25"],
           "mistral kld corpus")
    sh([PY, "-u", "quantization/eval_quant.py", "--capture", "--ref", ref,
        "--text", kldtext, "--chunks", "32", "--ngl", "99"], "mistral kld capture")

    status(stage="mistral-build-required")
    sh([PY, "-u", "quantization/build_gguf.py", "--merged", MIS_MERGED,
        "--name", MIS, "--only", "Q8_0,Q6_K,Q5_K_M"], "mistral build required")
    have = [mis_path(q) for q in ("Q8_0", "Q6_K", "Q5_K_M") if os.path.exists(mis_path(q))]
    if have:
        sh([PY, "-u", "quantization/eval_quant.py", "--compare", "--ref", ref,
            "--text", kldtext, "--chunks", "32", "--ngl", "99",
            "--quants", ",".join(have)], "mistral kld required")

    # Conditional imatrix builds, on the SAME pre-registered narrow-miss rule
    # that promoted Q6_K-imat on Llama: it fires on a threshold, not a hunch.
    status(stage="mistral-build-conditional")
    want = []
    try:
        kld = {}
        for line in open(os.path.join(ROOT, "experiments", "quant_results.jsonl"),
                         encoding="utf-8"):
            r = json.loads(line)
            if r["label"].startswith(MIS):
                kld[r["label"].replace(MIS + "-", "")] = r["mean_kld"]
        for q in ("Q6_K", "Q5_K_M"):
            v = kld.get(q)
            if v is None:
                continue
            if v <= 0.01:
                log("  %s KLD %.6f - passes, no imatrix retry needed" % (q, v))
            elif v <= 0.012:
                want.append(q + "-imat")
                log("  %s missed the 0.01 gate at %.6f, under 20%% over -> "
                    "building %s-imat" % (q, v, q))
            else:
                log("  %s KLD %.6f - misses by too much for an imatrix retry"
                    % (q, v))
    except Exception as e:
        log("  could not read KLD rows: %s" % e)
    if want:
        sh([PY, "-u", "quantization/build_gguf.py", "--merged", MIS_MERGED,
            "--name", MIS, "--imatrix", imat, "--only", ",".join(want)],
           "mistral build conditional")
        have2 = [mis_path(q) for q in want if os.path.exists(mis_path(q))]
        if have2:
            sh([PY, "-u", "quantization/eval_quant.py", "--compare", "--ref", ref,
                "--text", kldtext, "--chunks", "32", "--ngl", "99",
                "--quants", ",".join(have2)], "mistral kld conditional")

    status(stage="mistral-l2")
    for q in MISTRAL_BUILDS:
        if os.path.exists(mis_path(q)):
            serve_and_score(mis_path(q), "B", "l2-" + q, "mistral7b-" + q)

    status(stage="mistral-bench")
    bench_cpu(MIS, MISTRAL_BUILDS)

    log("-" * 70)
    log("MISTRAL COMPLETE. %.1f GB free." % free_gb())
    status(state="finished", stage="complete",
           finished=time.strftime("%Y-%m-%d %H:%M:%S"))


if __name__ == "__main__":
    if "--mistral" in sys.argv:
        mistral()
    else:
        main()
        mistral()
