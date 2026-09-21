"""Check that CPU and GPU produce the SAME quality, not just similar speed.

Every quality number in Phase 9 - KLD and all the behavioural gates - was
measured on the CUDA backend, because it is ~3x faster for the same 147 items.
That rests on an assumption nobody had verified: that the backend does not
change what the model says.

The assumption is reasonable. Decoding is greedy (temperature 0) and the
weights are identical. But CUDA and CPU use different kernels that accumulate
floating point in a different order, so bit-identical output is not guaranteed,
and near-ties can resolve differently. VJLM deploys on CPU, so if the two ever
disagreed it is the CPU number that would count.

This scores the SELECTED build on the CPU backend and compares it to the row
already recorded from the GPU. Same model file, same items, same seed, same
decode settings - the only thing that changes is where the arithmetic happens.
"""
import json, os, subprocess, sys, time, urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY_ = os.path.join(ROOT, ".venv310", "Scripts", "python.exe")
L = os.path.join(ROOT, "runtime", "llama")
G = os.path.join(ROOT, "quantization", "builds", "gguf")
LOGS = os.path.join(ROOT, "quantization", "builds", "logs")
RES = os.path.join(ROOT, "experiments", "generation_results.jsonl")
PORT = 8080


def load():
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


def health():
    try:
        with urllib.request.urlopen("http://127.0.0.1:%d/health" % PORT, timeout=2):
            return True
    except Exception:
        return False


def run(gguf, arm, role, label):
    if any(r.get("arm") == arm and r.get("role") == role for r in load()):
        print("SKIP %s - already recorded" % role, flush=True)
        return True
    slog = open(os.path.join(LOGS, "server-%s.log" % role), "w",
                encoding="utf-8", errors="replace")
    # No -ngl: the CPU backend, which is what actually ships.
    srv = subprocess.Popen(
        [os.path.join(L, "cpu", "llama-server.exe"), "-m", gguf, "-c", "4096",
         "-t", "20", "--host", "127.0.0.1", "--port", str(PORT)],
        cwd=ROOT, stdout=slog, stderr=subprocess.STDOUT)
    try:
        ok = False
        for _ in range(240):
            if health():
                ok = True
                break
            if srv.poll() is not None:
                break
            time.sleep(2)
        if not ok:
            print("server failed to start"); return False
        print("serving %s on CPU; scoring 147 items" % os.path.basename(gguf), flush=True)
        t0 = time.time()
        with open(os.path.join(LOGS, "eval-%s.log" % role), "w",
                  encoding="utf-8", errors="replace") as f:
            rc = subprocess.run(
                [PY_, "-u", "evaluation/eval_generation.py", "--backend", "http",
                 "--arm", arm, "--role", role, "--label", label,
                 "--method", "lora", "--seed", "42"],
                cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
        print("  rc=%d after %.1f min" % (rc, (time.time() - t0) / 60), flush=True)
        return rc == 0
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=30)
        except Exception:
            srv.kill()
        slog.close()


def main():
    jobs = [(os.path.join(G, "vjlm-llama3b-Q6_K-imat.gguf"), "A",
             "l2-Q6_K-imat-cpu", "llama3b-Q6_K-imat-cpu")]
    mis = os.path.join(G, "vjlm-mistral7b-Q6_K-imat.gguf")
    if not os.path.exists(mis):
        mis = os.path.join(G, "vjlm-mistral7b-Q6_K.gguf")
    if os.path.exists(mis):
        lbl = os.path.basename(mis).replace("vjlm-mistral7b-", "").replace(".gguf", "")
        jobs.append((mis, "B", "l2-%s-cpu" % lbl, "mistral7b-%s-cpu" % lbl))

    for gguf, arm, role, label in jobs:
        if os.path.exists(gguf):
            run(gguf, arm, role, label)

    print("\n=== CPU vs GPU, same file, same items, same seed ===")
    rows = load()
    for gguf, arm, role, label in jobs:
        gpu_role = role[:-4]
        g = next((r for r in rows if r.get("arm") == arm and r.get("role") == gpu_role), None)
        c = next((r for r in rows if r.get("arm") == arm and r.get("role") == role), None)
        if not (g and c):
            print("  %-22s incomplete (gpu=%s cpu=%s)" % (role, bool(g), bool(c)))
            continue
        print("  %s" % gpu_role)
        for k in ("grounded_f1", "citation_rate", "abstain_recall",
                  "abstain_precision", "false_abstain", "mean_answer_tokens"):
            gv, cv = g.get(k), c.get(k)
            same = "IDENTICAL" if gv == cv else "DIFFERS"
            print("    %-20s gpu=%-9s cpu=%-9s %s" % (k, gv, cv, same))


if __name__ == "__main__":
    main()
