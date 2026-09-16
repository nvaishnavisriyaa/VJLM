"""Measure local CPU inference speed and peak RAM for a GGUF build.

This is what decides which model SIZES are viable on the target machine.
Run per candidate model / quantization: results go into DECISIONS.md.
"""
import os, sys, time, json, argparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def rss_mb():
    try:
        import ctypes, ctypes.wintypes as w

        class PMC(ctypes.Structure):
            _fields_ = [("cb", w.DWORD), ("PageFaultCount", w.DWORD),
                        ("PeakWorkingSetSize", ctypes.c_size_t),
                        ("WorkingSetSize", ctypes.c_size_t),
                        ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                        ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                        ("PagefileUsage", ctypes.c_size_t),
                        ("PeakPagefileUsage", ctypes.c_size_t)]
        c = PMC(); c.cb = ctypes.sizeof(PMC)
        ctypes.windll.psapi.GetProcessMemoryInfo(
            ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb)
        return round(c.PeakWorkingSetSize / 1e6)
    except Exception:
        return -1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--threads", type=int, default=0, help="0 = sweep")
    ap.add_argument("--ntok", type=int, default=128)
    a = ap.parse_args()

    from llama_cpp import Llama
    size_mb = round(os.path.getsize(a.model) / 1e6)
    threads = [a.threads] if a.threads else [4, 8, 12, 14]
    prompt = [{"role": "system", "content": "You are a helpful assistant."},
              {"role": "user", "content":
               "Summarise in four sentences what an academic regulation document typically contains."}]

    print("model: %s  (%d MB)" % (os.path.basename(a.model), size_mb))
    rows = []
    for nt in threads:
        t0 = time.time()
        llm = Llama(model_path=a.model, n_ctx=2048, n_threads=nt, verbose=False)
        load = time.time() - t0

        t1 = time.time()
        first = None
        ntok = 0
        for ch in llm.create_chat_completion(messages=prompt, max_tokens=a.ntok,
                                             temperature=0.2, stream=True):
            d = ch["choices"][0].get("delta", {})
            if d.get("content"):
                if first is None:
                    first = time.time() - t1
                ntok += 1
        total = time.time() - t1
        tps = ntok / total if total else 0
        peak = rss_mb()
        rows.append({"threads": nt, "load_s": round(load, 1), "ttft_s": round(first or 0, 2),
                     "tok_s": round(tps, 1), "peak_rss_mb": peak, "size_mb": size_mb})
        print("  threads=%-3d load=%5.1fs  first_token=%5.2fs  %5.1f tok/s  peak_rss=%d MB"
              % (nt, load, first or 0, tps, peak))
        del llm

    best = max(rows, key=lambda r: r["tok_s"])
    print("\nBEST: threads=%d  %.1f tok/s  peak_rss=%d MB" %
          (best["threads"], best["tok_s"], best["peak_rss_mb"]))
    out = os.path.join(ROOT, "experiments", "runtime_bench.jsonl")
    with open(out, "a", encoding="utf-8") as f:
        for r in rows:
            r["model"] = os.path.basename(a.model)
            f.write(json.dumps(r) + "\n")
    print("appended ->", out)


if __name__ == "__main__":
    main()
