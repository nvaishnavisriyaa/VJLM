"""Drive the whole Phase 8 matrix unattended, and survive being killed.

    python training/run_matrix.py

Why this exists
---------------
The first attempt at arm B ran as a child of the Claude Code session. When that
process exited the training died at step 34 of 150 - 48 minutes of GPU time,
and nothing on disk, because save_strategy="epoch" put the first checkpoint at
step 75. This script is launched DETACHED so the matrix does not share a
lifetime with any session, and every stage is idempotent so that re-running
after any interruption continues instead of restarting.

Idempotence rules, checked before each stage runs:
  * a training stage is done when its adapter dir holds run.json
  * a training stage that is STARTED but unfinished resumes from its newest
    checkpoint (--resume), so a kill costs at most --save-steps
  * a scoring stage is done when generation_results.jsonl already holds a row
    for that (arm, role) pair

Inference representation is NF4 for EVERY scored configuration, including arm A
which TRAINS in bf16. That is deliberate and is what makes one untouched
Llama-3.2-3B base score a valid baseline for both A and C: training
representation is the variable under test, inference representation is held
constant, so base -> tuned isolates the adapter in both arms. Scoring therefore
always passes --qlora even when training did not.

A failing stage never aborts the matrix. It marks that arm failed, skips the
stages that depend on it, and carries on to the next arm - an overnight run
should return as much of the matrix as it can, not stop at the first problem.
"""
import json, os, subprocess, sys, time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv310", "Scripts", "python.exe")
RESULTS = os.path.join(ROOT, "experiments", "generation_results.jsonl")
ADAPTERS = os.path.join(ROOT, "training", "adapters")
LOG = os.path.join(ROOT, "experiments", "matrix.log")
STATUS = os.path.join(ROOT, "experiments", "matrix_status.json")

MISTRAL = "mistralai/Mistral-7B-Instruct-v0.3"
MISTRAL_REV = "c170c708c41dac9275d15a8fff4eca08d52bab71"
LLAMA = "meta-llama/Llama-3.2-3B-Instruct"
LLAMA_REV = "0cb88a4f764b7a12671c53f0838cd831a0843b95"

EPOCHS = "2"

# Order is B, A, C. B is the 7B arm and by far the slowest, so it runs while the
# night is longest. Both bases are already cached, so no stage can be blocked by
# a download. Each arm is scored immediately after it trains.
ARMS = [
    {"arm": "B", "base": MISTRAL, "rev": MISTRAL_REV, "qlora_train": True,
     "name": "vjlm-qlora-mistral-7b-v0.3", "label": "mistral7b-qlora",
     "base_label": None},          # arm B base already scored: run e38613e2
    {"arm": "A", "base": LLAMA, "rev": LLAMA_REV, "qlora_train": False,
     "name": "vjlm-lora-llama-3.2-3b", "label": "llama3b-lora",
     # Scored ONCE and reused as the baseline for A and C, per the standing
     # instruction not to duplicate it as a separate experimental
     # configuration. Recorded under arm "AC" so the row cannot be misread as
     # two independent measurements.
     "base_label": "llama3b-base-nf4", "base_arm": "AC"},
    {"arm": "C", "base": LLAMA, "rev": LLAMA_REV, "qlora_train": True,
     "name": "vjlm-qlora-llama-3.2-3b", "label": "llama3b-qlora",
     # --no-fp32-head is why this arm can run at all. PEFT's 4-bit preparation
     # upcasts the frozen 128,256 x 3,072 tied embedding to fp32, which pushed
     # the first attempt past 16 GB at step 45 - see FT-005. Arm A never had
     # that upcast, so removing it here leaves the NF4-vs-bf16 base as the only
     # difference between A and C, which is what the comparison is for.
     "extra": ["--no-fp32-head"],
     "base_label": None},          # reuses the arm-A base row
]


def log(msg):
    line = "%s  %s" % (time.strftime("%Y-%m-%d %H:%M:%S"), msg)
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def set_status(**kw):
    """A tiny status file so progress can be read without touching the process."""
    st = {}
    if os.path.exists(STATUS):
        try:
            st = json.load(open(STATUS, encoding="utf-8"))
        except Exception:
            st = {}
    st.update(kw)
    st["heartbeat"] = time.strftime("%Y-%m-%d %H:%M:%S")
    st["pid"] = os.getpid()
    tmp = STATUS + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATUS)


def scored(arm, role):
    if not os.path.exists(RESULTS):
        return False
    for line in open(RESULTS, encoding="utf-8"):
        try:
            r = json.loads(line)
        except Exception:
            continue
        if r.get("arm") == arm and r.get("role") == role:
            return True
    return False


def run(cmd, what):
    """Stream a stage to the log. Returns True on success."""
    log("START %s" % what)
    log("      %s" % " ".join(cmd[1:]))
    set_status(stage=what, stage_started=time.strftime("%Y-%m-%d %H:%M:%S"))
    env = dict(os.environ, PYTHONUTF8="1", PYTHONUNBUFFERED="1",
               HF_TOKEN="", HF_HUB_OFFLINE="1")
    t0 = time.time()
    with open(LOG, "a", encoding="utf-8") as f:
        p = subprocess.Popen(cmd, cwd=ROOT, env=env, stdout=f,
                             stderr=subprocess.STDOUT, text=True)
        rc = p.wait()
    mins = (time.time() - t0) / 60
    log("%s %s after %.1f min (rc=%d)"
        % ("DONE " if rc == 0 else "FAIL ", what, mins, rc))
    return rc == 0


def train(a):
    out = os.path.join(ADAPTERS, a["name"])
    if os.path.exists(os.path.join(out, "run.json")):
        log("SKIP  train arm %s - run.json already present" % a["arm"])
        return True
    cmd = [PY, "-u", "training/train_local.py", "--base", a["base"],
           "--epochs", EPOCHS, "--name", a["name"], "--resume"]
    if a["qlora_train"]:
        cmd.append("--qlora")
    cmd += a.get("extra", [])
    return run(cmd, "train arm %s (%s)" % (a["arm"], a["name"]))


def score(a, role):
    arm_label = a.get("base_arm", a["arm"]) if role == "base" else a["arm"]
    if scored(arm_label, role):
        log("SKIP  score arm %s %s - already in generation_results.jsonl"
            % (arm_label, role))
        return True
    # --qlora on EVERY scoring run: inference representation is held constant
    # at NF4 across all five configurations. See module docstring.
    cmd = [PY, "-u", "evaluation/eval_generation.py", "--backend", "hf",
           "--base", a["base"], "--revision", a["rev"], "--qlora",
           "--arm", arm_label, "--role", role, "--seed", "42"]
    if role == "tuned":
        cmd += ["--adapter", os.path.join("training", "adapters", a["name"]),
                "--label", a["label"], "--method",
                "qlora" if a["qlora_train"] else "lora"]
    else:
        cmd += ["--label", a["base_label"], "--method", "none"]
    return run(cmd, "score arm %s %s" % (arm_label, role))


def main():
    os.makedirs(os.path.dirname(LOG), exist_ok=True)
    log("=" * 70)
    log("matrix start  pid=%d  epochs=%s" % (os.getpid(), EPOCHS))
    set_status(state="running", started=time.strftime("%Y-%m-%d %H:%M:%S"),
               failed=[], completed=[])
    failed, completed = [], []

    for a in ARMS:
        arm = a["arm"]
        if not train(a):
            log("arm %s training failed; skipping its scoring stage" % arm)
            failed.append("train-" + arm)
            set_status(failed=failed)
            continue
        completed.append("train-" + arm)

        # The shared base score, when this arm is the one that owns it.
        if a.get("base_label"):
            if score(a, "base"):
                completed.append("score-base-" + arm)
            else:
                failed.append("score-base-" + arm)

        if score(a, "tuned"):
            completed.append("score-tuned-" + arm)
        else:
            failed.append("score-tuned-" + arm)
        set_status(completed=completed, failed=failed)

    log("-" * 70)
    log("matrix finished | completed=%s | failed=%s"
        % (completed or "none", failed or "none"))
    set_status(state="finished", completed=completed, failed=failed,
               finished=time.strftime("%Y-%m-%d %H:%M:%S"))
    run([PY, "-u", "evaluation/eval_generation.py", "--compare"], "compare")


if __name__ == "__main__":
    main()
