"""Download a base model's weights, avoiding two Windows traps.

    python training/fetch_base.py mistralai/Mistral-7B-Instruct-v0.3

Trap 1 - duplicated weights in the repo. mistralai/Mistral-7B-Instruct-v0.3
ships the model TWICE: `consolidated.safetensors` (14.5 GB, the reference
format) and the 3-shard `model-0000x-of-00003.safetensors` set (14.5 GB, what
transformers actually loads). A plain snapshot_download pulls 29 GB. On a disk
with 58 GB free that is not a rounding error.

Trap 2 - the symlink fallback. huggingface_hub normally stores one blob and
symlinks it into the snapshot directory. Windows refuses symlinks without
Developer Mode or admin, so the hub silently COPIES instead, doubling
on-disk size again. We report which behaviour is in effect rather than
discovering it from a full disk.
"""
import argparse, os, shutil, sys

# transformers loads the sharded safetensors; everything else is either a
# duplicate of the weights or a format we do not use.
IGNORE = ["consolidated*", "*.pth", "*.bin", "*.msgpack", "*.h5", "*.gguf"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--revision", default="main")
    a = ap.parse_args()

    from huggingface_hub import snapshot_download

    free = shutil.disk_usage(os.path.abspath(os.sep)).free / 1e9
    print("free disk: %.1f GB" % free)
    if free < 20:
        sys.exit("ABORT: under 20 GB free; a 7B download plus its cache will not fit.")

    print("downloading %s (revision %s)" % (a.repo, a.revision))
    print("ignoring: %s" % ", ".join(IGNORE))
    path = snapshot_download(a.repo, revision=a.revision, ignore_patterns=IGNORE)
    print("snapshot -> %s" % path)

    # Report the real on-disk cost, following symlinks only when they are real.
    total = n_link = 0
    for root, _, files in os.walk(path):
        for f in files:
            p = os.path.join(root, f)
            if os.path.islink(p):
                n_link += 1
            else:
                try:
                    total += os.path.getsize(p)
                except OSError:
                    pass
    print("snapshot: %d real files (%.2f GB), %d symlinks" % (
        sum(len(fs) for _, _, fs in os.walk(path)) - n_link, total / 1e9, n_link))
    if n_link == 0:
        print("  NOTE: no symlinks - Windows copied the blobs, so the cache holds")
        print("        two copies. Expect roughly double the download size.")

    # Record the resolved commit so the run config can cite a revision, not a
    # moving branch name (MASTER_PLAN Phase 8: 'Save the exact versions used').
    sha = os.path.basename(os.path.normpath(path))
    print("resolved_revision=%s" % sha)


if __name__ == "__main__":
    main()
