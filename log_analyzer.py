import glob, re, json
paths = sorted(list(set(glob.glob("logs/*_2220*.out") + glob.glob("logs/*22198*.out") + glob.glob("logs/*2219*.out"))))
errors = {}
files_examined = 0
for p in paths:
    try:
        with open(p, "r", errors="ignore") as f:
            txt = f.read()
    except:
        continue
    files_examined += 1
    # classify
    if re.search(r"CUDA out of memory|Out of memory|OutOfMemoryError", txt, re.I):
        k = "oom"
    elif "InvalidHeaderDeserialization" in txt or "InvalidHeader" in txt:
        k = "invalid_header"
    elif re.search(r"couldn.t connect to|Could not reach|Failed to reach", txt, re.I):
        k = "network"
    elif re.search(r"Generation failed", txt, re.I):
        k = "generation_failed"
    elif re.search(r"Saved summary to|Saved LaTeX table to|Saved markdown summary", txt):
        k = "success"
    else:
        # look for Traceback
        if "Traceback (most recent call last)" in txt:
            lines = [l for l in txt.splitlines() if l.strip()]
            k = f"traceback: {lines[-1]}" if lines else "traceback: empty"
        else:
            lines = [l for l in txt.splitlines() if l.strip()]
            k = f"other: {lines[-1]}" if lines else "other: empty"
    errors.setdefault(k, []).append(p)
# produce counts and sample files
out = {"files_examined": files_examined, "counts": {k: len(v) for k, v in errors.items()}, "samples": {k: v[:5] for k, v in errors.items()}}
print(json.dumps(out, indent=2))
