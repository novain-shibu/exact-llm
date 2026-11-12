# exact-ai vLLM batching nondeterminism (2026-11-12), Tesla T4. Measurement, no pass/fail.
# Same prompt + same per-request seed + same sampling params; only the co-batched traffic changes:
#  A: each request alone; B: all 64 requests in one generate call; C: the 64 requests mixed with 64 other distractor requests.
# Report: share of requests whose 64-token output differs between A/B, A/C; first divergence position; greedy (T=0) too.
import subprocess, sys, json, os
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], check=False)
code = r'''
import json, time
from vllm import LLM, SamplingParams
import vllm, torch
R = {"vllm": vllm.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
try:
    from datasets import load_dataset
    rows = [t for t in load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"] if len(t) > 300]
except Exception:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    rows = [t for t in pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"))["text"] if len(t) > 300]
prompts = [r[:300] for r in rows[:64]]; distract = [r[:300] for r in rows[200:264]]
for name in ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"):
    tag = name.split("-")[-1]; out = {}
    llm = LLM(model=name, dtype="float16", max_model_len=512, gpu_memory_utilization=0.5, enforce_eager=False, seed=0)
    for temp in (0.8, 0.0):
        sp = [SamplingParams(temperature=temp, max_tokens=64, min_tokens=64, ignore_eos=True, seed=1000 + i) for i in range(64)]
        A = [llm.generate([p], [s], use_tqdm=False)[0].outputs[0].token_ids for p, s in zip(prompts, sp)]
        B = [o.outputs[0].token_ids for o in llm.generate(prompts, sp, use_tqdm=False)]
        spd = [SamplingParams(temperature=temp, max_tokens=64, min_tokens=64, ignore_eos=True, seed=5000 + i) for i in range(64)]
        mix = []; mixs = []
        for i in range(64): mix += [prompts[i], distract[i]]; mixs += [sp[i], spd[i]]
        Cr = llm.generate(mix, mixs, use_tqdm=False); C = [Cr[2 * i].outputs[0].token_ids for i in range(64)]
        def cmp(X, Y):
            diff = [i for i in range(64) if list(X[i]) != list(Y[i])]
            first = [next(t for t in range(64) if X[i][t] != Y[i][t]) for i in diff]
            return dict(requests_differing=len(diff), of=64, first_divergence_positions=first)
        out[f"T{temp}"] = dict(A_vs_B=cmp(A, B), A_vs_C=cmp(A, C), B_vs_C=cmp(B, C))
        print(tag, temp, json.dumps({k: v["requests_differing"] for k, v in out[f"T{temp}"].items()}), flush=True)
    R[tag] = out; del llm; import gc; gc.collect(); torch.cuda.empty_cache()
    json.dump(R, open("/kaggle/working/results.json", "w"), indent=1)
json.dump(R, open("/kaggle/working/results.json", "w"), indent=1)
'''
open("/kaggle/working/vleak.py", "w").write(code)
r = subprocess.run([sys.executable, "/kaggle/working/vleak.py"], capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0", "VLLM_ENABLE_V1_MULTIPROCESSING": "0"})
print(r.stdout[-4000:]); errs = [l for l in r.stderr.splitlines() if "Error" in l or "error" in l]; print(chr(10).join(errs[:40])); print(r.stderr[-3000:])
if not os.path.exists("/kaggle/working/results.json"):
    json.dump({"error_lines": errs[:40], "error_tail": r.stderr[-1500:]}, open("/kaggle/working/results.json", "w"), indent=1)
