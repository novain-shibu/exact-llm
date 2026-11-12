# exact-ai float-leak measurement (2026-11-12), Tesla T4, HF transformers fp16.
# QUESTION: how much freedom does honest float nondeterminism give a covert server (the residual channel in arXiv 2511.02620, Thm 4.1)?
# METHOD: 16 WikiText prompts x 64 teacher-forced continuation tokens (the same fixed token sequence under every config).
# Configs an honest server may use: C1 batch-1 full forward; C2 batch-8 (prompt inside a batch of 8 different sequences, right padding);
# C3 KV-cache incremental (prefill prompt, then 1 token per step); C4 batch-1 full forward with SDPA math backend.
# For each position and fixed seed u (inverse-CDF sampling, temperature 0.8), record: token chosen under each config; max over config pairs of
# |CDF_a(t) - CDF_b(t)| at the sampled boundary (the tolerance delta an honest-traffic verifier must accept at that position).
# REPORT (no pass/fail; disclosure measurement): fraction of positions where configs pick different tokens (fork rate); delta quantiles;
# capacity model: mean log2(#tokens within the delta needed for 0 false alarms on 99.99% / 100% of honest positions).
# Exact integer inference: 0 forks and delta = 0 by construction (measured in our other jobs: identical hashes).
import os, json, math, time, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
dev = torch.device("cuda"); R = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}; P = lambda *a: print(*a, flush=True)
try:
    from datasets import load_dataset
    TEXT = (chr(10) * 2).join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
except Exception:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    TEXT = (chr(10) * 2).join(pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"))["text"])
NP, PL, CL, TEMP = 16, 64, 64, 0.8
rng = np.random.default_rng(0)
for name in ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"):
    tag = name.split("-")[-1]; t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name); ids = tok(TEXT)["input_ids"]
    seqs = [ids[2000 + i * 300: 2000 + i * 300 + PL + CL] for i in range(NP)]
    others = [ids[60000 + i * 300: 60000 + i * 300 + PL + CL] for i in range(7)]
    m = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to(dev).eval()
    def logp(lg): return torch.log_softmax(lg.float() / TEMP, -1)
    out = {c: [] for c in ("C1", "C2", "C3", "C4")}
    with torch.no_grad():
        for s in seqs:
            x = torch.tensor([s], device=dev)
            out["C1"].append(logp(m(x).logits[0, PL - 1:PL + CL - 1]).cpu())
            xb = torch.tensor([s] + others, device=dev)
            out["C2"].append(logp(m(xb).logits[0, PL - 1:PL + CL - 1]).cpu())
            o = m(torch.tensor([s[:PL]], device=dev), use_cache=True); pkv = o.past_key_values; rows = [o.logits[0, -1]]
            for t in range(PL, PL + CL - 1):
                o = m(torch.tensor([[s[t]]], device=dev), past_key_values=pkv, use_cache=True); pkv = o.past_key_values; rows.append(o.logits[0, -1])
            out["C3"].append(logp(torch.stack(rows)).cpu())
            try:
                with torch.nn.attention.sdpa_kernel(torch.nn.attention.SDPBackend.MATH):
                    out["C4"].append(logp(m(x).logits[0, PL - 1:PL + CL - 1]).cpu())
            except Exception as ex:
                out["C4"].append(out["C1"][-1]); R.setdefault("c4_note", str(ex)[:160])
    forks = 0; n = 0; deltas = []; maxabs_logit = 0.0; pairs_diff = {}
    U = rng.random((NP, CL))
    for i in range(NP):
        cdfs = {c: torch.cumsum(out[c][i].exp().double(), -1).numpy() for c in out}
        for t in range(CL):
            u = U[i, t]; toks = {c: int(np.searchsorted(cdfs[c][t], u * cdfs[c][t][-1], side="right")) for c in cdfs}
            n += 1; forks += len(set(toks.values())) > 1
            base = toks["C1"]; d = 0.0
            for c in cdfs:
                for c2 in cdfs:
                    if c < c2: d = max(d, float(np.abs(cdfs[c][t] - cdfs[c2][t]).max()))
            deltas.append(d)
        for c in out:
            if c != "C1": pairs_diff[c] = max(pairs_diff.get(c, 0.0), float((out[c][i] - out["C1"][i]).abs().max()))
    deltas = np.array(deltas)
    def capacity(delta):
        bits = []
        for i in range(NP):
            p1 = out["C1"][i].exp().double(); cdf = torch.cumsum(p1, -1).numpy()
            for t in range(CL):
                u = U[i, t]; lo = max(u - delta, 0.0); hi = min(u + delta, 1.0)
                a = int(np.searchsorted(cdf[t], lo * cdf[t][-1], side="right")); b = int(np.searchsorted(cdf[t], hi * cdf[t][-1], side="right"))
                bits.append(math.log2(max(b - a + 1, 1)))
        return float(np.mean(bits)), float(np.mean(np.array(bits) > 0))
    d9999 = float(np.quantile(deltas, 0.9999)); dmax = float(deltas.max())
    R[tag] = dict(positions=n, fork_positions=forks, fork_rate=round(forks / n, 5),
                  delta_quantiles={q: float(np.quantile(deltas, q)) for q in (0.5, 0.9, 0.99, 0.999, 1.0)},
                  max_logprob_diff_vs_C1=pairs_diff,
                  capacity_bits_per_token_at_delta_max=capacity(dmax), capacity_bits_per_token_at_delta_q9999=capacity(d9999),
                  seconds=round(time.time() - t0))
    P(tag, json.dumps(R[tag])); json.dump(R, open("/kaggle/working/results.json", "w"), indent=1)
    del m; torch.cuda.empty_cache()
P(json.dumps(R, indent=1)); json.dump(R, open("/kaggle/working/results.json", "w"), indent=1)
