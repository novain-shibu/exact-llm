"""Exact-AI perplexity job, run at Qwen2.5-14B (2026-11-13; derived from the 7B job of 2026-11-10, whose notes follow).
Exact-AI 7B job (2026-11-10). Plan: exact-ai/PLAN_7B.md. MODEL set below (smoke = Qwen2.5-0.5B, then Qwen2.5-7B).
Spec v0.1 = BC2 v2 spec + overflow-safe requant: drop = max(0, bitlen(max|acc|) + bitlen(max M) - 62).
Streaming: weights quantised to int8 tensor by tensor from safetensors; float reference streams float32 layers from disk.
PRE-REGISTERED:
 SMOKE (0.5B): window-0 integer hash == ab68ea173cea07fd (BC1) bit for bit, and streamed float ref within 0.5% of HF fp32 (4 windows).
 7B: harness = streamed float32 ref vs HF bf16 ppl within 2% (4 windows) else INCONCLUSIVE.
     PASS if integer ppl within 2% of streamed float ref on 16x256 WikiText-2 tokens AND window-0 integer hash identical
     for default and permuted reduction order. Quality fail kills the route (next: group-128 scales), not the mountain.
"""
MODEL = "Qwen/Qwen2.5-14B"   # 2026-11-13 run: 8 x 256 tokens, HF cross-check skipped (memory)
import os, sys, json, math, time, hashlib
import numpy as np

PRE = os.environ.get("PREFLIGHT") == "1"
OUT = "results.json" if PRE else "/kaggle/working/results.json"
F = 16; ONE = 1 << F
log = {}
def P(*a):
    print(*a, flush=True)
def h(a): return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]

# ---------------- integer primitives ----------------
def shr(a, n):
    """round-half-up arithmetic shift right by n (n may be array/negative -> left shift)."""
    n = np.asarray(n, dtype=np.int64)
    if n.ndim == 0:
        n = int(n)
        if n <= 0: return a << (-n)
        return (a + (1 << (n - 1))) >> n
    pos = np.maximum(n, 0); neg = np.maximum(-n, 0)
    r = np.where(pos > 0, (a + (np.int64(1) << np.maximum(pos - 1, 0))) >> pos, a)
    return r << neg
def bitlen(m):
    m = np.asarray(m, dtype=np.int64); out = np.zeros_like(m)
    v = m.copy()
    for b in (32, 16, 8, 4, 2, 1):
        big = v >= (np.int64(1) << b); out += big * b; v = np.where(big, v >> b, v)
    return out + (v > 0)
def qrow(x, bits=15):
    """per-row int16 quantisation of int64 Q16 x -> (xq int64 values in +-2^15, shift s per row)."""
    m = np.abs(x).max(-1, keepdims=True)
    s = np.maximum(bitlen(m) - bits, 0)
    xq = np.clip(shr(x, s), -(1 << bits) + 1, (1 << bits) - 1)
    return xq, s
CFG = {"order": None, "rev": False, "abits": 15}
LOG2_SQRT_HD = {16: 2, 64: 3, 128: 3.5}
def exact_mm(a, b):
    """a (..,K) int, b (K,N) int -> exact int64 product via float64 BLAS (all partial sums < 2^53)."""
    K = a.shape[-1]
    if CFG["order"] is not None:
        p = CFG["order"](K); a = a[..., p]; b = b[p, :]
    if CFG["rev"]:
        a = a[..., ::-1]; b = b[::-1, :]
    r = np.matmul(a.astype(np.float64), b.astype(np.float64))
    return np.rint(r).astype(np.int64)
class QLin:
    def __init__(self, w, bias=None):  # w (in,out) float
        sc = np.abs(w).max(0); sc = np.where(sc == 0, 1e-12, sc) / 127.0
        self.w = np.clip(np.rint(w / sc), -127, 127).astype(np.int8)   # stored int8; cast per matmul
        self.M = np.rint(sc * (1 << 28)).astype(np.int64)
        self.Mbits = int(bitlen(np.array([self.M.max()]))[0])
        self.b = None if bias is None else np.rint(bias * ONE).astype(np.int64)
    def __call__(self, x):
        xq, s = qrow(x, 15)
        N = self.w.shape[1]
        acc = exact_mm(xq, self.w) if N <= 32768 else np.concatenate([exact_mm(xq, self.w[:, j:j + 32768]) for j in range(0, N, 32768)], -1)
        drop = max(0, int(bitlen(np.array([np.abs(acc).max()]))[0]) + self.Mbits - 62)
        y = shr(shr(acc, drop) * self.M, 28 - s - drop)
        return y if self.b is None else y + self.b
def isqrt(n):
    r = np.floor(np.sqrt(n.astype(np.float64))).astype(np.int64)
    r = np.where(r * r > n, r - 1, r); r = np.where((r + 1) * (r + 1) <= n, r + 1, r); return r
def irms(x, g):
    k = np.maximum(bitlen(np.abs(x).max(-1, keepdims=True)) - 24, 0)
    xs = shr(x, k)
    ms = (xs * xs).sum(-1, keepdims=True) // x.shape[-1] + 1
    r = np.maximum(isqrt(ms), 1)
    y = (x << F) // (r << k)
    return shr(y * g, F)
EXP_STEP = 8   # 1/256
EXP_T = np.array([round(math.exp(-i / 256) * (1 << 24)) for i in range(256 * 40 + 2)], np.int64)
def isoftmax(sc, valid):  # sc int64 Q16; returns Q20 probs
    mx = np.where(valid, sc, np.iinfo(np.int64).min // 4).max(-1, keepdims=True)
    d = np.where(valid, mx - sc, 0)
    i = np.minimum(d >> EXP_STEP, len(EXP_T) - 2); fr = d & ((1 << EXP_STEP) - 1)
    e = EXP_T[i] - shr((EXP_T[i] - EXP_T[i + 1]) * fr, EXP_STEP)
    e = np.where(valid & (d < (40 << F)), e, 0)
    return (e << 20) // e.sum(-1, keepdims=True)
SILU_STEP = 8   # own constant (BC1c bug: shared EXP_STEP)
SILU_T = np.array([round((v := (i - 16 * 256) / 256) / (1 + math.exp(-v)) * ONE) for i in range(32 * 256 + 2)], np.int64)
def isilu(x):
    t = np.clip(x, -16 * ONE, 16 * ONE - 1) + 16 * ONE
    i, fr = t >> SILU_STEP, t & ((1 << SILU_STEP) - 1)
    y = SILU_T[i] + shr((SILU_T[i + 1] - SILU_T[i]) * fr, SILU_STEP)
    return np.where(x >= 16 * ONE, x, np.where(x < -16 * ONE, 0, y))


# ---------------- model (streaming, v0.1) ----------------
import gc, glob
def rope_tables(T, hd, theta):
    inv = 1.0 / (theta ** (np.arange(0, hd, 2, dtype=np.float64) / hd))
    ang = np.outer(np.arange(T, dtype=np.float64), inv); ang = np.concatenate([ang, ang], -1)
    return np.cos(ang), np.sin(ang)
def rot_half(x):
    a, b = np.split(x, 2, -1); return np.concatenate([-b, a], -1)
def qlin_chunked(w_out_in, bias=None, chunk=16384):
    """QLin from an (out,in) array, quantising row blocks (per-output-channel scales, so identical to one-shot)."""
    parts = [QLin(np.asarray(w_out_in[j:j + chunk], dtype=np.float64).T) for j in range(0, w_out_in.shape[0], chunk)]
    q = QLin.__new__(QLin); q.w = np.concatenate([p.w for p in parts], 1); q.M = np.concatenate([p.M for p in parts])
    q.Mbits = int(bitlen(np.array([q.M.max()]))[0])
    q.b = None if bias is None else np.rint(np.asarray(bias, dtype=np.float64) * ONE).astype(np.int64)
    return q
PFX = "model.layers.{}."
class IntModel:
    def __init__(s, get, c):
        s.c = c; s.emb = get("model.embed_tokens.weight")
        g = lambda k: np.rint(np.asarray(get(k), dtype=np.float64) * ONE).astype(np.int64)
        s.layers = []
        for i in range(c["NL"]):
            p = PFX.format(i)
            s.layers.append(dict(g1=g(p + "input_layernorm.weight"), g2=g(p + "post_attention_layernorm.weight"),
                q=qlin_chunked(get(p + "self_attn.q_proj.weight"), get(p + "self_attn.q_proj.bias")),
                k=qlin_chunked(get(p + "self_attn.k_proj.weight"), get(p + "self_attn.k_proj.bias")),
                v=qlin_chunked(get(p + "self_attn.v_proj.weight"), get(p + "self_attn.v_proj.bias")),
                o=qlin_chunked(get(p + "self_attn.o_proj.weight")), gate=qlin_chunked(get(p + "mlp.gate_proj.weight")),
                up=qlin_chunked(get(p + "mlp.up_proj.weight")), down=qlin_chunked(get(p + "mlp.down_proj.weight"))))
            if i % 4 == 0: P(f"  quantised layer {i}")
        s.gf = g("model.norm.weight")
        s.head = qlin_chunked(s.emb if c["tie"] else get("lm_head.weight"))
        cos, sin = rope_tables(c["T"], c["hd"], c["theta"])
        s.cos = np.rint(cos * ONE).astype(np.int64); s.sin = np.rint(sin * ONE).astype(np.int64)
    def rope(s, x):
        T = x.shape[1]; return shr(x * s.cos[:T] + rot_half(x) * s.sin[:T], F)
    def __call__(s, ids):
        c = s.c; T = len(ids); H, KV, hd = c["H"], c["KV"], c["hd"]
        x = np.rint(np.asarray(s.emb[ids], dtype=np.float64) * ONE).astype(np.int64)
        valid = np.tril(np.ones((T, T), bool)); INV = int(round((1 << 16) / math.sqrt(hd)))
        for L in s.layers:
            a = irms(x, L["g1"])
            q = s.rope(L["q"](a).reshape(T, H, hd).transpose(1, 0, 2))
            k = s.rope(L["k"](a).reshape(T, KV, hd).transpose(1, 0, 2))
            v = L["v"](a).reshape(T, KV, hd).transpose(1, 0, 2)
            out = np.empty((H, T, hd), np.int64)
            for hh in range(H):
                kh = hh // (H // KV)
                qq, sq = qrow(q[hh]); kk, sk = qrow(k[kh])
                sc = shr(exact_mm(qq, kk.T) * INV, (F - sq - sk.T) + 16)
                p = isoftmax(sc, valid)
                sv = max(int(bitlen(np.array([np.abs(v[kh]).max()]))[0]) - 15, 0)
                out[hh] = shr(exact_mm(p, shr(v[kh], sv)), 20 - sv)
            x = x + L["o"](out.transpose(1, 0, 2).reshape(T, H * hd))
            m = irms(x, L["g2"])
            x = x + L["down"](shr(isilu(L["gate"](m)) * L["up"](m), F))
        return s.head(irms(x, s.gf))
def float_ref(get, c, ids, emb):
    T = len(ids); H, KV, hd = c["H"], c["KV"], c["hd"]; dt = np.float32
    def rms(x, gw): return (x / np.sqrt((x * x).mean(-1, keepdims=True) + c["eps"]) * gw).astype(dt)
    cos, sin = rope_tables(T, hd, c["theta"]); cos = cos.astype(dt); sin = sin.astype(dt)
    x = np.asarray(emb[ids], dtype=dt); mask = np.triu(np.full((T, T), -np.inf, dt), 1)
    for i in range(c["NL"]):
        p = PFX.format(i)
        G = lambda k, p=p: np.asarray(get(p + k), dtype=dt)
        a = rms(x, G("input_layernorm.weight"))
        q = (a @ G("self_attn.q_proj.weight").T + G("self_attn.q_proj.bias")).reshape(T, H, hd).transpose(1, 0, 2)
        k = (a @ G("self_attn.k_proj.weight").T + G("self_attn.k_proj.bias")).reshape(T, KV, hd).transpose(1, 0, 2)
        v = (a @ G("self_attn.v_proj.weight").T + G("self_attn.v_proj.bias")).reshape(T, KV, hd).transpose(1, 0, 2)
        q = q * cos + rot_half(q) * sin; k = k * cos + rot_half(k) * sin
        out = np.empty((H, T, hd), dt)
        for hh in range(H):
            kh = hh // (H // KV); sc = q[hh] @ k[kh].T / dt(math.sqrt(hd)) + mask
            sc = np.exp(sc - sc.max(-1, keepdims=True)); sc /= sc.sum(-1, keepdims=True); out[hh] = sc @ v[kh]
        x = x + out.transpose(1, 0, 2).reshape(T, H * hd) @ G("self_attn.o_proj.weight").T
        m = rms(x, G("post_attention_layernorm.weight")); gt = m @ G("mlp.gate_proj.weight").T
        x = x + ((gt / (1 + np.exp(-gt))) * (m @ G("mlp.up_proj.weight").T)) @ G("mlp.down_proj.weight").T
    hw = emb if c["tie"] else get("lm_head.weight")
    xn = rms(x, np.asarray(get("model.norm.weight"), dtype=dt))
    return np.concatenate([xn @ np.asarray(hw[j:j + 32768], dtype=dt).T for j in range(0, hw.shape[0], 32768)], -1)
def nll(logits, ids):
    lg = logits[:-1].astype(np.float64); y = np.asarray(ids[1:])
    lse = np.log(np.exp(lg - lg.max(-1, keepdims=True)).sum(-1)) + lg.max(-1)
    return float((lse - lg[np.arange(len(y)), y]).sum()), len(y)
def ppl(pairs):
    return math.exp(sum(p[0] for p in pairs) / sum(p[1] for p in pairs))

# ---------------- run ----------------
t0 = time.time()
if PRE:
    r = np.random.default_rng(0); c = dict(D=64, H=4, KV=2, hd=16, I=160, V=500, NL=2, T=64, theta=1e4, eps=1e-6, tie=True)
    n = lambda *sh: r.standard_normal(sh) * 0.08
    E = r.standard_normal((c["V"], c["D"])) * 0.5; GF = 1 + 0.1 * r.standard_normal(c["D"]); T_ = {}
    for i in range(c["NL"]):
        d = dict(g1=1 + .1 * r.standard_normal(64), g2=1 + .1 * r.standard_normal(64), wq=n(64, 64) * 2, bq=n(64), wk=n(64, 32) * 2, bk=n(32),
                 wv=n(64, 32) * 2, bv=n(32), wo=n(64, 64), wg=n(64, 160) * 2, wu=n(64, 160) * 2, wd=n(160, 64))
        p = PFX.format(i)
        T_.update({p + "input_layernorm.weight": d["g1"], p + "post_attention_layernorm.weight": d["g2"],
            p + "self_attn.q_proj.weight": d["wq"].T, p + "self_attn.q_proj.bias": d["bq"], p + "self_attn.k_proj.weight": d["wk"].T,
            p + "self_attn.k_proj.bias": d["bk"], p + "self_attn.v_proj.weight": d["wv"].T, p + "self_attn.v_proj.bias": d["bv"],
            p + "self_attn.o_proj.weight": d["wo"].T, p + "mlp.gate_proj.weight": d["wg"].T, p + "mlp.up_proj.weight": d["wu"].T,
            p + "mlp.down_proj.weight": d["wd"].T})
    T_["model.embed_tokens.weight"] = E; T_["model.norm.weight"] = GF
    get = lambda k: T_[k]
    r2 = np.random.default_rng(1); windows = [r2.integers(0, c["V"], c["T"]).tolist() for _ in range(4)]
    hf_ppl = None; NW = 4; small = True
else:
    import torch
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer, AutoModelForCausalLM
    path = snapshot_download(MODEL, allow_patterns=["*.safetensors", "*.json", "*.txt"])
    idx = {}
    for f in glob.glob(os.path.join(path, "*.safetensors")):
        with safe_open(f, "pt") as fh:
            for k in fh.keys(): idx[k] = f
    def get(k):
        with safe_open(idx[k], "pt") as fh: return fh.get_tensor(k).float().numpy()
    cf = json.load(open(os.path.join(path, "config.json")))
    c = dict(D=cf["hidden_size"], H=cf["num_attention_heads"], KV=cf["num_key_value_heads"], hd=cf["hidden_size"] // cf["num_attention_heads"],
             I=cf["intermediate_size"], V=cf["vocab_size"], NL=cf["num_hidden_layers"], T=256,
             theta=float(cf.get("rope_theta") or (cf.get("rope_parameters") or {}).get("rope_theta", 1e6)), eps=cf["rms_norm_eps"],
             tie="lm_head.weight" not in idx)
    log["config"] = c; P("config", c, round(time.time() - t0))
    tok = AutoTokenizer.from_pretrained(path)
    try:
        from datasets import load_dataset
        text = (chr(10) * 2).join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    except Exception:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        text = (chr(10) * 2).join(pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"))["text"])
    ids = tok(text)["input_ids"]; windows = [ids[i * 256:(i + 1) * 256] for i in range(8)]; NW = 8
    small = False; hf_ppl = None; log["hf_check"] = "skipped at 14B: bf16 HF model (29.5 GB) does not fit next to the int model in Kaggle CPU RAM; float_ref harness validated at 0.5B-7B"
im = IntModel(get, c); gc.collect(); P("int model built", round(time.time() - t0))
ip = []
for wi, w in enumerate(windows[:NW]):
    li = im(w); ip.append(nll(li / ONE, w))
    if wi == 0:
        h0 = h(li); CFG.update(order=lambda K: np.random.default_rng(K).permutation(K)); h0p = h(im(w)); CFG.update(order=None)
    P(f"int window {wi}", round(time.time() - t0))
fp = []
for wi, w in enumerate(windows[:NW]):
    fp.append(nll(float_ref(get, c, w, im.emb), w)); P(f"ref window {wi}", round(time.time() - t0))
log.update(int_ppl=ppl(ip), ref_ppl=ppl(fp), ref_ppl_4w=ppl(fp[:4]), int_hash_w0=h0, int_hash_w0_permuted=h0p)
log["gap_int_vs_ref"] = log["int_ppl"] / log["ref_ppl"] - 1
log["hash_stable"] = h0 == h0p
if PRE:
    log["smoke_hash_match"] = h0 == "084576e12554478c"
    log["verdict"] = "PRE-OK" if log["smoke_hash_match"] and log["hash_stable"] else "PRE-FAIL"
elif small:
    log["harness_gap"] = log["ref_ppl_4w"] / hf_ppl - 1
    log["smoke_hash_match"] = h0 == "ab68ea173cea07fd"
    log["verdict"] = "SMOKE-PASS" if log["smoke_hash_match"] and abs(log["harness_gap"]) <= 0.005 and log["hash_stable"] else "SMOKE-FAIL"
else:
    # PRE-REGISTERED (2026-11-13): PASS if |exact ppl / streamed float32 reference ppl - 1| <= 2% (same gate as 7B) and the hash is order-stable.
    log["verdict"] = "PASS" if abs(log["gap_int_vs_ref"]) <= 0.02 and log["hash_stable"] else "FAIL"
log["seconds"] = round(time.time() - t0); P(json.dumps(log, indent=1)); json.dump(log, open(OUT, "w"), indent=1)
