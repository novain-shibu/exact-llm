# Usage: commit-and-audit zero-leak verification v2 (K/V commitment). Default Qwen/Qwen2.5-0.5B; ZL_MODEL overrides; ZL_TINY=1 = local tiny-model preflight.
"""exact_llm - bit-exact integer inference for Qwen2-family LLMs (spec v0.4, see SPEC.md).

Every reduction is an exact integer sum; every nonlinearity is a shipped integer table or integer arithmetic.
Same weights + same tokens -> the same int64 logits on any conforming hardware (tested: x86 CPUs, NVIDIA T4 float64
and int8 tensor cores). Reference implementation: numpy (slow, for verification), optional torch/CUDA matmul backends.

Usage:
    import exact_llm as ex
    get, c, tok = ex.open_hf("Qwen/Qwen2.5-0.5B")      # streams safetensors, needs huggingface_hub + safetensors + torch
    model = ex.IntModel(get, c)
    logits = model(tok("Hello world")["input_ids"])     # int64, Q16 fixed point
    ex.h(logits)                                        # conformance hash
"""
import os, sys, json, math, time, hashlib
import numpy as np

PRE = False
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
_GC = {}
def _split_mm(a, b, int_mm):
    a = a.astype(np.int64); hi = a >> 8; lo = (a & 255) - 128
    return (int_mm(hi, b) << 8) + int_mm(lo, b) + 128 * b.astype(np.int64).sum(0)
def exact_mm(a, b):
    """a (..,K) int, b (K,N) int -> exact int64 product (all partial sums < 2^53 on every backend)."""
    be = CFG.get("backend", "cpu")
    w8 = b.dtype == np.int8 and a.ndim == 2 and a.shape[0] > 16 and b.shape[0] % 8 == 0 and b.shape[1] % 8 == 0 and np.abs(a).max() < 32768
    if be == "emu_int8" and w8:
        return _split_mm(a, b, lambda x, w: (x.astype(np.int32) @ w.astype(np.int32)).astype(np.int64))
    if be == "cuda_int8" and w8:
        import torch
        if id(b) in CFG.get("wids", ()):
            if id(b) not in _GC: _GC[id(b)] = (b, torch.from_numpy(np.ascontiguousarray(b)).cuda())
            wb = _GC[id(b)][1]
        else:
            wb = torch.from_numpy(np.ascontiguousarray(b)).cuda()
        def imm(x, w):
            return torch._int_mm(torch.from_numpy(np.ascontiguousarray(x).astype(np.int8)).cuda(), wb).to(torch.int64).cpu().numpy()
        return _split_mm(a, b, imm)
    if be in ("cuda64", "cuda_int8"):
        import torch
        bt = torch.from_numpy(np.ascontiguousarray(b)).cuda().to(torch.float64)
        at = torch.from_numpy(np.ascontiguousarray(a).astype(np.float64)).cuda()
        r = torch.round(at @ bt).to(torch.int64).cpu().numpy(); del bt, at
        return r
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
        drop = np.maximum(bitlen(np.abs(acc).max(-1, keepdims=True)) + self.Mbits - 62, 0)   # per row (spec v0.2)
        if drop.any(): CFG["drops"] = CFG.get("drops", 0) + 1
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
                out[hh] = shr(np.matmul(p.astype(np.int64), v[kh].astype(np.int64)), 20)   # spec v0.4: exact int64, no length cap
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



def open_hf(name):
    """Stream a Qwen2-family checkpoint: returns (get(tensor_name)->float32 ndarray, config dict, tokenizer)."""
    import glob as _glob
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from transformers import AutoTokenizer
    path = snapshot_download(name, allow_patterns=["*.safetensors", "*.json", "*.txt"])
    idx = {}
    for f in _glob.glob(os.path.join(path, "*.safetensors")):
        with safe_open(f, "pt") as fh:
            for k in fh.keys(): idx[k] = f
    def get(k):
        with safe_open(idx[k], "pt") as fh: return fh.get_tensor(k).float().numpy()
    cf = json.load(open(os.path.join(path, "config.json")))
    c = dict(D=cf["hidden_size"], H=cf["num_attention_heads"], KV=cf["num_key_value_heads"], hd=cf["hidden_size"] // cf["num_attention_heads"],
             I=cf["intermediate_size"], V=cf["vocab_size"], NL=cf["num_hidden_layers"], T=4096,
             theta=float(cf.get("rope_theta") or (cf.get("rope_parameters") or {}).get("rope_theta", 1e6)), eps=cf["rms_norm_eps"],
             tie="lm_head.weight" not in idx)
    return get, c, AutoTokenizer.from_pretrained(path)

# ======================= exact sampling + proof recorder (spec v0.2 draft) =======================
# PRE-REGISTERED (2026-11-11, before first run):
#  S1 receipts (sampled token ids + per-step logits hash chain) identical under default / permuted / reversed reduction order
#  S2 a recorded generation with ONE token changed is rejected by recomputation (and the first bad position is reported)
#  S3 a cheating model (7-bit down_proj in the middle layer) produces a receipt that verification rejects
#  S4 sampler fidelity: integer sampler on step-0 logits, 20,000 seeds, total-variation distance to the float softmax
#     (same temperature) <= 0.02 (sampling noise alone is ~0.01)
#  S5 it really samples: at least 1 of the generated tokens differs from greedy argmax
#  PASS = S1..S5. Kill: if S1 fails, sampling is not exact -> recorder limited to greedy.
MASK64 = (1 << 64) - 1
def splitmix64(seed, pos):
    z = (seed + (pos + 1) * 0x9E3779B97F4A7C15) & MASK64
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK64
    return z ^ (z >> 31)
def u20(seed, pos): return splitmix64(seed, pos) >> 44            # uniform integer in [0, 2^20)
def temp_num(t): return int(round((1 << 16) / t))                    # temperature as integer multiplier
def isoftmax_row(lg):                                                # lg int64 Q16 (1-D) -> p int64 Q20
    d = lg.max() - lg
    i = np.minimum(d >> 8, len(EXP_T) - 2); fr = d & 255
    e = EXP_T[i] - shr((EXP_T[i] - EXP_T[i + 1]) * fr, 8)
    e = np.where(d < (40 << 16), e, 0)
    return (e << 20) // e.sum()
def sample_int(lg_row, seed, pos, tnum):
    p = isoftmax_row(shr(lg_row * tnum, 16)); cum = np.cumsum(p)
    target = (u20(seed, pos) * int(cum[-1])) >> 20
    return int(np.searchsorted(cum, target, side="right")), p
def step_hash(prev, lg_row):
    return hashlib.sha256(prev.encode() + np.ascontiguousarray(lg_row).tobytes()).hexdigest()[:16]
def generate(model, prompt_ids, n, seed, temp):
    ids = list(prompt_ids); chain = "start"; toks = []; greedy = []; tnum = temp_num(temp)
    for t in range(n):
        lg = model(ids)[-1]
        tok, _ = sample_int(lg, seed, t, tnum); chain = step_hash(chain, lg)
        toks.append(tok); greedy.append(int(lg.argmax())); ids.append(tok)
    return dict(prompt=list(prompt_ids), seed=seed, temp=temp, tokens=toks, chain=chain), greedy
def verify(model, receipt):
    """Recompute the receipt; return (ok, first_bad_position or None)."""
    ids = list(receipt["prompt"]); chain = "start"; tnum = temp_num(receipt["temp"])
    for t, rec_tok in enumerate(receipt["tokens"]):
        lg = model(ids)[-1]; tok, _ = sample_int(lg, receipt["seed"], t, tnum); chain = step_hash(chain, lg)
        if tok != rec_tok: return False, t
        ids.append(rec_tok)
    return chain == receipt["chain"], (None if chain == receipt["chain"] else "chain")
def sampler_tv(lg_row, temp, n=20000):
    tnum = temp_num(temp); _, p = sample_int(lg_row, 1, 0, tnum)
    cum = np.cumsum(p); tot = int(cum[-1])
    draws = np.array([np.searchsorted(cum, (u20(s, 0) * tot) >> 20, side="right") for s in range(n)])
    emp = np.bincount(draws, minlength=len(p)) / n
    lf = lg_row.astype(np.float64) / ONE / temp; pf = np.exp(lf - lf.max()); pf /= pf.sum()
    return float(0.5 * np.abs(emp - pf).sum())

def run_sampling_spark(model, prompt_ids, get, cheat_fn, n_tokens, seed=12345, temp=0.8):
    t0 = time.time(); out = {}
    receipts = {}
    for mode in ("default", "perm", "rev"):
        CFG.update(order=(lambda K: np.random.default_rng(K).permutation(K)) if mode == "perm" else None, rev=(mode == "rev"))
        receipts[mode], greedy = generate(model, prompt_ids, n_tokens, seed, temp)
        P("generated", mode, receipts[mode]["chain"], round(time.time() - t0))
    CFG.update(order=None, rev=False)
    r = receipts["default"]
    out["S1_identical_receipts"] = len({(tuple(x["tokens"]), x["chain"]) for x in receipts.values()}) == 1
    ok, bad = verify(model, r); out["honest_verifies"] = ok
    tam = dict(r); tam["tokens"] = list(r["tokens"]); k = len(tam["tokens"]) // 2; tam["tokens"][k] = (tam["tokens"][k] + 1) % 1000
    ok_t, bad_t = verify(model, tam); out["S2_tamper_rejected"] = (not ok_t); out["S2_first_bad_position"] = bad_t; out["S2_tampered_position"] = k
    restore = cheat_fn(model)
    cr, _ = generate(model, prompt_ids, n_tokens, seed, temp); restore()
    okc, badc = verify(model, cr); out["S3_cheat_rejected"] = (not okc); out["S3_cheat_tokens_differ"] = cr["tokens"] != r["tokens"]
    lg0 = model(list(prompt_ids))[-1]
    out["S4_tv_distance"] = sampler_tv(lg0, temp); out["S4_pass"] = out["S4_tv_distance"] <= 0.02
    out["S5_tokens_not_greedy"] = int(sum(a != b for a, b in zip(r["tokens"], greedy)))
    out["S5_pass"] = out["S5_tokens_not_greedy"] >= 1
    out["receipt"] = r
    out["verdict"] = "PASS" if (out["S1_identical_receipts"] and out["honest_verifies"] and out["S2_tamper_rejected"]
                                and out["S3_cheat_rejected"] and out["S4_pass"] and out["S5_pass"]) else "FAIL"
    out["seconds"] = round(time.time() - t0)
    return out

# ======================= spot-check protocol (GPU prover / CPU verifier) =======================
# Provider: runs the full exact model, commits H_l = hash(x_l) for every layer input x_0..x_L, plus hash(logits).
# Verifier: picks random (layer l, position t); provider reveals x_l and x_{l+1}; verifier checks both against the
# commitments and recomputes ONLY block l at position t (k/v for all rows, everything else for one row).
# PRE-REGISTERED (2026-11-11, before first run):
#  SP1 exactness of the shortcut: block_row(l, x_l, t) == block_full(l, x_l)[t] for every layer at 3 positions (0.5B), and
#      forward_record logits == IntModel logits (hash).
#  SP2 honest provider passes 30 random spot checks (0.5B) and 10 (7B).
#  SP3 cheat A (7-bit down_proj in ONE middle layer, 0.5B): fraction of random checks that catch it ~= 1/L (report;
#      pass if the check at the cheated layer always fails and checks elsewhere pass); cheat B (int4 MLP in ALL layers): every check fails.
#  SP4 report verifier seconds per check on CPU (0.5B, 7B) vs full forward seconds; no gate (speed is reported, not claimed).
def _attn_mlp_full(m, L, x):
    c = m.c; T = x.shape[0]; H, KV, hd = c["H"], c["KV"], c["hd"]
    valid = np.tril(np.ones((T, T), bool)); INV = int(round((1 << 16) / math.sqrt(hd)))
    a = irms(x, L["g1"])
    q = m.rope(L["q"](a).reshape(T, H, hd).transpose(1, 0, 2))
    k = m.rope(L["k"](a).reshape(T, KV, hd).transpose(1, 0, 2))
    v = L["v"](a).reshape(T, KV, hd).transpose(1, 0, 2)
    out = np.empty((H, T, hd), np.int64)
    for hh in range(H):
        kh = hh // (H // KV)
        qq, sq = qrow(q[hh]); kk, sk = qrow(k[kh])
        sc = shr(exact_mm(qq, kk.T) * INV, (F - sq - sk.T) + 16)
        p = isoftmax(sc, valid)
        out[hh] = shr(np.matmul(p.astype(np.int64), v[kh].astype(np.int64)), 20)   # spec v0.4: exact int64, no length cap
    x = x + L["o"](out.transpose(1, 0, 2).reshape(T, H * hd))
    mm_ = irms(x, L["g2"])
    return x + L["down"](shr(isilu(L["gate"](mm_)) * L["up"](mm_), F))
def block_row(m, L, x, t):
    """Recompute one transformer block for ONE position t (needs the full layer input x for keys/values)."""
    c = m.c; T = x.shape[0]; H, KV, hd = c["H"], c["KV"], c["hd"]
    INV = int(round((1 << 16) / math.sqrt(hd)))
    a = irms(x, L["g1"])
    qt = L["q"](a[t:t + 1]).reshape(1, H, hd).transpose(1, 0, 2)
    qt = shr(qt * m.cos[t:t + 1] + rot_half(qt) * m.sin[t:t + 1], F)
    k = m.rope(L["k"](a).reshape(T, KV, hd).transpose(1, 0, 2))
    v = L["v"](a).reshape(T, KV, hd).transpose(1, 0, 2)
    valid = (np.arange(T) <= t)[None, :]
    out = np.empty((H, 1, hd), np.int64)
    for hh in range(H):
        kh = hh // (H // KV)
        qq, sq = qrow(qt[hh]); kk, sk = qrow(k[kh])
        sc = shr(exact_mm(qq, kk.T) * INV, (F - sq - sk.T) + 16)
        p = isoftmax(sc, valid)
        out[hh] = shr(np.matmul(p.astype(np.int64), v[kh].astype(np.int64)), 20)   # spec v0.4: exact int64, no length cap
    xt = x[t:t + 1] + L["o"](out.transpose(1, 0, 2).reshape(1, H * hd))
    mt = irms(xt, L["g2"])
    return (xt + L["down"](shr(isilu(L["gate"](mt)) * L["up"](mt), F)))[0]
def forward_record(m, ids):
    x = np.rint(np.asarray(m.emb[ids], dtype=np.float64) * ONE).astype(np.int64); xs = [x]
    for L in m.layers:
        x = _attn_mlp_full(m, L, x); xs.append(x)
    return xs, m.head(irms(x, m.gf))
def commit(xs, logits): return dict(layers=[h(x) for x in xs], logits=h(logits))
def spot_check(m_true, commitment, reveal, l, t):
    xl, xn = reveal(l), reveal(l + 1)
    if h(xl) != commitment["layers"][l] or h(xn) != commitment["layers"][l + 1]: return False, "reveal mismatch"
    ok = np.array_equal(block_row(m_true, m_true.layers[l], xl, t), xn[t])
    return ok, ("ok" if ok else f"block {l} pos {t} differs")

def run_spotcheck(m, get, ids, cheats, n_honest, n_cheat, full_check=True):
    out = {}; t0 = time.time()
    xs, lg = forward_record(m, ids); out["sec_full_forward"] = round(time.time() - t0, 2)
    out["SP1_record_matches_model"] = h(lg) == h(m(ids))
    T = len(ids); NL = len(m.layers); r = np.random.default_rng(7)
    if full_check:
        ok = True
        for l in range(NL):
            for t in (0, T // 2, T - 1):
                ok &= np.array_equal(block_row(m, m.layers[l], xs[l], t), xs[l + 1][t])
        out["SP1_row_equals_full_all_layers"] = bool(ok)
    com = commit(xs, lg); secs = []; passes = 0
    for _ in range(n_honest):
        l, t = int(r.integers(NL)), int(r.integers(T)); t1 = time.time()
        ok, _ = spot_check(m, com, lambda i: xs[i], l, t); secs.append(time.time() - t1); passes += ok
    out["SP2_honest_pass"] = f"{passes}/{n_honest}"; out["verifier_sec_per_check_median"] = round(float(np.median(secs)), 3)
    for name, (apply_fn, cheated_layers) in cheats.items():
        restore = apply_fn(m); cxs, clg = forward_record(m, ids); restore()
        ccom = commit(cxs, clg); caught = 0; at_cheat = []; elsewhere = []
        for _ in range(n_cheat):
            l, t = int(r.integers(NL)), int(r.integers(T))
            ok, _ = spot_check(m, ccom, lambda i: cxs[i], l, t); caught += (not ok)
            (at_cheat if l in cheated_layers else elsewhere).append(not ok)
        out[name] = dict(caught=f"{caught}/{n_cheat}", caught_when_cheated_layer_picked=f"{sum(at_cheat)}/{len(at_cheat)}",
                         false_alarms_elsewhere=f"{sum(elsewhere)}/{len(elsewhere)}", expected_rate=round(len(cheated_layers) / NL, 3))
        P(name, out[name], round(time.time() - t0))
    return out


os.environ.setdefault("ZL_MODEL", "Qwen/Qwen2.5-0.5B")
if os.environ.get("ZL_TINY") == "1":
    TEXT = "tiny preflight"
else:
    try:
        from datasets import load_dataset
        TEXT = (chr(10) * 2).join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
    except Exception:
        import pandas as pd
        from huggingface_hub import hf_hub_download
        TEXT = (chr(10) * 2).join(pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"))["text"])

# ======================= ZERO-LEAK VERIFIED INFERENCE (2026-11-12), Qwen2.5-0.5B, Kaggle CPU =======================
# Verifier holds NO weight matrices: per linear layer a secret Freivalds digest d = W r (mod two 31-bit primes), the per-channel
# integer scales M, biases, norm gains, and a digest per embedding row. Prover sends the integer accumulators of every linear
# (the trace) + embedding rows. Verifier re-runs the exact integer program with every big matmul REPLACED by a digest check;
# norms, requantisation, RoPE, attention, softmax, SiLU and the sampler are recomputed exactly. Credit: Freivalds 1977, Slalom 2019,
# CommitLLM (same key idea; its attention is not verified on float GPU kernels). New here: the whole model is an integer program, so
# nothing is left approximate, and the sampling seed is fixed by the protocol (seed = H(prompt, verifier nonce)).
# PRE-REGISTERED GATES (before first run):
#  Z1 honest: 4 x 128-token windows verify with 0 failed checks, and verifier logits hash == prover logits hash.
#  Z2 forged accumulators: one entry +-1 (30 trials, random linear/row/col) and one entry +P1 (10 trials) -> all rejected.
#  Z3 cheating model (7-bit down_proj, middle layer) -> rejected.
#  Z4 consistent fake attention (CommitLLM's residual hole): attention probabilities altered at one layer/head, everything downstream
#     recomputed honestly by the prover -> rejected in 10/10 trials.
#  Z5 embedding-row swap -> rejected 10/10.
#  Z6 zero-leak generation: honest 32-token sampled generation with protocol seed verifies; attacker hiding bits by token substitution
#     (consistent continuation, honest trace of the altered text) -> rejected 10/10; attacker using its own seed -> rejected when tokens differ.
#  Z7 (report, no gate) capacity model of a tolerance-based verifier on the same logits: mean log2(#tokens acceptable) per token when the
#     verifier must tolerate a CDF error delta; delta = 0 is the exact verifier. A MODEL of tolerance checking, not a measurement of any published system.
#  Report: key bytes vs weight bytes, trace bytes/token, verifier seconds vs prover forward seconds, row checks.
#  Z8 (added 2026-11-12 before the 1.5B/7B runs) CommitLLM red-team A1 replicated: consistent fake post-attention output a_L (the o_proj
#     input) at the first and last layer, modes single+1, single+16, shift+16, shift+64, zero (on the int16-quantised a of the last
#     position); everything downstream recomputed honestly by the prover. CommitLLM's audit-only verifier passes 10/10 of these
#     (docs/security/redteam_audit_only.md, 2026-04-25, toy 2-layer model). Gate: our verifier rejects 10/10.
#  PASS = Z1..Z6 (and Z8 when run). Kill route if any forgery is accepted.
OUT = os.environ.get("ZL_OUT", "/kaggle/working/results.json"); R = {}; t0 = time.time()
TINY = os.environ.get("ZL_TINY") == "1"
def save(): json.dump(R, open(OUT, "w"), indent=1)
P1, P2 = 2147483647, 2147483629
_orig_qcall = QLin.__call__
def requant(M, Mbits, b, acc, s):
    drop = np.maximum(bitlen(np.abs(acc).max(-1, keepdims=True)) + Mbits - 62, 0)
    y = shr(shr(acc, drop) * M, 28 - s - drop)
    return y if b is None else y + b
def prover_forward(m, ids, hook=None, xq_hook=None):
    """exact forward that records every linear's integer accumulator. hook(i, acc) / xq_hook(self, xq) may tamper (attacks)."""
    tr = []
    def call(self, x):
        xq, s = qrow(x, 15); N = self.w.shape[1]
        if xq_hook is not None: xq = xq_hook(self, xq)
        acc = exact_mm(xq, self.w) if N <= 32768 else np.concatenate([exact_mm(xq, self.w[:, j:j + 32768]) for j in range(0, N, 32768)], -1)
        if hook is not None: acc = hook(len(tr), acc)
        tr.append(acc); return requant(self.M, self.Mbits, self.b, acc, s)
    QLin.__call__ = call
    try: lg = m(ids)
    finally: QLin.__call__ = _orig_qcall
    x0 = np.rint(np.asarray(m.emb[ids], dtype=np.float64) * ONE).astype(np.int64)
    return lg, tr, x0
VER = dict(trace=None, i=0, fail=[], checks=0)
class VLin:
    """verifier's stand-in for a linear layer: no weights, only digests d_i = W r_i mod p_i (secret r_i), scales M, bias."""
    def __init__(s, q, rng, name):
        K, N = q.w.shape; s.M, s.Mbits, s.b, s.K, s.N, s.name = q.M, q.Mbits, q.b, K, N, name; s.keys = []
        for p in (P1, P2):
            r = rng.integers(0, p, N, dtype=np.int64); d = np.zeros(K, np.int64)
            for j in range(0, N, 8192): d = (d + q.w[:, j:j + 8192].astype(np.int64) @ r[j:j + 8192]) % p     # |term| < 2^7 * 2^31 * 2^13
            s.keys.append((p, r, d))
    def key_bytes(s): return 2 * 8 * s.K + 8 * s.N + (0 if s.b is None else 8 * s.N)        # r regenerated from a seed, not stored
    def __call__(s, x):
        xq, sh = qrow(x, 15); idx = VER["i"]; acc = VER["trace"][idx]; VER["i"] += 1
        ok = isinstance(acc, np.ndarray) and acc.dtype == np.int64 and acc.shape == (x.shape[0], s.N) and int(np.abs(acc).max()) <= 127 * 32767 * s.K
        if ok:
            for p, r, d in s.keys:
                lhs = (((acc % p) * r) % p).sum(-1) % p; rhs = (((xq % p) * d) % p).sum(-1) % p
                ok = ok and bool((lhs == rhs).all())
        VER["checks"] += x.shape[0]
        if not ok:
            VER["fail"].append((idx, s.name)); raise ValueError("reject " + s.name)
        return requant(s.M, s.Mbits, s.b, acc, sh)
class VEmb:
    """verifier's stand-in for the embedding table: one digest per token row."""
    def __init__(s, emb, rng):
        D = emb.shape[1]; s.keys = []
        for p in (P1, P2):
            r = rng.integers(0, p, D, dtype=np.int64); de = np.empty(emb.shape[0], np.int64)
            for j in range(0, emb.shape[0], 16384):
                X = np.rint(np.asarray(emb[j:j + 16384], dtype=np.float64) * ONE).astype(np.int64)
                de[j:j + 16384] = (((X % p) * r) % p).sum(-1) % p
            s.keys.append((p, r, de))
        s.x0 = None; s.V = emb.shape[0]; s.D = D
    def key_bytes(s): return 2 * 8 * s.V
    def __getitem__(s, ids):
        x0 = s.x0; ids = np.asarray(ids)
        ok = isinstance(x0, np.ndarray) and x0.dtype == np.int64 and x0.shape == (len(ids), s.D) and int(np.abs(x0).max()) < (1 << 40)
        if ok:
            for p, r, de in s.keys: ok = ok and bool(((((x0 % p) * r) % p).sum(-1) % p == de[ids]).all())
        VER["checks"] += len(ids)
        if not ok:
            VER["fail"].append((-1, "embedding")); raise ValueError("reject embedding")
        return x0.astype(np.float64) / ONE
def make_verifier(m, seed=2026):
    rng = np.random.default_rng(seed); vm = IntModel.__new__(IntModel); vm.c = m.c; vm.cos, vm.sin, vm.gf = m.cos, m.sin, m.gf; kb = 0; wb = 0
    vm.layers = []
    for i, L in enumerate(m.layers):
        d = dict(g1=L["g1"], g2=L["g2"]); kb += 16 * len(L["g1"])
        for nm in ("q", "k", "v", "o", "gate", "up", "down"):
            d[nm] = VLin(L[nm], rng, f"L{i}.{nm}"); kb += d[nm].key_bytes(); wb += L[nm].w.size
        vm.layers.append(d)
    vm.head = VLin(m.head, rng, "head"); kb += vm.head.key_bytes(); wb += m.head.w.size
    vm.emb = VEmb(m.emb, rng); kb += vm.emb.key_bytes() + 8 * len(m.gf)
    return vm, kb, wb
def verify_forward(vm, ids, x0, trace):
    VER.update(trace=trace, i=0, fail=[], checks=0); vm.emb.x0 = x0
    try:
        lg = vm(list(ids))
        if VER["i"] != len(trace): return False, None, "trace length"
        return True, lg, None
    except Exception as e:
        return False, None, (VER["fail"][:1] or [repr(e)[:80]])[0]

if TINY:      # local preflight: random tiny model, no downloads
    _r = np.random.default_rng(0); c = dict(D=64, H=4, KV=2, hd=16, I=128, V=512, NL=2, T=256, theta=1e4, eps=1e-6, tie=True)
    _sh = {"embed_tokens.weight": (512, 64), "q_proj.weight": (64, 64), "q_proj.bias": (64,), "k_proj.weight": (32, 64), "k_proj.bias": (32,),
           "v_proj.weight": (32, 64), "v_proj.bias": (32,), "o_proj.weight": (64, 64), "gate_proj.weight": (128, 64), "up_proj.weight": (128, 64),
           "down_proj.weight": (64, 128)}
    _cache = {}
    def get(k):
        if k not in _cache:
            if k.endswith("norm.weight") or "layernorm" in k: _cache[k] = (1 + 0.1 * _r.standard_normal(64)).astype(np.float32)
            else:
                sfx = [s_ for s_ in _sh if k.endswith(s_)][0]; _cache[k] = (0.08 * _r.standard_normal(_sh[sfx])).astype(np.float32)
        return _cache[k]
    class _Tok:
        def __call__(s, text): return {"input_ids": [int(x) for x in np.random.default_rng(len(text) % 97).integers(0, 512, 8)]}
        def decode(s, t): return " ".join(map(str, t))
    tok = _Tok(); ids_all = [int(x) for x in _r.integers(0, 512, 4000)]
else:
    MODEL = os.environ.get("ZL_MODEL", "Qwen/Qwen2.5-0.5B")
    get, c, tok = open_hf(MODEL); c["T"] = 512; ids_all = tok(TEXT)["input_ids"]; R["model"] = MODEL
m = IntModel(get, c); P("model built", round(time.time() - t0))
BIG = (not TINY) and c["NL"] > 24 and c["D"] >= 3000             # 7B-class: smaller windows to fit Kaggle's CPU time
WN = 32 if TINY else (48 if BIG else 128); WF = 24 if TINY else (32 if BIG else 64)
tk = time.time(); vm, key_bytes, w_bytes = make_verifier(m); R["key_build_s"] = round(time.time() - tk, 1)
w_unique = w_bytes - (m.head.w.size if c["tie"] else 0)
R["sizes"] = dict(verifier_key_MB=round(key_bytes / 1e6, 3), int8_weight_MB=round(w_unique / 1e6, 2), ratio=round(w_unique / key_bytes, 1),
                  trace_ints_per_token=int(sum(L[n].w.shape[1] for L in m.layers for n in ("q", "k", "v", "o", "gate", "up", "down")) + m.head.w.shape[1]))
R["sizes"]["trace_MB_per_token_int64"] = round(R["sizes"]["trace_ints_per_token"] * 8 / 1e6, 3); save(); P("sizes", R["sizes"])


# ======================= COMMIT-AND-AUDIT ZERO-LEAK (2026-11-12) =======================
# Problem: the full trace is 3.65 MB/token at 0.5B. Fix: the prover sends one 32-byte Merkle root over its integer intermediates; the verifier
# (weightless key, as before) challenges random cells (layer L, position t); the prover opens only what that cell needs:
#   X(L,s), KV(L,s) for s <= t (residual rows + k/v accumulators, to rebuild the causal K/V), Q(L,t), REST(L,t) (o/gate/up/down accumulators), X(L+1,t).
# Final cell (L = NL, t): X(NL,t) + HEAD(t) (head accumulator) -> exact logits -> the sampler must give the recorded token t+1 (zero-leak check).
# Credit: Merkle commitments + random audits are standard (e.g. CommitLLM's commit-and-audit, 2511.02620's sampled verification); what is
# tested here is that every audited cell is checked EXACTLY (no tolerance), with no weights on the verifier, including attention.
# v2 (2026-11-12): K/V commitment - cached, roped K/V rows are committed leaves KR(L,s); a cell re-derives only its own row t (and checks it
# against KR(L,t)), cutting the opened bytes. New gate A6: a forged cached K row is rejected by its owner cell (L, s*). Gates A1-A5 unchanged.
# PRE-REGISTERED GATES (0.5B, one 128-token window + 16 generated tokens):
#  A1 honest: 40 random cells (incl. final cells) all verify; report root bytes, opened bytes per audit (mean/max), verifier ms per audit.
#  A2 cheating model (7-bit down_proj in layer 12): every audited cell at layer 12 rejected, every other cell accepted (localisation).
#  A3 consistent fake attention at one layer for rows >= T/2: audited cells (L*, t >= T/2) rejected; others accepted.
#  A4 zero-leak: honest generation's final cells all accepted; a substituted token at position j is rejected by the final cell at j.
#  A5 lying opening (any opened leaf altered) -> Merkle rejection, 20/20.
import hashlib
def Hb(*arrs):
    hh = hashlib.sha256()
    for a in arrs: hh.update(np.ascontiguousarray(a, dtype=np.int64).tobytes()); hh.update(b"|")
    return hh.digest()
def merkle(leaves):
    lv = [hashlib.sha256(b"L" + l).digest() for l in leaves]; levels = [lv]
    while len(levels[-1]) > 1:
        cur = levels[-1]; nxt = [hashlib.sha256(b"N" + cur[i] + (cur[i + 1] if i + 1 < len(cur) else cur[i])).digest() for i in range(0, len(cur), 2)]; levels.append(nxt)
    return levels
def mproof(levels, idx):
    pr = []
    for lv in levels[:-1]:
        sib = idx ^ 1; pr.append(lv[sib] if sib < len(lv) else lv[idx]); idx >>= 1
    return pr
def mverify(root, leaf, idx, pr):
    hcur = hashlib.sha256(b"L" + leaf).digest()
    for sib in pr:
        hcur = hashlib.sha256(b"N" + (hcur + sib if idx % 2 == 0 else sib + hcur)).digest(); idx >>= 1
    return hcur == root
LIN = ("q", "k", "v", "o", "gate", "up", "down")
def prover_trace(m, ids, fake=None, cheat_layer=None, fake_k=None):
    """exact forward recording per layer: residual input X, accumulators. fake=(layer, fn) alters attention probs at that layer (all heads)."""
    c_ = m.c; T = len(ids); H_, KV_, hd = c_["H"], c_["KV"], c_["hd"]; INV = int(round((1 << 16) / math.sqrt(hd)))
    x = np.rint(np.asarray(m.emb[ids], dtype=np.float64) * ONE).astype(np.int64); valid = np.tril(np.ones((T, T), bool))
    X, A, KR = [], [], []
    def lin(q, xin):
        xq, s = qrow(xin, 15); acc = exact_mm(xq, q.w); return acc, requant(q.M, q.Mbits, q.b, acc, s)
    for li, L in enumerate(m.layers):
        X.append(x.copy()); acc = {}
        a = irms(x, L["g1"])
        acc["q"], qo = lin(L["q"], a); acc["k"], ko = lin(L["k"], a); acc["v"], vo = lin(L["v"], a)
        q = m.rope(qo.reshape(T, H_, hd).transpose(1, 0, 2)); k = m.rope(ko.reshape(T, KV_, hd).transpose(1, 0, 2)); v = vo.reshape(T, KV_, hd).transpose(1, 0, 2)
        if fake_k is not None and fake_k[0] == li: k = k.copy(); k[:, fake_k[1], :] += fake_k[2]      # attack A6: forged cached K row
        KR.append((k.copy(), v.copy()))
        out = np.empty((H_, T, hd), np.int64)
        for hh in range(H_):
            kh = hh // (H_ // KV_); qq, sq = qrow(q[hh]); kk, sk = qrow(k[kh])
            sc = shr(exact_mm(qq, kk.T) * INV, (F - sq - sk.T) + 16); p = isoftmax(sc, valid)
            if fake is not None and fake[0] == li: p = fake[1](p)
            out[hh] = shr(np.matmul(p.astype(np.int64), v[kh].astype(np.int64)), 20)
        acc["o"], oo = lin(L["o"], out.transpose(1, 0, 2).reshape(T, H_ * hd)); x = x + oo
        mm_ = irms(x, L["g2"]); acc["gate"], go = lin(L["gate"], mm_); acc["up"], uo = lin(L["up"], mm_)
        dn = L["down"] if cheat_layer != li else CHEAT_DOWN
        acc["down"], do_ = lin(dn, shr(isilu(go) * uo, F)); x = x + do_
        A.append(acc)
    X.append(x.copy()); hacc, lg = lin(m.head, irms(x, m.gf))
    return dict(X=X, A=A, KR=KR, head=hacc, logits=lg, T=T, ids=list(ids))
def leaves_of(tr, NL):
    T = tr["T"]; names, leaves = [], []
    for L in range(NL + 1):
        for t in range(T):
            names.append(("X", L, t)); leaves.append(Hb(tr["X"][L][t]))
    for L in range(NL):
        for t in range(T):
            a = tr["A"][L]
            names.append(("KV", L, t)); leaves.append(Hb(a["k"][t], a["v"][t]))
            names.append(("KR", L, t)); leaves.append(Hb(tr["KR"][L][0][:, t, :], tr["KR"][L][1][:, t, :]))
            names.append(("Q", L, t)); leaves.append(Hb(a["q"][t]))
            names.append(("R", L, t)); leaves.append(Hb(a["o"][t], a["gate"][t], a["up"][t], a["down"][t]))
    for t in range(T):
        names.append(("HEAD", NL, t)); leaves.append(Hb(tr["head"][t]))
    return {n: i for i, n in enumerate(names)}, leaves
class Prover:
    def __init__(s, tr, NL):
        s.tr, s.NL = tr, NL; s.index, s.leaves = leaves_of(tr, NL); s.levels = merkle(s.leaves); s.root = s.levels[-1][0]
    def value(s, name):
        kind, L, t = name; a = s.tr["A"][L] if kind in ("KV", "Q", "R") else None
        if kind == "KR": return [s.tr["KR"][L][0][:, t, :], s.tr["KR"][L][1][:, t, :]]
        if kind == "X": return [s.tr["X"][L][t]]
        if kind == "KV": return [a["k"][t], a["v"][t]]
        if kind == "Q": return [a["q"][t]]
        if kind == "R": return [a["o"][t], a["gate"][t], a["up"][t], a["down"][t]]
        return [s.tr["head"][t]]
    def open(s, names, lie=None):
        out = {}
        for n in names:
            vals = [v.copy() for v in s.value(n)]
            if lie is not None and n == lie: vals[0] = vals[0].copy(); vals[0][0] += 1
            out[n] = (vals, mproof(s.levels, s.index[n]))
        return out
def needed(cell, NL):
    L, t = cell
    if L == NL: return [("X", NL, t), ("HEAD", NL, t)]
    # v2 (K/V commitment): rows s < t come from the committed, already-roped K/V (KR); only row t is re-derived and must match KR(L, t).
    return [("X", L, t), ("KV", L, t)] + [("KR", L, s) for s in range(t + 1)] + [("Q", L, t), ("R", L, t), ("X", L + 1, t)]
def fr_row(vl, xq_row, acc_row):
    if acc_row.shape != (vl.N,) or int(np.abs(acc_row).max()) > 127 * 32767 * vl.K: return False
    for p, r, d in vl.keys:
        if int((((acc_row % p) * r) % p).sum() % p) != int((((xq_row % p) * d) % p).sum() % p): return False
    return True
def rq(vl, acc_row, s):
    return requant(vl.M, vl.Mbits, vl.b, acc_row[None, :], s)[0]
def verify_cell(vm, cfg, root, index, cell, opened, ids, token_check=None):
    """returns (ok, reason). vm = weightless verifier (VLin keys, norms, rope tables, emb digests)."""
    NL = len(vm.layers)
    for n in needed(cell, NL):
        if n not in opened: return False, f"missing {n}"
        vals, pr = opened[n]
        if not mverify(root, Hb(*vals), index[n], pr): return False, f"merkle {n}"
    L, t = cell; H_, KV_, hd = cfg["H"], cfg["KV"], cfg["hd"]; INV = int(round((1 << 16) / math.sqrt(hd)))
    if L == NL:
        x = opened[("X", NL, t)][0][0]; hacc = opened[("HEAD", NL, t)][0][0]
        xq, s = qrow(irms(x[None, :], vm.gf), 15)
        if not fr_row(vm.head, xq[0], hacc): return False, "head"
        lg = rq(vm.head, hacc, s[0])
        if token_check is not None and not token_check(t, lg): return False, f"sampler@{t}"
        return True, None
    V_ = vm.layers[L]
    if L == 0:                                   # embedding row of position t must match token t
        xr = opened[("X", 0, t)][0][0]
        for p, r, de in vm.emb.keys:
            if int((((xr % p) * r) % p).sum() % p) != int(de[ids[t]]): return False, f"embedding@{t}"
    xr = opened[("X", L, t)][0][0]; kacc, vacc = opened[("KV", L, t)][0]
    xq, sh = qrow(irms(xr[None, :], V_["g1"]), 15)
    if not (fr_row(V_["k"], xq[0], kacc) and fr_row(V_["v"], xq[0], vacc)): return False, f"kv@{t}"
    ko = rq(V_["k"], kacc, sh[0]).reshape(KV_, hd); vo = rq(V_["v"], vacc, sh[0]).reshape(KV_, hd)
    kt = shr(ko * vm.cos[t] + rot_half(ko) * vm.sin[t], F)
    krk, krv = opened[("KR", L, t)][0]
    if not (np.array_equal(krk, kt) and np.array_equal(krv, vo)): return False, f"kr@{t}"
    ks = [opened[("KR", L, s_)][0][0] for s_ in range(t + 1)]; vs = [opened[("KR", L, s_)][0][1] for s_ in range(t + 1)]
    K_ = np.stack(ks, 1); Vv = np.stack(vs, 1)                 # (KV, t+1, hd)
    x = opened[("X", L, t)][0][0]; qacc = opened[("Q", L, t)][0][0]; oacc, gacc, uacc, dacc = opened[("R", L, t)][0]
    xq, sh = qrow(irms(x[None, :], V_["g1"]), 15)
    if not fr_row(V_["q"], xq[0], qacc): return False, "q"
    qo = rq(V_["q"], qacc, sh[0]).reshape(H_, hd); qo = shr(qo * vm.cos[t] + rot_half(qo) * vm.sin[t], F)
    out = np.empty((H_, hd), np.int64); valid = np.ones((1, t + 1), bool)
    for hh in range(H_):
        kh = hh // (H_ // KV_); qq, sq = qrow(qo[hh][None, :]); kk, sk = qrow(K_[kh])
        sc = shr(exact_mm(qq, kk.T) * INV, (F - sq - sk.T) + 16); p = isoftmax(sc, valid)
        out[hh] = shr(np.matmul(p.astype(np.int64), Vv[kh].astype(np.int64)), 20)[0]
    oq, so = qrow(out.reshape(1, H_ * hd), 15)
    if not fr_row(V_["o"], oq[0], oacc): return False, "o"
    x2 = x + rq(V_["o"], oacc, so[0]); mq, sm = qrow(irms(x2[None, :], V_["g2"]), 15)
    if not (fr_row(V_["gate"], mq[0], gacc) and fr_row(V_["up"], mq[0], uacc)): return False, "gate/up"
    act = shr(isilu(rq(V_["gate"], gacc, sm[0])) * rq(V_["up"], uacc, sm[0]), F)
    dq, sd = qrow(act[None, :], 15)
    if not fr_row(V_["down"], dq[0], dacc): return False, "down"
    if not np.array_equal(x2 + rq(V_["down"], dacc, sd[0]), opened[("X", L + 1, t)][0][0]): return False, "residual"
    return True, None
def opened_bytes(op): return int(sum(sum(v.size for v in vals) * 8 + 32 * len(pr) for vals, pr in op.values()))

AR = {"version": "v2 K/V commitment"}
NLm = len(m.layers); T_A = 32 if TINY else (64 if BIG else 128)          # 7B: 64-token window (trace memory)
ST_A = min(5000, len(ids_all) - T_A - 1); wA = ids_all[ST_A:ST_A + T_A]; trA = prover_trace(m, wA); assert h(trA["logits"]) == h(m(wA)), "audit trace differs from model"
pv = Prover(trA, NLm); rngA = np.random.default_rng(11)
cells = [(int(rngA.integers(NLm + 1)), int(rngA.integers(T_A))) for _ in range(40)]
a1 = dict(ok=0, n=0, bytes=[], ms=[], fails=[])
for cell in cells:
    op = pv.open(needed(cell, NLm)); a = time.time(); ok, why = verify_cell(vm, c, pv.root, pv.index, cell, op, wA)
    a1["ms"].append(round(1e3 * (time.time() - a), 1)); a1["bytes"].append(opened_bytes(op)); a1["n"] += 1; a1["ok"] += ok
    if not ok: a1["fails"].append([cell, why])
AR["A1_honest"] = dict(cells=a1["n"], verified=a1["ok"], root_bytes=32, opened_KB_mean=round(np.mean(a1["bytes"]) / 1e3, 1), opened_KB_max=round(max(a1["bytes"]) / 1e3, 1),
                       verifier_ms_mean=round(float(np.mean(a1["ms"])), 1), verifier_ms_max=round(float(max(a1["ms"])), 1), fails=a1["fails"][:3], **{"pass": a1["ok"] == a1["n"]})
R["audit"] = AR; save(); P("A1", AR["A1_honest"])
# A2 cheating model at layer CL: localisation
CL = NLm // 2
wq_ = np.asarray(get(PFX.format(CL) + "mlp.down_proj.weight"), dtype=np.float64).T; sc_ = np.abs(wq_).max(0); sc_ = np.where(sc_ == 0, 1e-12, sc_) / 63.0
CHEAT_DOWN = QLin.__new__(QLin); CHEAT_DOWN.w = np.clip(np.rint(wq_ / sc_), -63, 63).astype(np.int8); CHEAT_DOWN.M = np.rint(sc_ * (1 << 28)).astype(np.int64)
CHEAT_DOWN.Mbits = int(bitlen(np.array([CHEAT_DOWN.M.max()]))[0]); CHEAT_DOWN.b = None
pc = Prover(prover_trace(m, wA, cheat_layer=CL), NLm); a2 = dict(at_layer_rejected=0, at_layer=0, elsewhere_accepted=0, elsewhere=0)
for t in rngA.integers(T_A, size=6):
    ok, _ = verify_cell(vm, c, pc.root, pc.index, (CL, int(t)), pc.open(needed((CL, int(t)), NLm)), wA); a2["at_layer"] += 1; a2["at_layer_rejected"] += (not ok)
for _ in range(10):
    L = int(rngA.integers(NLm + 1))
    if L == CL: continue
    t = int(rngA.integers(T_A)); ok, why = verify_cell(vm, c, pc.root, pc.index, (L, t), pc.open(needed((L, t), NLm)), wA); a2["elsewhere"] += 1; a2["elsewhere_accepted"] += ok
a2["pass"] = a2["at_layer_rejected"] == a2["at_layer"] and a2["elsewhere_accepted"] == a2["elsewhere"]; AR["A2_cheat_localised"] = a2; save(); P("A2", a2)
# A3 consistent fake attention at layer FL for rows >= T/2
FL = 1 if NLm > 1 else 0
def fakefn(p):
    p = p.copy(); T_ = p.shape[0]
    for row in range(max(1, T_ // 2), T_):
        a0, a1_ = int(p[row, 0]), int(p[row, row]); p[row, 0], p[row, row] = a1_, a0
    return p
pf = Prover(prover_trace(m, wA, fake=(FL, fakefn)), NLm); a3 = dict(late_rejected=0, late=0, early_accepted=0, early=0)
for t in rngA.integers(T_A // 2 + 1, T_A, size=5):
    ok, _ = verify_cell(vm, c, pf.root, pf.index, (FL, int(t)), pf.open(needed((FL, int(t)), NLm)), wA); a3["late"] += 1; a3["late_rejected"] += (not ok)
for t in rngA.integers(0, T_A // 2, size=5):
    ok, _ = verify_cell(vm, c, pf.root, pf.index, (FL, int(t)), pf.open(needed((FL, int(t)), NLm)), wA); a3["early"] += 1; a3["early_accepted"] += ok
a3["pass"] = a3["late_rejected"] == a3["late"] and a3["early_accepted"] == a3["early"]; AR["A3_fake_attention_localised"] = a3; save(); P("A3", a3)
# A4 zero-leak with final cells: generation with protocol seed, substitution at position j
def proto_seed(prompt, nonce): return int.from_bytes(hashlib.sha256(np.asarray(prompt, np.int64).tobytes() + nonce).digest()[:8], "big")
prompt = tok("The printing press changed Europe because")["input_ids"]; nonce = b"verifier-nonce-0001"; seed = proto_seed(prompt, nonce); NGa = 8 if (TINY or BIG) else 16; tnum = temp_num(0.8)
gen_ = list(prompt)
for t in range(NGa): gen_.append(sample_int(m(gen_)[-1], seed, t, tnum)[0])
def tok_check_for(seq):
    def chk(pos, lg):
        j = pos - (len(prompt) - 1)
        if j < 0 or j >= NGa: return True
        return sample_int(lg, seed, j, tnum)[0] == seq[pos + 1]
    return chk
ph = Prover(prover_trace(m, gen_), NLm); okall = all(verify_cell(vm, c, ph.root, ph.index, (NLm, t), ph.open(needed((NLm, t), NLm)), gen_, tok_check_for(gen_))[0]
                                                   for t in range(len(prompt) - 1, len(gen_) - 1))
jpos = len(prompt) + NGa // 2; bad = list(gen_); lgj = m(bad[:jpos])[-1]; alt = int(lgj.argmax()); bad[jpos] = alt if alt != bad[jpos] else int(np.argsort(lgj)[-2])
pb = Prover(prover_trace(m, bad), NLm); okb, whyb = verify_cell(vm, c, pb.root, pb.index, (NLm, jpos - 1), pb.open(needed((NLm, jpos - 1), NLm)), bad, tok_check_for(bad))
AR["A4_zero_leak_final_cells"] = {"honest_all_final_cells_ok": bool(okall), "substitution_rejected": (not okb), "reason": str(whyb), "pass": bool(okall and not okb)}; save(); P("A4", AR["A4_zero_leak_final_cells"])
# A6 forged cached K row (layer FL, position s*), downstream consistent: the cell that owns row s* must reject;
# cells at later positions of that layer use the committed row and pass (detection is per audited cell, reported, not hidden)
SS = T_A // 4; pk = Prover(prover_trace(m, wA, fake_k=(FL, SS, 1 << 12)), NLm)
ok_own, why_own = verify_cell(vm, c, pk.root, pk.index, (FL, SS), pk.open(needed((FL, SS), NLm)), wA)
later = [verify_cell(vm, c, pk.root, pk.index, (FL, int(t)), pk.open(needed((FL, int(t)), NLm)), wA)[0] for t in rngA.integers(SS + 1, T_A, size=5)]
AR["A6_forged_cached_K"] = {"owner_cell_rejected": (not ok_own), "reason": str(why_own), "later_cells_passing": f"{sum(later)}/{len(later)}",
                            "note": "caught when the owner cell (layer, s*) is audited; probability per random cell = 1/((NL+1)*T)", "pass": (not ok_own)}
save(); P("A6", AR["A6_forged_cached_K"])
# A5 lying openings
rej = 0
for _ in range(20):
    cell = (int(rngA.integers(NLm + 1)), int(rngA.integers(1, T_A))); names = needed(cell, NLm); lie = names[int(rngA.integers(len(names)))]
    ok, why = verify_cell(vm, c, pv.root, pv.index, cell, pv.open(names, lie=lie), wA); rej += (not ok)
AR["A5_lying_openings"] = {"rejected": f"{rej}/20", "pass": rej == 20}
AR["verdict"] = "PASS" if all(AR[k]["pass"] for k in AR if k.startswith("A") and isinstance(AR[k], dict)) else "FAIL"
AR["full_trace_MB_per_window"] = round(R["sizes"]["trace_MB_per_token_int64"] * T_A, 1)
R["audit"] = AR; R["seconds"] = round(time.time() - t0); P(json.dumps(AR, indent=1)); save()
