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

# PRE-REGISTERED (2026-11-11): PASS if exact-BitNet ppl is within 1.0% of HF BitNet (float32 compute) on 8 x 256 WikiText-2 tokens,
# AND logits hash identical for default vs permuted reduction order. Report the gap and HF bf16 vs fp32 for context.
OUT = "/kaggle/working/results.json"; R = {}; t0 = time.time()
import torch, glob as _glob
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM
NAME = "microsoft/bitnet-b1.58-2B-4T"
path = snapshot_download(NAME, allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"])
idx = {}
for f in _glob.glob(os.path.join(path, "*.safetensors")):
    with safe_open(f, "pt") as fh:
        for k in fh.keys(): idx[k] = f
def getT(k):
    with safe_open(idx[k], "pt") as fh: return fh.get_tensor(k)
cf = json.load(open(os.path.join(path, "config.json"))); R["config"] = {k: cf.get(k) for k in ("hidden_size", "num_attention_heads", "num_key_value_heads", "intermediate_size", "num_hidden_layers", "rope_theta", "rms_norm_eps", "vocab_size", "hidden_act", "quantization_config")}
P("config", R["config"])
D, H, KV = cf["hidden_size"], cf["num_attention_heads"], cf["num_key_value_heads"]; hd = cf.get("head_dim") or D // H
EPS = cf["rms_norm_eps"]; THETA = float(cf.get("rope_theta") or (cf.get("rope_parameters") or {}).get("rope_theta"))
def unpack(packed, out_features):
    p = packed.to(torch.int64); rows = p.shape[0]; un = torch.zeros((rows * 4, p.shape[1]), dtype=torch.int64)
    for i in range(4): un[i * rows:(i + 1) * rows] = (p >> (2 * i)) & 3
    return (un - 1)[:out_features].numpy()                         # ternary {-1,0,1}, shape (out, in)
class BitQLin:
    """exact AutoBitLinear (offline): x_q = round_half_up(x*127/absmax) per row (int8), acc = x_q . w (ternary), y = acc*absmax*ws/127."""
    S = 40
    def __init__(self, key, out_features):
        w = unpack(getT(key + ".weight"), out_features); self.w = np.ascontiguousarray(w.T).astype(np.float64)   # (in, out)
        ws = float(getT(key + ".weight_scale").float().reshape(-1)[0]); self.ws = ws
        # AutoBitLinear offline mode: output = linear(fake_quant(x), W_ternary) * weight_scale  (MULTIPLY by ws)
        self.M = int(round((1 << self.S) * ws / 127.0)); self.Mb = int(self.M).bit_length()
    def __call__(self, x):                                         # x int64 Q16 (R, in)
        am = np.maximum(np.abs(x).max(-1, keepdims=True), 1)
        xq = np.clip((2 * 127 * x + am) // (2 * am), -128, 127)    # integer round-half-up of x*127/absmax
        acc = exact_mm(xq, self.w)                                 # exact: |acc| <= 128*in < 2^53
        t = acc * am                                               # < 2^20 * 2^31
        d = np.maximum(bitlen(np.abs(t).max(-1, keepdims=True)) + self.Mb - 62, 0)
        return shr(shr(t, d) * self.M, self.S - d)                 # Q16
def irms_w(x, g):
    """integer RMSNorm WITH eps (BitNet rms_norm_eps): eps added in the units of the shifted mean square."""
    k = np.maximum(bitlen(np.abs(x).max(-1, keepdims=True)) - 24, 0)
    xs = shr(x, k)
    ms = (xs * xs).sum(-1, keepdims=True) // x.shape[-1]
    eps_u = np.rint(EPS * np.power(2.0, 32 - 2 * k)).astype(np.int64)     # eps in units of 2^(2(k-16))
    r = np.maximum(isqrt(ms + eps_u), 1)
    y = (x << F) // (r << k)
    return shr(y * g, F)
def relu2_mul(r, u):
    """exact-spec relu(g)^2 * up in Q16 without int64 overflow (v4 fix: layer 0 has |g|,|u| ~ 2^28 -> naive product ~2^68).
    Per-row shifts bring each factor to <= 20 bits, product < 2^61, then rescale to Q16: r^2*u / 2^32."""
    ka = np.maximum(bitlen(np.abs(r).max(-1, keepdims=True)) - 20, 0)
    ku = np.maximum(bitlen(np.abs(u).max(-1, keepdims=True)) - 20, 0)
    rs = shr(r, ka); us = shr(u, ku)
    return shr(rs * rs * us, 32 - 2 * ka - ku)              # negative shift = exact left shift
class ExactBitNet:
    def __init__(self, T=512):
        NL = cf["num_hidden_layers"]; I = cf["intermediate_size"]; q = lambda k: np.rint(getT(k).float().numpy().astype(np.float64) * ONE).astype(np.int64)
        self.emb = getT("model.embed_tokens.weight").float().numpy()
        self.layers = []
        for i in range(NL):
            p = f"model.layers.{i}."
            self.layers.append(dict(g1=q(p + "input_layernorm.weight"), g2=q(p + "post_attention_layernorm.weight"),
                gs_a=q(p + "self_attn.attn_sub_norm.weight"), gs_f=q(p + "mlp.ffn_sub_norm.weight"),
                q=BitQLin(p + "self_attn.q_proj", H * hd), k=BitQLin(p + "self_attn.k_proj", KV * hd), v=BitQLin(p + "self_attn.v_proj", KV * hd),
                o=BitQLin(p + "self_attn.o_proj", D), gate=BitQLin(p + "mlp.gate_proj", I), up=BitQLin(p + "mlp.up_proj", I), down=BitQLin(p + "mlp.down_proj", D)))
            if i % 10 == 0: P("  built layer", i)
        self.gf = q("model.norm.weight")
        self.head = qlin_chunked(self.emb if "lm_head.weight" not in idx else getT("lm_head.weight").float().numpy())
        cos, sin = rope_tables(T, hd, THETA); self.cos = np.rint(cos * ONE).astype(np.int64); self.sin = np.rint(sin * ONE).astype(np.int64)
    def __call__(self, ids):
        T = len(ids); x = np.rint(np.asarray(self.emb[ids], dtype=np.float64) * ONE).astype(np.int64)
        valid = np.tril(np.ones((T, T), bool)); INV = int(round((1 << 16) / math.sqrt(hd))); rep = H // KV
        rope = lambda z: shr(z * self.cos[:T] + rot_half(z) * self.sin[:T], F)
        for L in self.layers:
            a = irms_w(x, L["g1"])
            qh = rope(L["q"](a).reshape(T, H, hd).transpose(1, 0, 2)); kh = rope(L["k"](a).reshape(T, KV, hd).transpose(1, 0, 2))
            vh = L["v"](a).reshape(T, KV, hd).transpose(1, 0, 2); out = np.empty((H, T, hd), np.int64)
            for hh in range(H):
                j = hh // rep; qq, sq = qrow(qh[hh]); kk, sk = qrow(kh[j])
                sc = shr(exact_mm(qq, kk.T) * INV, (F - sq - sk.T) + 16); p = isoftmax(sc, valid)
                out[hh] = shr(np.matmul(p.astype(np.int64), vh[j].astype(np.int64)), 20)
            att = irms_w(out.transpose(1, 0, 2).reshape(T, H * hd), L["gs_a"])
            x = x + L["o"](att)
            m = irms_w(x, L["g2"]); g = L["gate"](m); r = np.maximum(g, 0)
            hmid = relu2_mul(r, L["up"](m))                         # relu(g)^2 * up, Q16 (overflow-safe, v4)
            x = x + L["down"](irms_w(hmid, L["gs_f"]))
        return self.head(irms_w(x, self.gf))
tok = AutoTokenizer.from_pretrained(path)
try:
    from datasets import load_dataset
    TEXT = (chr(10) * 2).join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
except Exception:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    TEXT = (chr(10) * 2).join(pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"))["text"])
ids = tok(TEXT)["input_ids"]; ws_ = [ids[i * 256:(i + 1) * 256] for i in range(8)]

# ===== ZERO-LEAK VERIFIED BITNET (2026-11-12), Kaggle CPU. Combines exact BitNet (cross-device PASS, b571ee87) with the weightless verifier.
# Verifier holds NO weight matrices: Freivalds digests of the ternary matrices (mod 2 primes), the per-matrix scale M, norm gains, embedding-row
# digests, and digests of the int8 head. Prover sends every linear's integer accumulator + embedding rows; the verifier replays norms,
# activation quantisation, RoPE, attention, ReLU^2 and the sampler exactly.
# v2 (2026-11-12): v1 FAILED B1 (honest rejected) - my int64 overflow in the digest build (blk % p turned -1 into p-1). Fixed; same gates.
# PRE-REGISTERED GATES: B1 honest 2 x 64-token windows verify and verifier logits == prover logits == plain model;
#  B2 forged accumulator +-1: 20/20 rejected; B3 consistent fake attention (one layer/head, downstream honest): 5/5 rejected;
#  B4 sampled generation with protocol seed verifies; hidden-bit token substitution 5/5 rejected.
#  Report key MB vs 2-bit packed weight MB and vs int8 head MB.
import hashlib
OUT = "/kaggle/working/results.json"; R = {}
def save(): json.dump(R, open(OUT, "w"), indent=1)
P1, P2 = 2147483647, 2147483629
TR = {"on": False, "acc": []}
_bit_call = BitQLin.__call__; _q_call = QLin.__call__
def bit_call(self, x):
    am = np.maximum(np.abs(x).max(-1, keepdims=True), 1)
    xq = np.clip((2 * 127 * x + am) // (2 * am), -128, 127)
    acc = exact_mm(xq, self.w)
    if TR.get("hook"): acc = TR["hook"](len(TR["acc"]), acc)
    if TR["on"]: TR["acc"].append(acc)
    t = acc * am; d = np.maximum(bitlen(np.abs(t).max(-1, keepdims=True)) + self.Mb - 62, 0)
    return shr(shr(t, d) * self.M, self.S - d)
def q_call(self, x):
    xq, s = qrow(x, 15); N = self.w.shape[1]
    acc = exact_mm(xq, self.w) if N <= 32768 else np.concatenate([exact_mm(xq, self.w[:, j:j + 32768]) for j in range(0, N, 32768)], -1)
    if TR["on"]: TR["acc"].append(acc)
    drop = np.maximum(bitlen(np.abs(acc).max(-1, keepdims=True)) + self.Mbits - 62, 0)
    y = shr(shr(acc, drop) * self.M, 28 - s - drop)
    return y if self.b is None else y + self.b
BitQLin.__call__ = bit_call; QLin.__call__ = q_call
def prove(model, ids, hook=None):
    TR.update(on=True, acc=[], hook=hook)
    try: lg = model(ids)
    finally: TR.update(on=False, hook=None)
    return lg, TR["acc"], np.rint(np.asarray(model.emb[ids], dtype=np.float64) * ONE).astype(np.int64)
VER = dict(trace=None, i=0, fail=[])
def fr_keys(wmat_int_cols, K, N, rng):
    """digests d = W r mod p for W given as a function returning int64 column blocks (K, n)."""
    keys = []
    for p in (P1, P2):
        r = rng.integers(0, p, N, dtype=np.int64); d = np.zeros(K, np.int64)
        for j in range(0, N, 4096):
            blk = wmat_int_cols(j, min(j + 4096, N)); d = (d + (blk @ r[j:j + 4096]) % p) % p          # v2 fix: no "blk % p" (int64 overflow: (p-1)*r*4096 > 2^63)
        keys.append((p, r, d))
    return keys
def fr_check(keys, xq, acc):
    for p, r, d in keys:
        if not bool(((((acc % p) * r) % p).sum(-1) % p == (((xq % p) * d) % p).sum(-1) % p).all()): return False
    return True
class VBit:
    def __init__(s, q, rng, name):
        s.M, s.Mb, s.S, s.name = q.M, q.Mb, q.S, name; K, N = q.w.shape; s.N, s.K = N, K
        s.keys = fr_keys(lambda a, b: q.w[:, a:b].astype(np.int64), K, N, rng)
    def kbytes(s): return 2 * 8 * s.K + 16
    def __call__(s, x):
        am = np.maximum(np.abs(x).max(-1, keepdims=True), 1); xq = np.clip((2 * 127 * x + am) // (2 * am), -128, 127)
        acc = VER["trace"][VER["i"]]; VER["i"] += 1
        ok = isinstance(acc, np.ndarray) and acc.dtype == np.int64 and acc.shape == (x.shape[0], s.N) and int(np.abs(acc).max()) <= 128 * s.K and fr_check(s.keys, xq, acc)
        if not ok: VER["fail"].append(s.name); raise ValueError("reject " + s.name)
        t = acc * am; d = np.maximum(bitlen(np.abs(t).max(-1, keepdims=True)) + s.Mb - 62, 0)
        return shr(shr(t, d) * s.M, s.S - d)
class VQ:
    def __init__(s, q, rng, name):
        s.M, s.Mbits, s.b, s.name = q.M, q.Mbits, q.b, name; K, N = q.w.shape; s.N, s.K = N, K
        s.keys = fr_keys(lambda a, b: q.w[:, a:b].astype(np.int64), K, N, rng)
    def kbytes(s): return 2 * 8 * s.K + 8 * s.N
    def __call__(s, x):
        xq, sh = qrow(x, 15); acc = VER["trace"][VER["i"]]; VER["i"] += 1
        ok = isinstance(acc, np.ndarray) and acc.dtype == np.int64 and acc.shape == (x.shape[0], s.N) and int(np.abs(acc).max()) <= 127 * 32767 * s.K and fr_check(s.keys, xq, acc)
        if not ok: VER["fail"].append(s.name); raise ValueError("reject " + s.name)
        drop = np.maximum(bitlen(np.abs(acc).max(-1, keepdims=True)) + s.Mbits - 62, 0)
        y = shr(shr(acc, drop) * s.M, 28 - sh - drop); return y if s.b is None else y + s.b
class VEmb:
    def __init__(s, emb, rng):
        s.keys = []; s.D = emb.shape[1]; s.x0 = None
        for p in (P1, P2):
            r = rng.integers(0, p, s.D, dtype=np.int64); de = np.empty(emb.shape[0], np.int64)
            for j in range(0, emb.shape[0], 16384):
                X = np.rint(np.asarray(emb[j:j + 16384], dtype=np.float64) * ONE).astype(np.int64); de[j:j + 16384] = (((X % p) * r) % p).sum(-1) % p
            s.keys.append((p, r, de))
        s.V = emb.shape[0]
    def __getitem__(s, ids):
        ids = np.asarray(ids); x0 = s.x0
        ok = isinstance(x0, np.ndarray) and x0.shape == (len(ids), s.D) and int(np.abs(x0).max()) < (1 << 40)
        if ok:
            for p, r, de in s.keys: ok = ok and bool(((((x0 % p) * r) % p).sum(-1) % p == de[ids]).all())
        if not ok: VER["fail"].append("embedding"); raise ValueError("reject embedding")
        return x0.astype(np.float64) / ONE
def verify(vm, ids, x0, trace):
    VER.update(trace=trace, i=0, fail=[]); vm.emb.x0 = x0
    try:
        lg = vm(list(ids)); return (VER["i"] == len(trace)), lg, None
    except Exception as e:
        return False, None, (VER["fail"][:1] or [repr(e)[:80]])[0]
m = ExactBitNet(); P("exact model built", round(time.time() - t0))
rng = np.random.default_rng(2026); tk = time.time()
vm = ExactBitNet.__new__(ExactBitNet); vm.cos, vm.sin, vm.gf = m.cos, m.sin, m.gf; vm.layers = []; kb = 0
for i, L in enumerate(m.layers):
    d = {k: L[k] for k in ("g1", "g2", "gs_a", "gs_f")}
    for nm in ("q", "k", "v", "o", "gate", "up", "down"): d[nm] = VBit(L[nm], rng, f"L{i}.{nm}"); kb += d[nm].kbytes()
    kb += 8 * sum(len(L[k]) for k in ("g1", "g2", "gs_a", "gs_f")); vm.layers.append(d)
    if i % 10 == 0: P("  keyed layer", i, round(time.time() - tk))
vm.head = VQ(m.head, rng, "head"); kb += vm.head.kbytes(); vm.emb = VEmb(m.emb, rng); kb += 16 * vm.emb.V
tern = sum(L[n].w.size for L in m.layers for n in ("q", "k", "v", "o", "gate", "up", "down"))
R["sizes"] = dict(verifier_key_MB=round(kb / 1e6, 2), ternary_params=int(tern), packed_2bit_MB=round(tern / 4 / 1e6, 1), int8_head_MB=round(m.head.w.size / 1e6, 1),
                  key_build_s=round(time.time() - tk, 1))
save(); P("sizes", R["sizes"])
ids_all = ids
b1 = dict(ok=0, equal=0, prover_s=[], verifier_s=[])
for wi in range(2):
    w = ids_all[4000 + wi * 64: 4000 + (wi + 1) * 64]
    a = time.time(); lg, tr, x0 = prove(m, w); b1["prover_s"].append(round(time.time() - a, 1))
    a = time.time(); ok, vlg, why = verify(vm, w, x0, tr); b1["verifier_s"].append(round(time.time() - a, 1))
    b1["ok"] += bool(ok); b1["equal"] += bool(ok and h(vlg) == h(lg) and h(lg) == h(m(w))); P("B1", wi, ok, why)
b1["pass"] = b1["ok"] == 2 and b1["equal"] == 2; R["B1_honest"] = b1; save()
w = ids_all[4000:4064]; lg, tr, x0 = prove(m, w); rr = np.random.default_rng(5); rej = 0
for _ in range(20):
    i = int(rr.integers(len(tr))); t2 = list(tr); a_ = tr[i].copy(); a_[int(rr.integers(a_.shape[0])), int(rr.integers(a_.shape[1]))] += int(rr.choice([-1, 1])); t2[i] = a_
    ok, _, _ = verify(vm, w, x0, t2); rej += (not ok)
R["B2_forged"] = {"rejected": f"{rej}/20", "pass": rej == 20}; save(); P("B2", R["B2_forged"])
_iso = isoftmax; Hh = H; rej = 0; chg = 0
for trial in range(5):
    target = int(rr.integers(len(m.layers) * Hh)); cnt = {"n": 0}
    def fake_iso(sc, valid, target=target, cnt=cnt):
        p = _iso(sc, valid)
        if cnt["n"] == target:
            p = p.copy(); T_ = p.shape[0]
            for row in range(max(1, T_ // 2), T_):
                a0, a1 = int(p[row, 0]), int(p[row, row]); p[row, 0], p[row, row] = a1, a0
        cnt["n"] += 1; return p
    globals()["isoftmax"] = fake_iso
    try: lgf, trf, x0f = prove(m, w)
    finally: globals()["isoftmax"] = _iso
    ok, _, why = verify(vm, w, x0f, trf); rej += (not ok); chg += bool(h(lgf) != h(lg))
R["B3_fake_attention"] = {"rejected": f"{rej}/5", "output_changed": chg, "pass": rej == 5}; save(); P("B3", R["B3_fake_attention"])
def proto_seed(prompt, nonce): return int.from_bytes(hashlib.sha256(np.asarray(prompt, np.int64).tobytes() + nonce).digest()[:8], "big")
def gen(prompt, n, seed, temp, choose=None):
    ids_ = list(prompt); out = []; tnum = temp_num(temp)
    for t in range(n):
        lgr = m(ids_)[-1]; tk_, _ = sample_int(lgr, seed, t, tnum)
        if choose: tk_ = choose(t, tk_, lgr)
        out.append(int(tk_)); ids_.append(int(tk_))
    return out
def verify_gen(prompt, toks, nonce, temp):
    lgp, trp, x0p = prove(m, list(prompt) + toks); seed = proto_seed(prompt, nonce); tnum = temp_num(temp)
    ok, vlg, why = verify(vm, list(prompt) + toks, x0p, trp)
    if not ok: return False, f"trace:{why}"
    for t, tk_ in enumerate(toks):
        if sample_int(vlg[len(prompt) - 1 + t], seed, t, tnum)[0] != tk_: return False, f"token@{t}"
    return True, None
prompt = tok("The printing press changed Europe because")["input_ids"]; nonce = b"verifier-nonce-0001"; seed = proto_seed(prompt, nonce); NG = 12; TEMP = 0.8
toks = gen(prompt, NG, seed, TEMP); ok, why = verify_gen(prompt, toks, nonce, TEMP); b4 = dict(honest_ok=bool(ok), text=tok.decode(toks)); rej = 0
for trial in range(5):
    pay = np.random.default_rng(300 + trial).integers(0, 2, NG); pay[0] = 1
    def choose(t, tk_, lgr, pay=pay):
        if pay[t] == 0: return tk_
        alt = int(lgr.argmax()); return alt if alt != tk_ else int(np.argsort(lgr)[-2])
    ok, why = verify_gen(prompt, gen(prompt, NG, seed, TEMP, choose), nonce, TEMP); rej += (not ok); P("B4 sub", trial, ok, why)
b4["substitution_rejected"] = f"{rej}/5"; b4["pass"] = b4["honest_ok"] and rej == 5; R["B4_zero_leak_generation"] = b4
R["verdict"] = "PASS" if all(R[k]["pass"] for k in ("B1_honest", "B2_forged", "B3_fake_attention", "B4_zero_leak_generation")) else "FAIL"
R["seconds"] = round(time.time() - t0); P(json.dumps(R, indent=1)); save()
