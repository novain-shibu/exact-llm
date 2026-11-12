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


# ======================= MODEL IDENTITY WITHOUT DISCLOSURE (2026-11-12), Qwen2.5-0.5B-Instruct, Kaggle CPU =======================
# Governance need (MIRI TGT, "Mechanisms to verify international agreements about AI development", 2026): verify that the model deployed is the
# model that was evaluated / declared, e.g. catch a different model swapped in. Here: the verifier is keyed (weightless Freivalds key) on the
# DECLARED model = Qwen2.5-0.5B-Instruct (Apache-2.0). The provider then serves:
#  I1 the declared model (honest)                               -> must verify (2 windows), logits identical
#  I2 a different model of the same shape (Qwen2.5-0.5B base)   -> must be rejected
#  I3 the declared model with ONE int8 weight changed by +-1 (one random matrix, random position; 10 trials) -> must be rejected every time
#  I4 the declared model with one norm-gain entry changed by ~1% of its value (10 trials)                   -> must be rejected every time
# Report: key MB vs weights MB, which check fired. PASS = I1-I4.
MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
get, c, tok = open_hf(MODEL); c["T"] = 512; ids_all = tok(TEXT)["input_ids"]; R["declared_model"] = MODEL
m = IntModel(get, c); P("declared model built", round(time.time() - t0))
vm, key_bytes, w_bytes = make_verifier(m)
for d_ in vm.layers: d_['g1'] = d_['g1'].copy(); d_['g2'] = d_['g2'].copy()          # verifier holds its OWN copies (not the provider's arrays)
vm.gf = vm.gf.copy()
R["sizes"] = dict(verifier_key_MB=round(key_bytes / 1e6, 2), int8_weight_MB=round((w_bytes - (m.head.w.size if c["tie"] else 0)) / 1e6, 1)); save()
W1 = ids_all[2000:2064]; W2 = ids_all[4000:4064]
i1 = []
for w in (W1, W2):
    lg, tr, x0 = prover_forward(m, w); ok, vlg, why = verify_forward(vm, w, x0, tr); i1.append(bool(ok and h(vlg) == h(lg)))
R["I1_declared_model_verifies"] = {"windows_ok": f"{sum(i1)}/2", "pass": all(i1)}; save(); P("I1", R["I1_declared_model_verifies"])
getb, cb, _ = open_hf("Qwen/Qwen2.5-0.5B"); cb["T"] = 512; mb = IntModel(getb, cb)
lg, tr, x0 = prover_forward(mb, W1); ok, _, why = verify_forward(vm, W1, x0, tr)
R["I2_other_model_rejected"] = {"rejected": (not ok), "where": str(why), "pass": (not ok)}; del mb; gc.collect(); save(); P("I2", R["I2_other_model_rejected"])
rng = np.random.default_rng(42); rej = 0; where = []
names = [(i, k) for i in range(len(m.layers)) for k in ("q", "k", "v", "o", "gate", "up", "down")]
for trial in range(10):
    li, k = names[int(rng.integers(len(names)))]; q = m.layers[li][k]; K_, N_ = q.w.shape
    a, b = int(rng.integers(K_)), int(rng.integers(N_)); old = int(q.w[a, b]); new = old + 1 if old < 127 else old - 1
    q.w[a, b] = new
    try:
        lg, tr, x0 = prover_forward(m, W1); ok, _, why = verify_forward(vm, W1, x0, tr)
    finally:
        q.w[a, b] = old
    rej += (not ok); where.append(f"L{li}.{k}[{a},{b}] {old}->{new}: {'rejected at ' + str(why) if not ok else 'PASSED'}")
R["I3_one_weight_changed"] = {"rejected": f"{rej}/10", "trials": where, "of_total_weights": int(sum(m.layers[i][k].w.size for i, k in names)), "pass": rej == 10}
save(); P("I3", R["I3_one_weight_changed"]["rejected"])
rej = 0; where = []
for trial in range(10):
    li = int(rng.integers(len(m.layers))); gk = ("g1", "g2")[trial % 2]; j = int(rng.integers(len(m.layers[li][gk])))
    dlt = max(1, abs(int(m.layers[li][gk][j])) // 100); m.layers[li][gk][j] += dlt
    try:
        lg, tr, x0 = prover_forward(m, W1); ok, _, why = verify_forward(vm, W1, x0, tr)
    finally:
        m.layers[li][gk][j] -= dlt
    rej += (not ok); where.append(f"L{li}.{gk}[{j}] +{dlt}/65536: {'rejected at ' + str(why) if not ok else 'PASSED'}")
R["I4_one_norm_gain_changed"] = {"rejected": f"{rej}/10", "trials": where, "pass": rej == 10}
R["verdict"] = "PASS" if all(R[k]["pass"] for k in ("I1_declared_model_verifies", "I2_other_model_rejected", "I3_one_weight_changed", "I4_one_norm_gain_changed")) else "FAIL"
R["seconds"] = round(time.time() - t0); P(json.dumps(R, indent=1)); save()
