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

# ======================= exact_torch: the same integer program as torch tensor ops (spec v0.3) =======================
# Any device. Linear layers on CUDA use int8 tensor cores (torch._int_mm) with int16 activations split into two int8 halves;
# attention matmuls use float64 (exact: partial sums < 2^53). KV cache supported (spec v0.3 is causal).
import torch
TCFG = {"int8_tc": True}
_EXP_T = None
def _tables(dev):
    global _EXP_T, _SILU_T
    if _EXP_T is None or _EXP_T.device != dev:
        _EXP_T = torch.tensor(EXP_T, dtype=torch.int64, device=dev); _SILU_T = torch.tensor(SILU_T, dtype=torch.int64, device=dev)
    return _EXP_T, _SILU_T
def t_shr(a, n):
    if isinstance(n, int):
        if n <= 0: return a << (-n)
        return (a + (1 << (n - 1))) >> n
    pos = n.clamp(min=0); neg = (-n).clamp(min=0)
    r = torch.where(pos > 0, (a + (torch.ones_like(pos) << (pos - 1).clamp(min=0))) >> pos, a)
    return r << neg
def t_bitlen(m):
    if TCFG.get("fast"):                                   # exact for 0 <= m < 2^53 (guaranteed by the spec bounds)
        return torch.frexp(m.double())[1].long()
    out = torch.zeros_like(m); v = m.clone()
    for b in (32, 16, 8, 4, 2, 1):
        big = v >= (1 << b); out = out + big.long() * b; v = torch.where(big, v >> b, v)
    return out + (v > 0).long()
def t_qrow(x, bits=15):
    m = x.abs().amax(-1, keepdim=True); s = (t_bitlen(m) - bits).clamp(min=0)
    return t_shr(x, s).clamp(-(1 << bits) + 1, (1 << bits) - 1), s
def t_isqrt(n):
    r = torch.floor(torch.sqrt(n.double())).long()
    r = torch.where(r * r > n, r - 1, r); return torch.where((r + 1) * (r + 1) <= n, r + 1, r)
def t_irms(x, g):
    k = (t_bitlen(x.abs().amax(-1, keepdim=True)) - 24).clamp(min=0)
    xs = t_shr(x, k)
    ms = torch.div((xs * xs).sum(-1, keepdim=True), x.shape[-1], rounding_mode="floor") + 1
    r = t_isqrt(ms).clamp(min=1)
    y = torch.div(x << F, r << k, rounding_mode="floor")
    return t_shr(y * g, F)
def t_isoftmax(sc, valid):
    ET, _ = _tables(sc.device)
    mx = torch.where(valid, sc, torch.full_like(sc, -(1 << 60))).amax(-1, keepdim=True)
    d = torch.where(valid, mx - sc, torch.zeros_like(sc))
    i = (d >> EXP_STEP).clamp(max=ET.numel() - 2); fr = d & ((1 << EXP_STEP) - 1)
    e = ET[i] - t_shr((ET[i] - ET[i + 1]) * fr, EXP_STEP)
    e = torch.where(valid & (d < (40 << F)), e, torch.zeros_like(e))
    return torch.div(e << 20, e.sum(-1, keepdim=True), rounding_mode="floor")
def t_isilu(x):
    _, ST = _tables(x.device)
    t = x.clamp(-16 * ONE, 16 * ONE - 1) + 16 * ONE
    i = t >> SILU_STEP; fr = t & ((1 << SILU_STEP) - 1)
    y = ST[i] + t_shr((ST[i + 1] - ST[i]) * fr, SILU_STEP)
    return torch.where(x >= 16 * ONE, x, torch.where(x < -16 * ONE, torch.zeros_like(x), y))
def t_rot_half(x):
    a, b = x.chunk(2, -1); return torch.cat([-b, a], -1)

# ---- balanced base-256 digits: every integer = sum d_i 256^i with d_i in [-128,127] (= int8). Exact GEMM via int8 tensor cores. ----
def t_bal_digits(x, n):
    out = []
    for _ in range(n):
        d = ((x + 128) & 255) - 128
        out.append(d.to(torch.int8)); x = (x - d) >> 8
    return out
def _up32(n): return 32 if n <= 32 else ((n + 63) // 64) * 64
def t_imm8(a8, b8):
    if a8.device.type == "cuda": return torch._int_mm(a8, b8).long()
    return (a8.int() @ b8.int()).long()                       # CPU emulation (exact)
def t_bal_mm(A, B, na, nb):
    """exact int64 A (R,K) @ B (K,N) with |A| < ~2^(8na-1), |B| < ~2^(8nb-1); int32 partial sums need K < 131072."""
    R, K = A.shape; N = B.shape[1]
    Rp, Kp, Np = _up32(R), max(256, _up32(K)), max(128, _up32(N))      # keep inside the T4-proven int8 shape range (zeros add nothing)
    A = torch.nn.functional.pad(A, (0, Kp - K, 0, Rp - R)); B = torch.nn.functional.pad(B, (0, Np - N, 0, Kp - K))
    Ad = [d.contiguous() for d in t_bal_digits(A, na)]; Bd = [d.contiguous() for d in t_bal_digits(B, nb)]
    acc = None
    for i, a in enumerate(Ad):
        for j, b in enumerate(Bd):
            t = t_imm8(a, b) << (8 * (i + j)); acc = t if acc is None else acc + t
    return acc[:R, :N]
def t_mm_f64(a, b):
    return torch.round(a.double() @ b.double()).long()
class TLin:
    def __init__(s, q, dev):
        s.dev = dev; s.K, s.N = q.w.shape
        s.w8 = torch.from_numpy(np.ascontiguousarray(q.w)).to(dev)                     # int8 (K, N)
        s.colsum = s.w8.long().sum(0)
        s.M = torch.from_numpy(q.M).to(dev); s.Mbits = q.Mbits
        s.b = None if q.b is None else torch.from_numpy(q.b).to(dev)
        s.use_tc = dev.type == "cuda" and s.K % 32 == 0 and s.N % 32 == 0 and s.K >= 256 and s.N >= 128
    def _acc(s, xq):
        if TCFG.get("fast"):
            if s.use_tc and TCFG["int8_tc"]:
                R = xq.shape[0]; P = 32 if R <= 32 else ((R + 63) // 64) * 64      # T4 cuBLASLt int8 failed at 224 rows; 32/64-multiples proven
                if P != R: xq = torch.nn.functional.pad(xq, (0, 0, 0, P - R))
                if TCFG.get("abits", 15) <= 7:
                    return torch._int_mm(xq.to(torch.int8), s.w8).long()[:R]
                hi = (xq >> 8).to(torch.int8); lo = ((xq & 255) - 128).to(torch.int8)
                return ((torch._int_mm(hi, s.w8).long() << 8) + torch._int_mm(lo, s.w8).long() + 128 * s.colsum)[:R]
            return t_mm_f64(xq, s.w8)
        if s.dev.type == "cuda" and TCFG["int8_tc"] and s.K % 32 == 0 and s.N % 32 == 0 and s.K >= 256 and s.N >= 128:
            R = xq.shape[0]; P = 32 if R <= 32 else ((R + 63) // 64) * 64      # T4 cuBLASLt int8 failed at 224 rows; 32/64-multiples proven
            if P != R: xq = torch.cat([xq, torch.zeros(P - R, s.K, dtype=xq.dtype, device=xq.device)])
            try:
                if TCFG.get("abits", 15) <= 7:                                             # A8: ONE int8 tensor-core pass
                    acc = torch._int_mm(xq.to(torch.int8).contiguous(), s.w8).long()
                else:                                                                      # A16: two int8 passes (hi/lo)
                    hi = (xq >> 8).to(torch.int8).contiguous(); lo = ((xq & 255) - 128).to(torch.int8).contiguous()
                    acc = (torch._int_mm(hi, s.w8).long() << 8) + torch._int_mm(lo, s.w8).long() + 128 * s.colsum
                TCFG["tc_calls"] = TCFG.get("tc_calls", 0) + 1
                return acc[:R]
            except RuntimeError as e:
                TCFG["tc_fallbacks"] = TCFG.get("tc_fallbacks", 0) + 1; TCFG["tc_err"] = str(e)[:120]
                xq = xq[:R]
        TCFG["f64_calls"] = TCFG.get("f64_calls", 0) + 1
        return t_mm_f64(xq, s.w8)
    def __call__(s, x):
        xq, sft = t_qrow(x, TCFG.get("abits", 15))
        acc = s._acc(xq)
        drop = (t_bitlen(acc.abs().amax(-1, keepdim=True)) + s.Mbits - 62).clamp(min=0)
        y = t_shr(t_shr(acc, drop) * s.M, 28 - sft - drop)
        return y if s.b is None else y + s.b
class TModel:
    """Torch twin of IntModel (numpy). forward(ids, cache) returns int64 Q16 logits for the new tokens."""
    def __init__(s, m, dev):
        s.m = m; s.c = m.c; s.dev = dev
        s.emb = m.emb                                                                      # float32 rows, converted per call
        s.layers = []
        for L in m.layers:
            s.layers.append({k: (TLin(v, dev) if isinstance(v, QLin) else torch.from_numpy(v).to(dev)) for k, v in L.items()})
        s.gf = torch.from_numpy(m.gf).to(dev); s.head = TLin(m.head, dev)
        s.cos = torch.from_numpy(m.cos).to(dev); s.sin = torch.from_numpy(m.sin).to(dev)
        s.INV = int(round((1 << 16) / math.sqrt(s.c["hd"])))
        s._core = (lambda *a: TModel._layer(s, *a))
    def enable_compile(s):
        s._core = torch.compile(lambda *a: TModel._layer(s, *a), dynamic=True)
    def new_cache(s): return [dict(kq=None, sk=None, v=None) for _ in s.layers] + [dict(n=0)]
    def _layer(s, x, L, cos, sin, prev, p0):
        c = s.c; H, KV, hd = c["H"], c["KV"], c["hd"]; T = x.shape[0]; rep = H // KV
        a = t_irms(x, L["g1"])
        q = L["q"](a).view(T, H, hd).transpose(0, 1); k = L["k"](a).view(T, KV, hd).transpose(0, 1)
        v = L["v"](a).view(T, KV, hd).transpose(0, 1)
        q = t_shr(q * cos + t_rot_half(q) * sin, F); k = t_shr(k * cos + t_rot_half(k) * sin, F)
        kq, sk = t_qrow(k)
        if prev is not None:
            kq = torch.cat([prev[0], kq], 1); sk = torch.cat([prev[1], sk], 1); v = torch.cat([prev[2], v], 1)
        S = kq.shape[1]
        qq, sq = t_qrow(q)
        skH = sk.repeat_interleave(rep, 0)
        valid = (torch.arange(S, device=x.device)[None, :] <= (p0 + torch.arange(T, device=x.device))[:, None])[None]
        if TCFG.get("att_tc"):                                     # exact attention on int8 tensor cores (no float64, no length cap)
            acc = torch.cat([t_bal_mm(qq[j * rep:(j + 1) * rep].reshape(rep * T, hd), kq[j].transpose(0, 1).contiguous(), 3, 3).view(rep, T, S)
                             for j in range(KV)], 0)
            sc = t_shr(acc * s.INV, (F - sq - skH.transpose(1, 2)) + 16)
            p = t_isoftmax(sc, valid)
            out = t_shr(torch.cat([t_bal_mm(p[j * rep:(j + 1) * rep].reshape(rep * T, S), v[j], 3, 4).view(rep, T, hd)
                                   for j in range(KV)], 0), 20)
        else:
            kqH = kq.repeat_interleave(rep, 0); vH = v.repeat_interleave(rep, 0)
            acc = torch.round(qq.double() @ kqH.double().transpose(1, 2)).long()
            sc = t_shr(acc * s.INV, (F - sq - skH.transpose(1, 2)) + 16)
            p = t_isoftmax(sc, valid)
            out = t_shr(torch.round(p.double() @ vH.double()).long(), 20)
        x = x + L["o"](out.transpose(0, 1).reshape(T, H * hd))
        mm_ = t_irms(x, L["g2"])
        x = x + L["down"](t_shr(t_isilu(L["gate"](mm_)) * L["up"](mm_), F))
        return x, kq, sk, v
    def forward(s, ids, cache=None, last_only=False):
        c = s.c; H, KV, hd = c["H"], c["KV"], c["hd"]; T = len(ids); rep = H // KV
        p0 = cache[-1]["n"] if cache is not None else 0
        x = torch.from_numpy(np.rint(np.asarray(s.emb[ids], dtype=np.float64) * ONE).astype(np.int64)).to(s.dev)
        cos, sin = s.cos[p0:p0 + T], s.sin[p0:p0 + T]
        vmax = None
        for li, L in enumerate(s.layers):
            C = cache[li] if cache is not None else None
            prev = (C["kq"], C["sk"], C["v"]) if (C is not None and C["kq"] is not None) else None
            if TCFG.get("fast"):
                x, kq, sk, v = s._core(x, L, cos, sin, prev, p0)
                vm = v.abs().amax(); vmax = vm if vmax is None else torch.maximum(vmax, vm)
            else:
                x, kq, sk, v = TModel._layer(s, x, L, cos, sin, prev, p0)
                assert v.abs().max().item() < (1 << 23) and kq.shape[1] <= 512, "spec v0.3 bound"
            if C is not None: C["kq"], C["sk"], C["v"] = kq, sk, v
        if vmax is not None and not TCFG.get("att_tc"):
            assert vmax.item() < (1 << 23) and kq.shape[1] <= 512, "fp64 attention bound"
        if vmax is not None and TCFG.get("att_tc"):
            assert vmax.item() < (1 << 30), "4-digit V bound (|V| < 2^30)"
        if cache is not None: cache[-1]["n"] = p0 + T
        if last_only: x = x[-1:]
        return s.head(t_irms(x, s.gf))

# ======================= CUDA-graph decode step (static KV cache, position as a tensor) =======================
# Bit-identical to TModel.forward with the dynamic cache: unused cache slots hold zeros and are masked out of the softmax
# (their probability is exactly 0), so every sum is the same integer.
class GraphDecoder:
    def __init__(s, tm, Smax=512):
        s.tm = tm; c = tm.c; s.c = c; s.Smax = Smax; dev = tm.dev
        H, KV, hd, D = c["H"], c["KV"], c["hd"], c["D"]
        s.x_in = torch.zeros((1, D), dtype=torch.int64, device=dev)
        s.pos = torch.zeros((), dtype=torch.int64, device=dev)
        s.kq = [torch.zeros((KV, Smax, hd), dtype=torch.int64, device=dev) for _ in tm.layers]
        s.sk = [torch.zeros((KV, Smax, 1), dtype=torch.int64, device=dev) for _ in tm.layers]
        s.vv = [torch.zeros((KV, Smax, hd), dtype=torch.int64, device=dev) for _ in tm.layers]
        s.ar = torch.arange(Smax, device=dev); s.graph = None; s.out = None
    def load_prefix(s, cache):
        """copy a TModel dynamic cache (after prefill) into the static buffers."""
        n = cache[-1]["n"]
        for li in range(len(s.tm.layers)):
            C = cache[li]; s.kq[li].zero_(); s.sk[li].zero_(); s.vv[li].zero_()
            s.kq[li][:, :n] = C["kq"]; s.sk[li][:, :n] = C["sk"]; s.vv[li][:, :n] = C["v"]
        s.pos.fill_(n)
    def _step(s):
        tm = s.tm; c = s.c; H, KV, hd = c["H"], c["KV"], c["hd"]; rep = H // KV
        x = s.x_in
        cos = tm.cos.index_select(0, s.pos.view(1)); sin = tm.sin.index_select(0, s.pos.view(1))
        valid = (s.ar <= s.pos).view(1, 1, s.Smax)
        for li, L in enumerate(tm.layers):
            a = t_irms(x, L["g1"])
            q = L["q"](a).view(1, H, hd).transpose(0, 1); k = L["k"](a).view(1, KV, hd).transpose(0, 1)
            v = L["v"](a).view(1, KV, hd).transpose(0, 1)
            q = t_shr(q * cos + t_rot_half(q) * sin, F); k = t_shr(k * cos + t_rot_half(k) * sin, F)
            kq1, sk1 = t_qrow(k)
            s.kq[li].index_copy_(1, s.pos.view(1), kq1); s.sk[li].index_copy_(1, s.pos.view(1), sk1); s.vv[li].index_copy_(1, s.pos.view(1), v)
            qq, sq = t_qrow(q)
            kqH = s.kq[li].repeat_interleave(rep, 0); skH = s.sk[li].repeat_interleave(rep, 0); vH = s.vv[li].repeat_interleave(rep, 0)
            acc = torch.round(qq.double() @ kqH.double().transpose(1, 2)).long()
            sc = t_shr(acc * tm.INV, (F - sq - skH.transpose(1, 2)) + 16)
            p = t_isoftmax(sc, valid)
            out = t_shr(torch.round(p.double() @ vH.double()).long(), 20)
            x = x + L["o"](out.transpose(0, 1).reshape(1, H * hd))
            mm_ = t_irms(x, L["g2"])
            x = x + L["down"](t_shr(t_isilu(L["gate"](mm_)) * L["up"](mm_), F))
        return tm.head(t_irms(x, tm.gf))
    def capture(s):
        st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
        saved = [(k.clone(), sk.clone(), v.clone()) for k, sk, v in zip(s.kq, s.sk, s.vv)]; p0 = s.pos.clone()
        with torch.cuda.stream(st):
            for _ in range(2): s._step()                            # warm-up (writes slot pos; restored below)
        torch.cuda.current_stream().wait_stream(st)
        for (k, sk, v), b in zip(saved, zip(s.kq, s.sk, s.vv)): b[0].copy_(k); b[1].copy_(sk); b[2].copy_(v)
        s.pos.copy_(p0)
        s.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(s.graph): s.out = s._step()
    def step(s, token):
        """one decode step: write token embedding, replay, advance position. returns int64 Q16 logits (1, V)."""
        e = torch.from_numpy(np.rint(np.asarray(s.tm.emb[[token]], dtype=np.float64) * ONE).astype(np.int64))
        s.x_in.copy_(e, non_blocking=True)
        if s.graph is None: out = s._step()
        else: s.graph.replay(); out = s.out
        s.pos.add_(1); return out

# PRE-REGISTERED (2026-11-11):
#  G1 GEMV kernel == exact CPU int64 reference on random inputs (all real layer shapes).
#  G2 group-128 model: torch prefill (bmm path) hash == numpy group-128 reference (64 tokens), and GEMV decode logits == full-forward rows
#     for 16 decode steps (cross-path exactness).
#  ROUND 2 (CUDA graphs): G4 graph-replayed decode logits == dynamic-cache decode logits for 16 steps (exact).
#  G5 decode tok/s, exact (GEMV + CUDA graph) vs HF fp16 BOTH eager and with static cache + torch.compile(reduce-overhead) (fair).
#     Claim "faster than fp16" only if exact > the FASTER of the two fp16 numbers. Kill route after round 3 if < 1.0x of eager fp16.
OUT = "/kaggle/working/results.json"; R = {}; t0 = time.time(); dev = torch.device("cuda")
from torch.utils.cpp_extension import load_inline
CUDA_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
// one warp per output column; 16 warps per block share the activation vector x (int16) in shared memory.
// each lane handles 8 consecutive k per iteration (8-byte weight load); 16 lanes cover one 128-wide group,
// so a warp iteration covers 2 groups: half-warp integer reduction, then int32 sum x group scale -> int64. Exact (integer).
__global__ void gemv_g128_v2(const int16_t* __restrict__ x, const int8_t* __restrict__ wT, const int64_t* __restrict__ Mg,
                             int64_t* __restrict__ out, int K, int N, int G) {
  extern __shared__ int16_t xs[];
  for (int i = threadIdx.x; i < K; i += blockDim.x) xs[i] = x[i];
  __syncthreads();
  int warp_in_block = threadIdx.x >> 5; int lane = threadIdx.x & 31;
  int n = blockIdx.x * (blockDim.x >> 5) + warp_in_block;
  if (n >= N) return;
  const int8_t* wrow = wT + (int64_t)n * K; int64_t acc = 0;
  for (int k0 = 0; k0 < K; k0 += 256) {
    int k = k0 + lane * 8; int s = 0;
    if (k < K) {
      int2 wv = *reinterpret_cast<const int2*>(wrow + k);
      const int8_t* wb = reinterpret_cast<const int8_t*>(&wv);
      #pragma unroll
      for (int j = 0; j < 8; ++j) s += (int)xs[k + j] * (int)wb[j];
    }
    #pragma unroll
    for (int o = 8; o > 0; o >>= 1) s += __shfl_down_sync(0xffffffff, s, o, 16);
    if ((lane & 15) == 0 && k < K) acc += (int64_t)s * Mg[(int64_t)n * G + (k >> 7)];
  }
  acc += __shfl_down_sync(0xffffffff, acc, 16);
  if (lane == 0) out[n] = acc;
}
torch::Tensor gemv(torch::Tensor x, torch::Tensor wT, torch::Tensor Mg) {
  int N = wT.size(0), K = wT.size(1), G = K / 128;
  auto out = torch::empty({N}, torch::dtype(torch::kInt64).device(x.device()));
  int threads = 512; int warps = threads / 32; int blocks = (N + warps - 1) / warps;
  gemv_g128_v2<<<blocks, threads, K * sizeof(int16_t), at::cuda::getCurrentCUDAStream()>>>(x.data_ptr<int16_t>(), wT.data_ptr<int8_t>(),
      Mg.data_ptr<int64_t>(), out.data_ptr<int64_t>(), K, N, G);
  return out;
}
"""
ext = load_inline(name="exact_gemv", cpp_sources="torch::Tensor gemv(torch::Tensor x, torch::Tensor wT, torch::Tensor Mg);",
                  cuda_sources=CUDA_SRC, functions=["gemv"], verbose=False)
P("compiled kernel", round(time.time() - t0))
@torch.library.custom_op("exact::gemv", mutates_args=())
def gemv(x16: torch.Tensor, wT: torch.Tensor, Mg: torch.Tensor) -> torch.Tensor:
    return ext.gemv(x16, wT, Mg)
@gemv.register_fake
def _gemv_fake(x16, wT, Mg): return torch.empty((wT.shape[0],), dtype=torch.int64, device=x16.device)
# ---- G1 unit test
g = torch.Generator().manual_seed(0); ok = True
for K_, N_ in ((896, 896), (896, 4864), (4864, 896), (1536, 8960), (8960, 1536), (896, 151936)):
    x = torch.randint(-32767, 32768, (K_,), generator=g, dtype=torch.int64); w = torch.randint(-127, 128, (K_, N_), generator=g, dtype=torch.int64)
    Mg = torch.randint(1, 1 << 27, (N_, K_ // 128), generator=g, dtype=torch.int64)
    ref = ((x.view(K_ // 128, 128, 1) * w.view(K_ // 128, 128, N_)).sum(1) * Mg.T).sum(0)
    got = gemv(x.to(torch.int16).to(dev), w.to(torch.int8).T.contiguous().to(dev), Mg.to(dev)).cpu()
    ok &= bool(torch.equal(got, ref))
R["G1_gemv_exact"] = ok; P("G1", ok)
# ---- group-128 linear (numpy reference + torch twin)
G_ = 128
class GQLin:
    def __init__(self, w_out_in, bias=None):
        w = np.asarray(w_out_in, dtype=np.float64).T; K, N = w.shape; self.groups = [(a, a + G_) for a in range(0, K, G_)]
        self.w = np.empty((K, N), np.int8); self.M = np.empty((len(self.groups), N), np.int64)
        for gi, (a, b) in enumerate(self.groups):
            sc = np.abs(w[a:b]).max(0); sc = np.where(sc == 0, 1e-12, sc) / 127.0
            self.w[a:b] = np.clip(np.rint(w[a:b] / sc), -127, 127).astype(np.int8); self.M[gi] = np.rint(sc * (1 << 28)).astype(np.int64)
        self.b = None if bias is None else np.rint(np.asarray(bias, dtype=np.float64) * ONE).astype(np.int64)
    def __call__(self, x):
        xq, s = qrow(x, 15); tot = np.zeros((x.shape[0], self.w.shape[1]), np.int64)
        for gi, (a, b) in enumerate(self.groups): tot += exact_mm(xq[:, a:b], self.w[a:b]) * self.M[gi]
        y = shr(tot, 28 - s); return y if self.b is None else y + self.b
class TLinG:
    """torch twin: R==1 -> custom exact GEMV (one weight read); R>1 -> exact fp64 batched group GEMMs (sums < 2^53)."""
    def __init__(s_, q, mode):
        s_.mode = mode; w = torch.from_numpy(np.ascontiguousarray(q.w)); s_.K, s_.N = w.shape; s_.G = s_.K // 128
        s_.wT = w.T.contiguous().to(dev)
        if mode == "group": s_.Mg = torch.from_numpy(np.ascontiguousarray(q.M.T)).to(dev)            # (N, G)
        else:
            s_.Mg = torch.ones((s_.N, s_.G), dtype=torch.int64, device=dev); s_.M = torch.from_numpy(q.M).to(dev); s_.Mbits = q.Mbits
        s_.b = None if q.b is None else torch.from_numpy(q.b).to(dev)
    def __call__(s_, x):
        xq, sft = t_qrow(x, 15); R_ = xq.shape[0]
        if R_ == 1:
            raw = gemv(xq[0].to(torch.int16).contiguous(), s_.wT, s_.Mg)[None]
        elif s_.K * s_.N > 30_000_000:                                    # big layers: exact GEMV per row (no float64 weight copy)
            raw = torch.stack([gemv(xq[i].to(torch.int16).contiguous(), s_.wT, s_.Mg) for i in range(R_)])
        else:
            Xg = xq.view(R_, s_.G, 128).permute(1, 0, 2).double(); Wg = s_.wT.T.reshape(s_.G, 128, s_.N).double()
            accg = torch.round(torch.bmm(Xg, Wg)).long()                                                 # (G, R, N)
            raw = (accg * s_.Mg.T[:, None, :]).sum(0)
        if s_.mode == "group": y = t_shr(raw, 28 - sft)
        else:
            drop = (t_bitlen(raw.abs().amax(-1, keepdim=True)) + s_.Mbits - 62).clamp(min=0)
            y = t_shr(t_shr(raw, drop) * s_.M, 28 - sft - drop)
        return y if s_.b is None else y + s_.b
KEYS = (("q", "self_attn.q_proj", "self_attn.q_proj.bias"), ("k", "self_attn.k_proj", "self_attn.k_proj.bias"),
        ("v", "self_attn.v_proj", "self_attn.v_proj.bias"), ("o", "self_attn.o_proj", None),
        ("gate", "mlp.gate_proj", None), ("up", "mlp.up_proj", None), ("down", "mlp.down_proj", None))
from transformers import AutoModelForCausalLM
try:
    from datasets import load_dataset
    TEXT = (chr(10) * 2).join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
except Exception:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    TEXT = (chr(10) * 2).join(pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"))["text"])
def timeit(fn, n=3):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize(); t1 = time.time(); fn(); torch.cuda.synchronize(); ts.append(time.time() - t1)
    return float(np.median(ts))


# ---------------- ROUND 6: exact 7B decode on a T4 (fp16 7B does not fit); 1.5B repeated as a control
# ---------------- (was ROUND 5: fair comparison - exact (compiled+fused+graph) vs fp16 (compiled+graph), same harness ----------------
# PRE-REGISTERED: claim "exact decode faster than fp16" at a size ONLY if exact_graph_compiled > fp16_graph_compiled at that size
# (both torch.compile + CUDA graph, batch 1, 32 steps after a 64-token prompt). Exactness: compiled exact graph logits == plain exact
# graph logits (8 steps). 7B: fp16 does not fit a T4 -> exact-only datapoint.
for name in ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"):
    tag = name.split("-")[-1]; r = {}
    get, c, tok = open_hf(name); c["T"] = 512; w = tok(TEXT)["input_ids"][:200]
    json.dump([int(t) for t in w[:96]], open("/kaggle/working/prompt_ids.json", "w"))
    m = IntModel(get, c)
    for i, L in enumerate(m.layers):
        p = PFX.format(i)
        for key, nm, bn in KEYS: L[key] = GQLin(get(p + nm + ".weight"), get(p + bn) if bn else None)
    TCFG.update(fast=True, abits=15, att_tc=False)
    tm = TModel.__new__(TModel); tm.m = m; tm.c = c; tm.dev = dev; tm.emb = m.emb
    tm.layers = [{k: (TLinG(v, "group") if isinstance(v, GQLin) else torch.from_numpy(v).to(dev)) for k, v in L.items()} for L in m.layers]
    tm.gf = torch.from_numpy(m.gf).to(dev); tm.head = TLinG(m.head, "channel")
    tm.cos = torch.from_numpy(m.cos).to(dev); tm.sin = torch.from_numpy(m.sin).to(dev)
    tm.INV = int(round((1 << 16) / math.sqrt(c["hd"]))); tm._core = (lambda *a: TModel._layer(tm, *a))
    if tag != "7B": r["prefill_eq_numpy"] = h(tm.forward(w[:64]).cpu().numpy()) == h(m(w[:64]))
    m.layers = None; gc.collect()
    def run_graph(gdx):
        c2 = tm.new_cache(); tm.forward(w[:64], c2); gdx.load_prefix(c2); torch.cuda.synchronize(); t1 = time.time()
        for t in w[64:96]: gdx.step(t)
        torch.cuda.synchronize(); return 32 / (time.time() - t1)
    try:
        gp = GraphDecoder(tm, Smax=512); c3 = tm.new_cache(); tm.forward(w[:64], c3); gp.load_prefix(c3); gp.capture()
        ref_logits = [gp.step(t).clone() for t in w[64:72]]
        r["exact_graph_plain_tok_s"] = round(float(np.median([run_graph(gp) for _ in range(3)])), 2); del gp
        gcmp = GraphDecoder(tm, Smax=512); gcmp._step = torch.compile(gcmp._step, fullgraph=False, dynamic=False)
        c4 = tm.new_cache(); tm.forward(w[:64], c4); gcmp.load_prefix(c4); t1 = time.time(); gcmp._step(); r["compile_s"] = round(time.time() - t1, 1)
        gcmp.load_prefix(c4); gcmp.capture()
        c5 = tm.new_cache(); tm.forward(w[:64], c5); gcmp.load_prefix(c5)
        r["compiled_eq_plain"] = all(torch.equal(gcmp.step(t), ref) for t, ref in zip(w[64:72], ref_logits))
        r["exact_graph_compiled_tok_s"] = round(float(np.median([run_graph(gcmp) for _ in range(3)])), 2); del gcmp
    except Exception as e:
        r["exact_error"] = str(e)[:300]
    r["exact_gpu_mem_GB"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    del tm; gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    R[tag] = r; json.dump(R, open(OUT, "w"), indent=1); P(tag, "exact", json.dumps(r))
    for COMPILE16 in ((True,) if tag != "7B" else ()):
        try:
            hf = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float16).to(dev).eval()
            # FAIR fp16 baseline: our own fp16 decode step (same structure, static cache) captured as one CUDA graph
            try:
                sd = {k: v for k, v in hf.state_dict().items()}; Hh, KVh, hdh = c["H"], c["KV"], c["hd"]; reph = Hh // KVh
                Ls = [{n: sd[f"model.layers.{i}.{n}"] for n in ("input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_proj.weight",
                       "self_attn.q_proj.bias", "self_attn.k_proj.weight", "self_attn.k_proj.bias", "self_attn.v_proj.weight", "self_attn.v_proj.bias",
                       "self_attn.o_proj.weight", "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight")} for i in range(c["NL"])]
                normw = sd["model.norm.weight"]; emb16 = sd["model.embed_tokens.weight"]; head16 = sd.get("lm_head.weight", emb16)
                Smax = 512; kc = [torch.zeros((KVh, Smax, hdh), dtype=torch.float16, device=dev) for _ in Ls]; vc = [torch.zeros_like(k_) for k_ in kc]
                inv = 1.0 / (c["theta"] ** (torch.arange(0, hdh, 2, device=dev, dtype=torch.float32) / hdh)); ang = torch.outer(torch.arange(Smax, device=dev, dtype=torch.float32), inv)
                cosT = torch.cat([ang, ang], -1).cos().half(); sinT = torch.cat([ang, ang], -1).sin().half()
                xin = torch.zeros((1, c["D"]), dtype=torch.float16, device=dev); posT = torch.zeros((), dtype=torch.int64, device=dev); arT = torch.arange(Smax, device=dev)
                def rms16(z, g_): zf = z.float(); return (zf * torch.rsqrt(zf.pow(2).mean(-1, keepdim=True) + c["eps"])).half() * g_
                def rh(z): a_, b_ = z.chunk(2, -1); return torch.cat([-b_, a_], -1)
                def step16():
                    x_ = xin; cs = cosT.index_select(0, posT.view(1)); sn = sinT.index_select(0, posT.view(1)); valid_ = (arT <= posT).view(1, 1, Smax)
                    for li, L_ in enumerate(Ls):
                        a_ = rms16(x_, L_["input_layernorm.weight"])
                        q_ = torch.nn.functional.linear(a_, L_["self_attn.q_proj.weight"], L_["self_attn.q_proj.bias"]).view(1, Hh, hdh).transpose(0, 1)
                        k_ = torch.nn.functional.linear(a_, L_["self_attn.k_proj.weight"], L_["self_attn.k_proj.bias"]).view(1, KVh, hdh).transpose(0, 1)
                        v_ = torch.nn.functional.linear(a_, L_["self_attn.v_proj.weight"], L_["self_attn.v_proj.bias"]).view(1, KVh, hdh).transpose(0, 1)
                        q_ = q_ * cs + rh(q_) * sn; k_ = k_ * cs + rh(k_) * sn
                        kc[li].index_copy_(1, posT.view(1), k_); vc[li].index_copy_(1, posT.view(1), v_)
                        kH = kc[li].repeat_interleave(reph, 0); vH = vc[li].repeat_interleave(reph, 0)
                        sc_ = (q_.float() @ kH.float().transpose(1, 2)) / math.sqrt(hdh)
                        sc_ = sc_.masked_fill(~valid_, float("-inf")).softmax(-1)
                        o_ = (sc_.half() @ vH).transpose(0, 1).reshape(1, Hh * hdh)
                        x_ = x_ + torch.nn.functional.linear(o_, L_["self_attn.o_proj.weight"])
                        m_ = rms16(x_, L_["post_attention_layernorm.weight"])
                        g_ = torch.nn.functional.linear(m_, L_["mlp.gate_proj.weight"]); u_ = torch.nn.functional.linear(m_, L_["mlp.up_proj.weight"])
                        x_ = x_ + torch.nn.functional.linear(torch.nn.functional.silu(g_) * u_, L_["mlp.down_proj.weight"])
                    return torch.nn.functional.linear(rms16(x_, normw), head16)
                if COMPILE16: step16 = torch.compile(step16, fullgraph=False, dynamic=False)
                st = torch.cuda.Stream(); st.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(st), torch.no_grad():
                    for _ in range(2): step16()
                torch.cuda.current_stream().wait_stream(st)
                g16 = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g16), torch.no_grad(): out16 = step16()
                def dec16():
                    posT.fill_(64); torch.cuda.synchronize(); t1 = time.time()
                    for t in w[64:96]:
                        xin.copy_(emb16[t:t + 1]); g16.replay(); posT.add_(1)
                    torch.cuda.synchronize(); return time.time() - t1
                dec16(); r["fp16_graph_compiled_tok_s" if COMPILE16 else "fp16_graph_plain_tok_s"] = round(32 / float(np.median([dec16() for _ in range(3)])), 2)
            except Exception as e:
                r["fp16_graph_error"] = str(e)[:240]


        except Exception as e:
            r["fp16_" + ("compiled" if COMPILE16 else "plain") + "_error"] = str(e)[:200]
        hf = None; gc.collect(); torch.cuda.empty_cache()
    if "fp16_graph_compiled_tok_s" in r and "exact_graph_compiled_tok_s" in r:
        r["exact_vs_fp16_compiled"] = round(r["exact_graph_compiled_tok_s"] / r["fp16_graph_compiled_tok_s"], 3)
    R[tag] = r; json.dump(R, open(OUT, "w"), indent=1); P(tag, json.dumps(r))
R["seconds"] = round(time.time() - t0); P(json.dumps(R, indent=1)); json.dump(R, open(OUT, "w"), indent=1)


# ======================= vLLM REALITY CHECK (2026-11-11) =======================
# PRE-REGISTERED: report exact_graph_compiled_tok_s / vLLM fp16 decode tok/s at batch 1 on this same T4 session, as measured.
# vLLM: dtype float16, CUDA graphs on (enforce_eager=False), max_model_len 512, same 64 prompt token ids, 32 new tokens (ignore_eos).
# Decode-only rate = 31 / (t(32 new tokens) - t(1 new token)), median of 5 (bitexact-llm's method). If exact/vLLM < 0.8,
# the README speed section states "vLLM fp16 is Nx faster" the same day. No pass/fail on the engine itself; this is a disclosure check.
import subprocess, sys
VB = r"""
import json, time, statistics, sys
R = {}
try:
    import vllm, torch
    from vllm import LLM, SamplingParams
    from vllm.inputs import TokensPrompt
    R["vllm_version"] = vllm.__version__; R["torch"] = torch.__version__
    ids = json.load(open("/kaggle/working/prompt_ids.json"))[:64]
    for name in ("Qwen/Qwen2.5-0.5B", "Qwen/Qwen2.5-1.5B"):
        tag = name.split("-")[-1]
        try:
            llm = LLM(model=name, dtype="float16", max_model_len=512, gpu_memory_utilization=0.5, enforce_eager=False, seed=0)
            def gen(n):
                sp = SamplingParams(max_tokens=n, min_tokens=n, temperature=0.0, ignore_eos=True)
                torch.cuda.synchronize(); t = time.time(); llm.generate([TokensPrompt(prompt_token_ids=ids)], sp, use_tqdm=False); torch.cuda.synchronize()
                return time.time() - t
            for _ in range(3): gen(32); gen(1)
            t32 = statistics.median([gen(32) for _ in range(5)]); t1 = statistics.median([gen(1) for _ in range(5)])
            R[tag] = dict(vllm_fp16_decode_tok_s=round(31 / (t32 - t1), 2), vllm_e2e_32_tok_s=round(32 / t32, 2), t32=round(t32, 4), t1=round(t1, 4))
            del llm; import gc; gc.collect(); torch.cuda.empty_cache()
        except Exception as e:
            R[tag] = dict(error=str(e)[-400:])
        print("VLLM", tag, R[tag], flush=True)
except Exception as e:
    R["error"] = str(e)[-400:]
json.dump(R, open("/kaggle/working/vllm_results.json", "w"), indent=1)
"""
open("/kaggle/working/vllm_bench.py", "w").write(VB); gc.collect(); torch.cuda.empty_cache(); P("free GB before vLLM", [round(x / 1e9, 2) for x in torch.cuda.mem_get_info()])
R["vllm"] = {}
try:
    pi = subprocess.run([sys.executable, "-m", "pip", "install", "-q", "vllm"], capture_output=True, text=True, timeout=1500)
    R["vllm_pip_rc"] = pi.returncode; R["vllm_pip_tail"] = (pi.stderr or "")[-600:]
    vb = subprocess.run([sys.executable, "/kaggle/working/vllm_bench.py"], capture_output=True, text=True, timeout=2400,
                        env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    P(vb.stdout[-3000:]); P(vb.stderr[-3000:])
    R["vllm"] = json.load(open("/kaggle/working/vllm_results.json"))
except Exception as e:
    R["vllm_error"] = str(e)[-400:]
for tag in ("0.5B", "1.5B"):
    ex_ = R.get(tag, {}).get("exact_graph_compiled_tok_s"); vl = R.get("vllm", {}).get(tag, {}).get("vllm_fp16_decode_tok_s")
    if ex_ and vl: R.setdefault("ratios", {})[tag] = dict(exact_vs_vllm=round(ex_ / vl, 3), ours_fp16_vs_vllm=round(R[tag].get("fp16_graph_compiled_tok_s", 0) / vl, 3))
R["seconds"] = round(time.time() - t0); P(json.dumps(R, indent=1)); json.dump(R, open(OUT, "w"), indent=1)
