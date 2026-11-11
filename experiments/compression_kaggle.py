"""exact_llm - bit-exact integer inference for Qwen2-family LLMs (spec v0.3, see SPEC.md).

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

PRE = os.environ.get("PREFLIGHT") == "1"
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
                assert np.abs(v[kh]).max() < (1 << 23) and v.shape[1] <= 512, 'spec v0.3 bound: |V| < 2^23 (Q16), T <= 512'
                out[hh] = shr(exact_mm(p, v[kh]), 20)                    # spec v0.3: V unquantised Q16 (causal)
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
        assert np.abs(v[kh]).max() < (1 << 23) and v.shape[1] <= 512, 'spec v0.3 bound: |V| < 2^23 (Q16), T <= 512'
        out[hh] = shr(exact_mm(p, v[kh]), 20)                    # spec v0.3: V unquantised Q16 (causal)
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
        assert np.abs(v[kh]).max() < (1 << 23) and v.shape[1] <= 512, 'spec v0.3 bound: |V| < 2^23 (Q16), T <= 512'
        out[hh] = shr(exact_mm(p, v[kh]), 20)                    # spec v0.3: V unquantised Q16 (causal)
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
    import torch
except ImportError:
    torch = None

# ======================= exact_torch: the same integer program as torch tensor ops (spec v0.3) =======================
# Any device. Linear layers on CUDA use int8 tensor cores (torch._int_mm) with int16 activations split into two int8 halves;
# attention matmuls use float64 (exact: partial sums < 2^53). KV cache supported (spec v0.3 is causal).
try:
    import torch
except ImportError:
    torch = None
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
TCFG = {"int8_tc": True}
def t_mm_f64(a, b):
    return torch.round(a.double() @ b.double()).long()
class TLin:
    def __init__(s, q, dev):
        s.dev = dev; s.K, s.N = q.w.shape
        s.w8 = torch.from_numpy(np.ascontiguousarray(q.w)).to(dev)                     # int8 (K, N)
        s.colsum = s.w8.long().sum(0)
        s.M = torch.from_numpy(q.M).to(dev); s.Mbits = q.Mbits
        s.b = None if q.b is None else torch.from_numpy(q.b).to(dev)
    def _acc(s, xq):
        if s.dev.type == "cuda" and TCFG["int8_tc"] and s.K % 8 == 0 and s.N % 8 == 0:
            R = xq.shape[0]; P = max(R, 32)
            if P != R: xq = torch.cat([xq, torch.zeros(P - R, s.K, dtype=xq.dtype, device=xq.device)])
            hi = (xq >> 8).to(torch.int8); lo = ((xq & 255) - 128).to(torch.int8)
            acc = (torch._int_mm(hi, s.w8).long() << 8) + torch._int_mm(lo, s.w8).long() + 128 * s.colsum
            return acc[:R]
        return t_mm_f64(xq, s.w8)
    def __call__(s, x):
        xq, sft = t_qrow(x, 15)
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
    def new_cache(s): return [dict(kq=None, sk=None, v=None) for _ in s.layers] + [dict(n=0)]
    def forward(s, ids, cache=None, last_only=False):
        c = s.c; H, KV, hd = c["H"], c["KV"], c["hd"]; T = len(ids); rep = H // KV
        p0 = cache[-1]["n"] if cache is not None else 0
        x = torch.from_numpy(np.rint(np.asarray(s.emb[ids], dtype=np.float64) * ONE).astype(np.int64)).to(s.dev)
        cos, sin = s.cos[p0:p0 + T], s.sin[p0:p0 + T]
        for li, L in enumerate(s.layers):
            a = t_irms(x, L["g1"])
            q = L["q"](a).view(T, H, hd).transpose(0, 1); k = L["k"](a).view(T, KV, hd).transpose(0, 1)
            v = L["v"](a).view(T, KV, hd).transpose(0, 1)
            q = t_shr(q * cos + t_rot_half(q) * sin, F); k = t_shr(k * cos + t_rot_half(k) * sin, F)
            kq, sk = t_qrow(k)                                                             # per key row: causal
            if cache is not None:
                C = cache[li]
                if C["kq"] is not None:
                    kq = torch.cat([C["kq"], kq], 1); sk = torch.cat([C["sk"], sk], 1); v = torch.cat([C["v"], v], 1)
                C["kq"], C["sk"], C["v"] = kq, sk, v
            S = kq.shape[1]
            assert v.abs().max().item() < (1 << 23) and S <= 512, "spec v0.3 bound"
            qq, sq = t_qrow(q)                                                             # (H, T, hd)
            kqH = kq.repeat_interleave(rep, 0); skH = sk.repeat_interleave(rep, 0); vH = v.repeat_interleave(rep, 0)
            acc = torch.round(qq.double() @ kqH.double().transpose(1, 2)).long()           # (H, T, S)
            sc = t_shr(acc * s.INV, (F - sq - skH.transpose(1, 2)) + 16)
            valid = (torch.arange(S, device=s.dev)[None, :] <= (p0 + torch.arange(T, device=s.dev))[:, None])[None]
            p = t_isoftmax(sc, valid)
            out = t_shr(torch.round(p.double() @ vH.double()).long(), 20)                  # (H, T, hd)
            x = x + L["o"](out.transpose(0, 1).reshape(T, H * hd))
            mm_ = t_irms(x, L["g2"])
            x = x + L["down"](t_shr(t_isilu(L["gate"](mm_)) * L["up"](mm_), F))
        if cache is not None: cache[-1]["n"] = p0 + T
        if last_only: x = x[-1:]
        return s.head(t_irms(x, s.gf))

# ======================= portable LLM compression (integer arithmetic coding) =======================
# PRE-REGISTERED (2026-11-11):
#  C1 round trip: text compressed with the numpy reference (one full forward) decompresses byte-exactly with the torch
#     KV-cache engine (a separate implementation, token by token) -> PASS required.
#  C2 ratio: report bits/byte vs gzip-9, bz2-9, xz-9 on the same bytes (no gate; expectation: well below all three).
#  C3 float control: the same coder fed HF fp32 probabilities quantised to the same integer frequencies; count positions where
#     full-forward (4 threads) vs incremental KV-cache (1 thread) frequencies differ. Any difference = the float codec cannot decode.
SB = 32; FULL = 1 << SB; HALF = FULL >> 1; QUART = HALF >> 1; MASK = FULL - 1
def int_freqs(lg_row):
    """integer frequencies from exact Q16 logits: integer softmax (Q20) + 1 so every token stays decodable."""
    f = isoftmax_row(lg_row) + 1
    return np.cumsum(f)                                   # inclusive cumulative; total = cum[-1] < 2^21
class Enc:
    def __init__(s): s.low, s.high, s.pend, s.bits = 0, MASK, 0, []
    def put(s, cum, sym):
        tot = int(cum[-1]); lo = int(cum[sym - 1]) if sym > 0 else 0; hi = int(cum[sym]); r = s.high - s.low + 1
        s.high = s.low + hi * r // tot - 1; s.low = s.low + lo * r // tot
        while ((s.low ^ s.high) & HALF) == 0:
            b = s.low >> (SB - 1); s.bits.append(b); s.bits.extend([b ^ 1] * s.pend); s.pend = 0
            s.low = (s.low << 1) & MASK; s.high = ((s.high << 1) & MASK) | 1
        while (s.low & ~s.high & QUART) != 0:
            s.pend += 1; s.low = (s.low << 1) ^ HALF; s.high = ((s.high ^ HALF) << 1) | HALF | 1
    def finish(s):
        s.bits.append(1); return s.bits
class Dec:
    def __init__(s, bits):
        s.bits, s.i = bits, 0; s.low, s.high, s.code = 0, MASK, 0
        for _ in range(SB): s.code = (s.code << 1) | s._rb()
    def _rb(s):
        b = s.bits[s.i] if s.i < len(s.bits) else 0; s.i += 1; return b
    def get(s, cum):
        tot = int(cum[-1]); r = s.high - s.low + 1; off = s.code - s.low
        val = ((off + 1) * tot - 1) // r
        sym = int(np.searchsorted(cum, val, side="right"))
        lo = int(cum[sym - 1]) if sym > 0 else 0; hi = int(cum[sym])
        s.high = s.low + hi * r // tot - 1; s.low = s.low + lo * r // tot
        while ((s.low ^ s.high) & HALF) == 0:
            s.low = (s.low << 1) & MASK; s.high = ((s.high << 1) & MASK) | 1; s.code = ((s.code << 1) & MASK) | s._rb()
        while (s.low & ~s.high & QUART) != 0:
            s.low = (s.low << 1) ^ HALF; s.high = ((s.high ^ HALF) << 1) | HALF | 1
            s.code = (s.code & HALF) | ((s.code << 1) & (MASK >> 1)) | s._rb()
        return sym
def uniform_cum(V): return np.arange(1, V + 1, dtype=np.int64)
def compress_block(m, ids):
    """one exact forward over the block (numpy reference); token 0 coded uniformly."""
    lg = m(ids); e = Enc(); e.put(uniform_cum(m.c["V"]), ids[0])
    for t in range(1, len(ids)): e.put(int_freqs(lg[t - 1]), ids[t])
    return e.finish()
def decompress_block(tm, bits, n):
    """token-by-token with the torch KV-cache engine (a different implementation)."""
    d = Dec(bits); ids = [d.get(uniform_cum(tm.c["V"]))]; cache = tm.new_cache()
    for t in range(1, n):
        lg = tm.forward([ids[-1]], cache)[0].cpu().numpy(); ids.append(d.get(int_freqs(lg)))
    return ids

# ---------------- driver ----------------
import gzip, bz2, lzma
R = {}; t0 = time.time(); TINY = 'r = np.random.default_rng(0); c = dict(D=64, H=4, KV=2, hd=16, I=160, V=500, NL=2, T=64, theta=1e4, eps=1e-6, tie=True)\nn = lambda *sh: r.standard_normal(sh) * 0.08\nE = r.standard_normal((c["V"], c["D"])) * 0.5; GF = 1 + 0.1 * r.standard_normal(c["D"]); T_ = {}\nfor i in range(c["NL"]):\n    d = dict(g1=1 + .1 * r.standard_normal(64), g2=1 + .1 * r.standard_normal(64), wq=n(64, 64) * 2, bq=n(64), wk=n(64, 32) * 2, bk=n(32),\n             wv=n(64, 32) * 2, bv=n(32), wo=n(64, 64), wg=n(64, 160) * 2, wu=n(64, 160) * 2, wd=n(160, 64))\n    p = PFX.format(i)\n    T_.update({p + "input_layernorm.weight": d["g1"], p + "post_attention_layernorm.weight": d["g2"],\n        p + "self_attn.q_proj.weight": d["wq"].T, p + "self_attn.q_proj.bias": d["bq"], p + "self_attn.k_proj.weight": d["wk"].T,\n        p + "self_attn.k_proj.bias": d["bk"], p + "self_attn.v_proj.weight": d["wv"].T, p + "self_attn.v_proj.bias": d["bv"],\n        p + "self_attn.o_proj.weight": d["wo"].T, p + "mlp.gate_proj.weight": d["wg"].T, p + "mlp.up_proj.weight": d["wu"].T,\n        p + "mlp.down_proj.weight": d["wd"].T})\nT_["model.embed_tokens.weight"] = E; T_["model.norm.weight"] = GF\nget = lambda k: T_[k]\nr2 = np.random.default_rng(1); windows = [r2.integers(0, c["V"], c["T"]).tolist() for _ in range(4)]\n'
if PRE:
    g_ = dict(globals()); exec(TINY, g_); get, c = g_["get"], g_["c"]; c["T"] = 64
    m = IntModel(get, c); ids = [int(x) for x in g_["windows"][0][:40]]
    bits = compress_block(m, ids)
    d = Dec(bits); out = [d.get(uniform_cum(c["V"]))]
    for t in range(1, len(ids)): out.append(d.get(int_freqs(m(out)[-1])))
    R["tiny_roundtrip"] = out == ids; R["tiny_bits"] = len(bits)
    print(json.dumps(R, indent=1)); raise SystemExit
import torch
OUT = "/kaggle/working/results.json"
get, c, tok = open_hf("Qwen/Qwen2.5-0.5B"); c["T"] = 512
m = IntModel(get, c); tm = TModel(m, torch.device("cpu"))
try:
    from datasets import load_dataset
    text = (chr(10) * 2).join(load_dataset("wikitext", "wikitext-2-raw-v1", split="test")["text"])
except Exception:
    import pandas as pd
    from huggingface_hub import hf_hub_download
    text = (chr(10) * 2).join(pd.read_parquet(hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset"))["text"])
all_ids = tok(text)["input_ids"]
blocks = [all_ids[2000 + i * 256: 2000 + (i + 1) * 256] for i in range(4)]
raw = tok.decode([t for b in blocks for t in b]).encode("utf-8")
R["tokenizer_lossless"] = tok(tok.decode(blocks[0]))["input_ids"] == blocks[0]
nbits = 0; ok = True
for bi, b in enumerate(blocks):
    bits = compress_block(m, b); nbits += len(bits)
    back = decompress_block(tm, bits, len(b)); ok &= back == b
    P("block", bi, "bits", len(bits), "roundtrip", back == b, round(time.time() - t0))
R["C1_roundtrip_numpy_to_torch_cache"] = bool(ok)
R["raw_bytes"] = len(raw); R["exact_llm_bytes"] = (nbits + 7) // 8
R["bits_per_byte"] = {"exact_llm_0.5B": nbits / len(raw), "gzip9": 8 * len(gzip.compress(raw, 9)) / len(raw),
                      "bz2_9": 8 * len(bz2.compress(raw, 9)) / len(raw), "xz9": 8 * len(lzma.compress(raw, preset=9)) / len(raw)}
# C3 float control: HF fp32, same integer-frequency quantisation, full forward (4 threads) vs KV-cache incremental (1 thread)
from transformers import AutoModelForCausalLM
hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B", dtype=torch.float32).eval()
def fq(logits_row):
    p = torch.softmax(logits_row.double(), -1).numpy(); return np.cumsum(np.floor(p * (1 << 20)).astype(np.int64) + 1)
b = blocks[0]; diff = 0
with torch.no_grad():
    torch.set_num_threads(4); full = hf(torch.tensor([b])).logits[0]
    torch.set_num_threads(1); out = hf(torch.tensor([b[:1]]), use_cache=True); pkv = out.past_key_values; inc = [out.logits[0, -1]]
    for t in range(1, len(b) - 1):
        out = hf(torch.tensor([[b[t]]]), past_key_values=pkv, use_cache=True); pkv = out.past_key_values; inc.append(out.logits[0, -1])
for t in range(len(b) - 1):
    diff += int(not np.array_equal(fq(full[t]), fq(inc[t])))
R["C3_float_positions_with_different_freqs"] = f"{diff}/{len(b) - 1}"
R["C3_float_codec_decodable"] = diff == 0
R["seconds"] = round(time.time() - t0); P(json.dumps(R, indent=1)); json.dump(R, open(OUT, "w"), indent=1)
