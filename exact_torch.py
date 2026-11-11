"""exact_torch - the same exact integer program as exact_llm.py, as torch tensor ops (CPU or CUDA).
Load after exact_llm (shares its tables/primitives): exec(open("exact_torch.py").read(), vars(exact_llm)).
CUDA: linear layers on int8 tensor cores (balanced digits), optional exact tensor-core attention (att_tc), KV cache,
fast mode (frexp bitlen), torch.compile support. The custom exact GEMV kernel + CUDA-graph decoder are in experiments/speed_fair_kaggle.py."""

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
