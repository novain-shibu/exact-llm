# exact-ai GEMM spark v2 (2026-11-12), T4. SAME gate as v1 (1.5B layer GEMMs, M=256, fp16_ms / int8_ms >= 1.3), but the int8 weight is passed
# COLUMN-MAJOR (stored (N,K) contiguous, passed as .t()). Reason: pytorch/pytorch#165230 (open) reports torch._int_mm ~3x slower with a row-major rhs,
# which is what v1 used (v1 result 0.543 = FAIL). Also reports row-major again in the same session, for a same-session comparison.
# exact-ai GEMM spark (2026-11-11), T4.
# PRE-REGISTERED GATE: for the GEMMs of ONE Qwen2.5-1.5B decoder layer at prefill M=256 tokens, the summed time of
#   int8 tensor-core GEMM (torch._int_mm, int8 x int8 -> int32, exact) must be <= fp16 GEMM time / 1.3  (i.e. int8 >= 1.3x faster).
#   PASS -> build the A8 single-pass fused prefill route. FAIL -> kill the A8 prefill route before building.
# Also reported (no gate): M = 64, 1024, 4096 (batched prefill), 7B layer shapes, and the 2-pass A16 cost (2x int8 GEMMs).
# Exactness check: int32 results == int64 CPU reference on a small slice (int8 sums are exact while |sum| < 2^31: K*128*128 <= 2^31 for K <= 131072).
import json, time, torch
assert torch.cuda.is_available(); dev = torch.device("cuda"); P = lambda *a: print(*a, flush=True)
R = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__}
SH = {"1.5B": dict(qkv=(1536, 2048), o=(1536, 1536), gate_up=(1536, 2 * 8960), down=(8960, 1536)),
      "7B": dict(qkv=(3584, 4608), o=(3584, 3584), gate_up=(3584, 2 * 18944), down=(18944, 3584))}
def bench(f, it=50):
    for _ in range(5): f()
    torch.cuda.synchronize(); s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(it): f()
    e.record(); torch.cuda.synchronize(); return s.elapsed_time(e) / it          # ms
g = torch.Generator(device="cpu").manual_seed(0)
for model, shapes in SH.items():
    for M in (64, 256, 1024, 4096):
        Mp = ((M + 63) // 64) * 64; row = {}; t8 = t16 = 0.0
        for name, (K, N) in shapes.items():
            a8 = torch.randint(-128, 128, (Mp, K), dtype=torch.int8, generator=g).to(dev)
            w8 = torch.randint(-128, 128, (N, K), dtype=torch.int8, generator=g).to(dev).t()   # column-major rhs
            w8r = w8.contiguous()                                                            # row-major copy (v1 layout)
            a16 = torch.randn(Mp, K, dtype=torch.float16, device=dev); w16 = torch.randn(K, N, dtype=torch.float16, device=dev)
            try:
                ti = bench(lambda: torch._int_mm(a8, w8)); err = None
                tir = bench(lambda: torch._int_mm(a8, w8r))
            except Exception as ex:
                ti = float("nan"); err = str(ex)[:160]
            tf = bench(lambda: a16 @ w16)
            row[name] = dict(K=K, N=N, int8_colmajor_ms=round(ti, 4), int8_rowmajor_ms=round(tir, 4), fp16_ms=round(tf, 4), ratio_fp16_over_int8=round(tf / ti, 3), ratio_rowmajor=round(tf / tir, 3), err=err)
            t8 += ti; t16 += tf
            if model == "1.5B" and M == 256 and name == "o":                   # exactness spot check
                ref = (a8[:8].cpu().to(torch.int64) @ w8.cpu().to(torch.int64)[:, :64])
                R["exact_int32_eq_int64"] = bool(torch.equal(torch._int_mm(a8, w8)[:8, :64].cpu().to(torch.int64), ref))
            del a8, w8, w8r, a16, w16
        row["layer_total"] = dict(int8_ms=round(t8, 4), fp16_ms=round(t16, 4), speedup_A8=round(t16 / t8, 3), speedup_A16_two_pass=round(t16 / (2 * t8), 3))
        R[f"{model}_M{M}"] = row; P(model, M, json.dumps(row["layer_total"]))
        torch.cuda.empty_cache()
g15 = R["1.5B_M256"]["layer_total"]["speedup_A8"]
R["gate"] = dict(metric="1.5B layer GEMMs, M=256, fp16_ms/int8_ms", value=g15, threshold=1.3, verdict="PASS" if g15 >= 1.3 else "FAIL")
P(json.dumps(R, indent=1)); json.dump(R, open("/kaggle/working/results.json", "w"), indent=1)
