# Freivalds spark (local, synthetic, 2026-11-12). PRE-REGISTERED before running:
#  F1 honest: check passes for 200 random (x) on a 7B-shaped grouped exact linear (K=3584 -> N=18944, group-128 integer scales).
#  F2 forged: y with ONE entry off by +-1 is rejected 200/200; y + p1*e (multiple of one prime) rejected 200/200.
#  F3 report verifier time per check vs exact GEMV time, and key bytes vs weight bytes. No gate on F3.
# Exact layer (spec): t_n = sum_g M[n,g] * sum_{k in g} W[n,k] x[k]   (int8 W, int16 x, integer scales M) -> y = requant(t) elementwise.
# Verifier key (secret): r1,r2 random mod primes p1,p2; d_i[k] = sum_n r_i[n] * M[n,g(k)] * W[n,k] mod p_i.  Check: r_i . t == d_i . x (mod p_i), |t| < bound.
import numpy as np, time, json
rng = np.random.default_rng(0); K, N, G = 3584, 18944, 128; NG = K // G
P1, P2 = 2147483647, 2147483629
W = rng.integers(-127, 128, (N, K), dtype=np.int8); M = rng.integers(1, 1 << 20, (N, NG), dtype=np.int64)
def exact_t(x):
    acc = (W.astype(np.float64).reshape(N, NG, G) * x.astype(np.float64).reshape(1, NG, G)).sum(-1)   # |acc| < 2^53: exact
    return (acc.astype(np.int64) * M).sum(-1)                                                          # < 2^22*2^20*28 < 2^47
BOUND = 127 * 32767 * G * (1 << 20) * NG
assert BOUND < P1 * P2 // 2
def mk_key(p, seed):
    r = np.random.default_rng(seed).integers(0, p, N, dtype=np.int64); d = np.zeros(K, dtype=np.int64)
    for g in range(NG):                                    # d[k] = sum_n (r[n]*M[n,g] mod p) * W[n,k] mod p, chunked to stay < 2^63
        c = (r * (M[:, g] % p)) % p                        # (N,) < 2^31
        Wg = W[:, g * G:(g + 1) * G].astype(np.int64)
        acc = np.zeros(G, dtype=np.int64)
        for s in range(0, N, 4096): acc = (acc + (c[s:s + 4096, None] * Wg[s:s + 4096]).sum(0)) % p   # 4096*2^31*2^7 = 2^50
        d[g * G:(g + 1) * G] = acc
    return r, d
t0 = time.time(); keys = [(p,) + mk_key(p, 100 + i) for i, p in enumerate((P1, P2))]; t_key = time.time() - t0
def check(x, t):
    if np.abs(t).max() >= BOUND: return False
    for p, r, d in keys:
        lhs = int(((r * (t % p)) % p).sum() % p); rhs = int(((d * (x.astype(np.int64) % p)) % p).sum() % p)
        if lhs != rhs: return False
    return True
R = {}; ok = 0; rej1 = 0; rejp = 0; tv = []; tg = []
for i in range(200):
    x = rng.integers(-32767, 32768, K, dtype=np.int64)
    a = time.perf_counter(); t = exact_t(x); tg.append(time.perf_counter() - a)
    a = time.perf_counter(); ok += check(x, t); tv.append(time.perf_counter() - a)
    f = t.copy(); f[rng.integers(N)] += rng.choice([-1, 1]); rej1 += not check(x, f)
    f = t.copy(); f[rng.integers(N)] += P1; rejp += not check(x, f)
R.update(F1_honest_pass=f"{ok}/200", F2_off_by_one_rejected=f"{rej1}/200", F2_plus_p1_rejected=f"{rejp}/200",
         verifier_ms=round(1e3 * float(np.median(tv)), 3), exact_gemv_numpy_ms=round(1e3 * float(np.median(tg)), 1),
         key_bytes=2 * (N + K) * 8, weight_bytes=N * K, key_build_s=round(t_key, 1),
         speedup=round(float(np.median(tg)) / float(np.median(tv)), 1), size_ratio=round(N * K / (2 * (N + K) * 8), 1))
print(json.dumps(R, indent=1)); json.dump(R, open("spark_out.json", "w"), indent=1)
