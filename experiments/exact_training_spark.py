"""Exact-TRAINING spark (2026-11-11). Local CPU, numpy, no downloads.
Task: char-level next-byte LM (context 8 bytes -> embed 32 each -> hidden 256 ReLU -> 256 logits) on Python stdlib source text.
Integer training: master weights int64 Q24; activations Q12; logits Q12; softmax = integer LUT (same as inference spec) -> Q20;
backward and SGD entirely in integers (exact sums, round-half-up shifts). Float baseline: identical init/data/order, float64.

PRE-REGISTERED (written before the first run):
  PASS if (a) held-out cross-entropy of the integer-trained model is within 5% of the float-trained model after 3000 steps, AND
          (b) integer-trained weights are bit-identical (SHA-256) for default vs permuted vs reversed reduction order.
  Reported: float32 training with permuted reduction order -> weight hashes differ.
  KILL ROUTE if (a) misses by >10%; next routes: stochastic-free higher-precision grads (Q28), larger shift budget.
"""
import os, sys, math, hashlib, glob, time, json
import numpy as np

t0 = time.time()
# ---------------- data: local Python stdlib source ----------------
lib = os.path.dirname(os.__file__)
files = sorted(glob.glob(os.path.join(lib, "*.py")))
buf = b"".join(open(f, "rb").read() for f in files)[:1_200_000]
data = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
split = int(len(data) * 0.9); train, test = data[:split], data[split:]
CTX, E, HID, V, B, STEPS = 8, 32, 256, 256, 64, 3000
rng = np.random.default_rng(0)
batches = [rng.integers(0, len(train) - CTX - 1, B) for _ in range(STEPS)]
test_idx = np.random.default_rng(1).integers(0, len(test) - CTX - 1, 4096)
def take(arr, idx):
    X = np.stack([arr[i:i + CTX] for i in idx]); Y = arr[idx + CTX]; return X, Y

# ---------------- init (shared) ----------------
QW = 24; QA = 12
init = {"E": rng.standard_normal((V, E)) * 0.5, "W1": rng.standard_normal((CTX * E, HID)) / math.sqrt(CTX * E),
        "b1": np.zeros(HID), "W2": rng.standard_normal((HID, V)) / math.sqrt(HID) * 0.5, "b2": np.zeros(V)}
initq = {k: np.rint(v * (1 << QW)).astype(np.int64) for k, v in init.items()}     # exact shared start
initf = {k: v.astype(np.float64) / (1 << QW) for k, v in initq.items()}
LR_SHIFT = 1                         # lr = 0.5
LOG2B = int(math.log2(B))

# ---------------- integer helpers ----------------
def shr(a, n):
    if n <= 0: return a << (-n)
    return (a + (np.int64(1) << (n - 1))) >> n
ORDER = {"perm": None}
def imm(a, b):
    """exact int64 matmul; reduction order selectable (must not matter)."""
    K = a.shape[1]
    if ORDER["perm"] == "perm": p = np.random.default_rng(K).permutation(K)
    elif ORDER["perm"] == "rev": p = np.arange(K)[::-1]
    else: p = np.arange(K)
    r = a[:, p] @ b[p, :]
    assert np.abs(r).max() < (1 << 62)
    return r
EXP_T = np.array([round(math.exp(-i / 256) * (1 << 24)) for i in range(256 * 40 + 2)], np.int64)
def isoftmax(lg):                     # lg Q12 -> p Q20
    d = lg.max(1, keepdims=True) - lg
    u = d << 4                        # Q12 -> Q16 units
    i = np.minimum(u >> 8, len(EXP_T) - 2); fr = u & 255
    e = EXP_T[i] - shr((EXP_T[i] - EXP_T[i + 1]) * fr, 8)
    e = np.where(u < (40 << 16), e, 0)
    return (e << 20) // e.sum(1, keepdims=True)

def int_forward(W, X):
    x = shr(W["E"][X].reshape(len(X), CTX * E), QW - QA)          # Q12
    hp = shr(imm(x, W["W1"]), QW) + shr(W["b1"], QW - QA)          # Q12
    h = np.maximum(hp, 0)
    lg = shr(imm(h, W["W2"]), QW) + shr(W["b2"], QW - QA)          # Q12
    return x, hp, h, lg
def int_step(W, X, Y):
    x, hp, h, lg = int_forward(W, X)
    p = isoftmax(lg); dl = p.copy(); dl[np.arange(B), Y] -= (1 << 20)   # Q20
    gW2 = imm(h.T, dl); gb2 = dl.sum(0)                                # Q32, Q20
    dh = shr(imm(dl, W["W2"].T), QW); dh = np.where(hp > 0, dh, 0)     # Q20
    gW1 = imm(x.T, dh); gb1 = dh.sum(0)                                # Q32, Q20
    dx = shr(imm(dh, W["W1"].T), QW).reshape(B, CTX, E)                # Q20
    gE = np.zeros_like(W["E"]); np.add.at(gE, X, dx)                   # Q20
    # update: W(Q24) -= lr * mean(grad); grads Q32 -> shift 8, Q20 -> shift -4
    W["W2"] -= shr(gW2, 8 + LOG2B + LR_SHIFT); W["W1"] -= shr(gW1, 8 + LOG2B + LR_SHIFT)
    W["b2"] -= shr(gb2, -4 + LOG2B + LR_SHIFT); W["b1"] -= shr(gb1, -4 + LOG2B + LR_SHIFT)
    W["E"] -= shr(gE, -4 + LOG2B + LR_SHIFT)

def flt_step(W, X, Y, dt, mm):
    x = W["E"][X].reshape(len(X), CTX * E).astype(dt)
    hp = mm(x, W["W1"]) + W["b1"]; h = np.maximum(hp, 0); lg = mm(h, W["W2"]) + W["b2"]
    p = np.exp(lg - lg.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
    dl = p.copy(); dl[np.arange(B), Y] -= 1; dl /= B
    gW2 = mm(h.T, dl); dh = mm(dl, W["W2"].T) * (hp > 0); gW1 = mm(x.T, dh)
    dx = mm(dh, W["W1"].T).reshape(B, CTX, E); gE = np.zeros_like(W["E"]); np.add.at(gE, X, dx)
    lr = dt(0.5)
    W["W2"] -= lr * gW2; W["W1"] -= lr * gW1; W["b2"] -= lr * dl.sum(0); W["b1"] -= lr * dh.sum(0); W["E"] -= lr * gE

def ce_float_logits(lg, Y):
    lg = lg.astype(np.float64); m = lg.max(1, keepdims=True)
    return float((np.log(np.exp(lg - m).sum(1)) + m[:, 0] - lg[np.arange(len(Y)), Y]).mean())
def eval_int(W):
    tot = 0.0
    for c in np.array_split(test_idx, 16):
        X, Y = take(test, c); tot += ce_float_logits(int_forward(W, X)[3] / (1 << QA), Y) * len(c)
    return tot / len(test_idx)
def eval_flt(W):
    tot = 0.0
    for c in np.array_split(test_idx, 16):
        X, Y = take(test, c); x = W["E"][X].reshape(len(X), CTX * E)
        lg = np.maximum(x @ W["W1"] + W["b1"], 0) @ W["W2"] + W["b2"]; tot += ce_float_logits(lg, Y) * len(c)
    return tot / len(test_idx)
def hW(W): return hashlib.sha256(b"".join(np.ascontiguousarray(W[k]).tobytes() for k in sorted(W))).hexdigest()[:16]

# ---------------- runs ----------------
log = {"bytes_train": int(len(train)), "steps": STEPS, "batch": B}
res_int = {}
for mode in (None, "perm", "rev"):
    ORDER["perm"] = mode; W = {k: v.copy() for k, v in initq.items()}
    for s in range(STEPS):
        X, Y = take(train, batches[s]); int_step(W, X, Y)
    res_int[str(mode)] = hW(W)
    if mode is None: log["int_test_ce"] = eval_int(W); Wint = W
    print("int", mode, res_int[str(mode)], round(time.time() - t0), "s", flush=True)
ORDER["perm"] = None
Wf = {k: v.copy() for k, v in initf.items()}
for s in range(STEPS):
    X, Y = take(train, batches[s]); flt_step(Wf, X, Y, np.float64, np.matmul)
log["float64_test_ce"] = eval_flt(Wf)
f32 = set()
for seed in (0, 1):
    W3 = {k: v.astype(np.float32) for k, v in initf.items()}
    def mm(a, b, seed=seed):
        if seed == 0: return a @ b
        p = np.random.default_rng(a.shape[1]).permutation(a.shape[1]); return a[:, p] @ b[p, :]
    for s in range(STEPS):
        X, Y = take(train, batches[s]); flt_step(W3, X, Y, np.float32, mm)
    f32.add(hW(W3))
log["init_test_ce"] = eval_flt(initf)
log["uniform_ce"] = math.log(256)
log["int_weight_hashes"] = res_int
log["float32_weight_hashes_2orders"] = len(f32)
log["gap_int_vs_float"] = log["int_test_ce"] / log["float64_test_ce"] - 1
log["pass_a"] = abs(log["gap_int_vs_float"]) <= 0.05
log["pass_b"] = len(set(res_int.values())) == 1
log["verdict"] = "PASS" if log["pass_a"] and log["pass_b"] else "FAIL"
log["seconds"] = round(time.time() - t0)
print(json.dumps(log, indent=1))
json.dump(log, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "spark_train_out.json"), "w"), indent=1)
