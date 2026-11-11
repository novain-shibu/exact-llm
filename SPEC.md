# Exact Model Release — spec v0.4 (2026-11-11)

Goal: a model release whose outputs are defined **to the bit**, so every honest implementation on any hardware
returns identical logits, and anyone can verify a provider with one replayed request.
Analogy: a video codec's bit-exact reference decoder plus conformance streams.

## 1. Artifact (conversion from a float checkpoint MUST use float64 arithmetic; the integer artifact, not the float checkpoint, is what conformance refers to)
An exact release = `weights.int8` + `scales.int` + `tables.int` + `manifest.json`. All values are integers; no floats ship.
- `manifest.json`: architecture (layers, dims, heads, kv-heads, vocab), fixed-point format (Q16), table ids, spec version,
  and SHA-256 of every file.
- Tables (exp, SiLU, RoPE cos/sin) ship as integer arrays. They are never recomputed from libm, because transcendental
  functions differ in the last bit across platforms.

## 2. Numeric program (v0.4)
| op | definition |
|---|---|
| activations | int64, Q16 fixed point |
| linear | weights int8 with integer scale M per output column (per-channel) or per 128-row group per column (group-128: `y = round_shift(Σ_g acc_g·M_g, 28 − s)`); per row: shift `s = max(bitlen(max\|x\|) - 15, 0)`, `xq = round_shift(x, s)` (int16); `acc = Σ xq·w` exactly (int8 weights); `y = round_shift(acc · M_c, 28 - s)` + bias(Q16) |
| round_shift(a, n) | `(a + 2^(n-1)) >> n` (arithmetic shift, round half up); n ≤ 0 means left shift |
| RMSNorm | `k = max(bitlen(max\|x\|) - 24, 0)`, `xs = round_shift(x, k)`, `ms = Σxs² // D + 1`, `r = isqrt(ms)` (floor), `y = (x << 16) // (r << k)`, then `round_shift(y·g, 16)` |
| RoPE | `round_shift(x·cos + rotate_half(x)·sin, 16)`, cos/sin = Q16 shipped tables |
| scores | q, k row-quantised as in linear; `acc = q·kᵀ`; `score = round_shift(acc · C, 32 - s_q - s_k)` with integer `C = round(2^16 / √d_head)` (never a fractional shift) |
| softmax | `d = max - score`; exp via shipped Q24 table at 1/256 steps with linear interpolation; `p = (e << 20) // Σe` (Q20) |
| p·V | V kept as Q16 integers (no shared shift, so it is causal); `out = round_shift(p·v, 20)` computed exactly in int64 (no length cap) |
| SiLU | shipped Q16 table on [-16, 16) at 1/256 steps, interpolated; identity above 16, 0 below -16 |
| logits | int64 Q16, the conformance output |

**Overflow rule (v0.2):** when `acc · M` could exceed int64, drop low bits first, **per row**: `drop = max(0, bitlen(max_row|acc|) + bitlen(max M) − 62)`, `y = round_shift(round_shift(acc, drop) · M, 28 − s − drop)`. It never triggered in any tested model. Per-row keeps the spot-check (one row) identical to the full computation.

**Exactness rule:** every reduction is an integer sum. An implementation may use int8 tensor cores, int32 or int64
accumulation, or float64 arithmetic **only** while every partial sum stays below 2^53. Under this rule all summation
orders give the same result, which BC1 and BC1b verified on CPU BLAS, T4 cuBLAS float64 and T4 int8 tensor cores.

## 3. Sampling (v0 open item)
Greedy decoding is exact as specified. Seeded sampling needs a specified integer RNG (e.g. a counter-based PRNG keyed on
seed and position) and an integer inverse-CDF over Q20 probabilities. This is not tested yet.

## 4. Verification protocol
1. The verifier picks a fresh, secret prompt. Fixed public challenges can be memorised by a cheating provider.
2. It sends the prompt to the provider and asks for logits (or the hash of the Q16 logits) at chosen positions.
3. It recomputes the same positions with any conforming implementation, on any device.
4. Equal hashes mean the provider conforms; any difference proves non-conformance. BC1c: a cheat costing only +0.07%
   perplexity was caught in 1 request, where a statistical next-token check needs about 3,000 tokens.

## 5. Measured so far
- Qwen2.5-0.5B: +0.55% perplexity versus fp32 (WikiText-2, 4,096 tokens).
- Identical bits across 3 summation orders, 2 CPU machines, T4 float64 and T4 int8.
- Qwen2.5-1.5B: +0.36% perplexity, 1 hash across 3 configs (BC2 v2, after fixing my fractional-shift bug).
- Qwen2.5-7B: +0.42% perplexity vs float32 reference, hash stable across reduction orders (spec v0.1 drop rule).

## 6. Open items before v1
- Mixture-of-experts routing ties (top-k on equal integer scores needs a specified tie-break, e.g. lowest index).
- Long context: run softmax and attention in blocks without changing results (requires exact integer normalisation).
- A full GPU elementwise path, and a speed comparison with bf16 serving.
- 7B-class quality: DONE (Qwen2.5-7B +0.42%, hash stable); next is cross-device at 7B.
- Licence: release under the same terms as the base model.
