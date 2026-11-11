# exact-llm: bit-exact LLM inference that is also fast

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24137508.svg)](https://doi.org/10.5281/zenodo.24137508)

**Same model + same input → the same output bits on any machine, at float-level quality, decoding at least as fast as an
equally optimised float16 implementation on the same GPU (batch 1, NVIDIA T4).**

Floating-point sums give different low-order bits when they are added in a different order. That order changes with the
hardware, kernel, batch and thread count. So today two honest servers return different numbers for the same request, nobody can
re-check a provider's output, and a provider serving a quietly degraded model is hard to catch.

`exact-llm` defines the model as an **integer program**: int8 weights (with integer scales), integer activations, exact integer sums,
and integer tables for the nonlinearities. Every summation order gives the same integer, so every conforming implementation gives the
same bits.

## Results (Qwen2.5, WikiText-2 test; all checks pre-registered before each run)

### Quality: perplexity vs the float32 model
| model | per-channel int8 weights | group-128 int8 weights | 1,024-token context |
|---|---|---|---|
| Qwen2.5-0.5B | +0.55% | **+0.14%** | +0.34% |
| Qwen2.5-1.5B | +0.37% | **−0.07%** (within noise) | +0.33% |
| Qwen2.5-7B | +0.42%¹ | not measured | not measured |

16 × 256 tokens (4 × 1,024 for long context). ¹ Against a float32 reference that matches HF fp32 to 1e-4 logits at 0.5B/1.5B; measured
on spec v0.1. A v0.4 re-run is in progress.

### Same bits everywhere
Integer-logit SHA-256 hashes were identical:
- on x86 CPU (numpy, 2 machines) and an NVIDIA T4 (float64 and int8 tensor cores), for 0.5B and 7B;
- for the fast compiled GPU path and the slow numpy reference, at 0.5B, 1.5B and 7B;
- across different summation orders, thread counts and batch layouts;
- for KV-cache decoding against full recomputation, and for a prefix computed alone against the same prefix inside a longer input.

Ordinary float32 and HF fp32 inference changed bits across summation orders, thread counts, and CPU vs GPU in the same runs.

### Decode speed (NVIDIA T4, batch 1, 32 tokens after a 64-token prompt)
Both sides used `torch.compile` and one CUDA graph per decode step, in the same harness:

| model | exact (group-128, custom exact int8 GEMV) | float16 (same-harness implementation) | ratio |
|---|---|---|---|
| Qwen2.5-0.5B | **153.0 tok/s** | 147.3 tok/s | 1.04× |
| Qwen2.5-1.5B, run 1 | **62.7 tok/s** | 49.8 tok/s | 1.26× |
| Qwen2.5-1.5B, run 2 (separate Kaggle session) | **66.2 tok/s** | 62.3 tok/s | 1.06× |
| Qwen2.5-7B | **19.3 tok/s** (11.2 GB) | does not fit in the T4's memory | — |

The float16 baseline varied between the two sessions (49.8 vs 62.3 tok/s), so the honest 1.5B result is **1.06–1.26×**: exact decoding at
least matched float16 in every run.

Exact decoding is at least as fast here because int8 weights are half the bytes of float16 and decoding is memory-bound. The step that
made this possible: an unfused exact step launched 26,656 tiny GPU kernels per token, and fusing them gave 3× the speed with identical bits.

**Scope:** batch-1 decoding on one older GPU, compared against our own float16 implementation under identical optimisation. HF's compiled
static cache crashed in our tests, and vLLM / TensorRT-LLM were not tested. **Prompt processing (prefill) is still 3–5× slower than float16.**

### Applications built on exactness
- **One-request provider check:** a provider that cuts one layer to 7 bits (+0.07% perplexity) changes the output hash on the first
  request. A statistical next-token comparison would need about 3,000 tokens.
- **Receipts for sampled text:** an integer RNG plus an integer sampler make sampling exact. A receipt is the prompt, seed and tokens,
  plus a hash chain over every step's logits. A cheating model whose sampled **text was identical** to the honest one was still rejected.
- **Spot-check (expensive prover, cheap verifier):** the verifier recomputes one random block at one position. That took 1.1 s on a CPU
  for 7B (a full forward pass takes 77.6 s), and it caught a one-layer cheat whenever that layer was picked, with 0 false alarms.
- **Portable LLM compression:** text compressed with one engine decompressed byte-exactly with a different engine. On a 4.5 KB sample
  that was 1.04 bits/byte vs xz 3.87. A float version could not decode (230 of 255 positions differed). A larger, fairer sample is in progress.

The receipt, spot-check and cheat numbers were measured on spec v0.2. A v0.4 re-run is in progress.

## How it works (see [SPEC.md](SPEC.md))
- **Linear layers:** per-row power-of-two quantisation of activations to int16, int8 weights with per-channel or per-group-128 integer
  scales, exact integer accumulation, and requantisation by an integer multiplier and a rounding shift.
- **Exactness rule:** any implementation may use int8 tensor cores (via balanced base-256 digits), int32/int64, or float64 arithmetic,
  **only while every partial sum stays exact**. Then every summation order gives the same integer.
- **Nonlinearities:** RMSNorm uses an integer square root. RoPE, exp and SiLU use shipped integer tables (never libm, whose last bits
  differ across platforms). The attention scale is an integer constant, and softmax is normalised by integer division.
- **Causal (spec v0.3+):** a prefix gives the same bits alone or inside a longer input, so KV caches work exactly.
- **No length cap (spec v0.4):** attention value products are exact int64.
- Conversion from a float checkpoint uses float64 arithmetic; conformance refers to the integer artifact.

## Relation to prior and concurrent work (credit where due)
- **tritify** (Pelloran, [arXiv 2611.13730](https://arxiv.org/abs/2611.13730)): exact-integer runtime, CPU vs GPU bit-identity,
  106B MoE across machines; no speed claim. Leads on scale.
- **samebit** ([GitHub](https://github.com/Shakhtar-Sankur/samebit)): integer-only spec on Qwen2.5-0.5B with five independent backends
  and ARM CI; uses the same float64 exactness bound; reports 11× slower than fp16 on a T4.
- **bitexact-llm** ([GitHub](https://github.com/mynotwo/bitexact-llm)): integer Llama-3.2-1B, identity across devices, batch and KV
  cache; vLLM plugin.
- **int-llm** ([GitHub](https://github.com/nmicic/int-llm)): integer TinyLlama, bit-identical from servers to microcontrollers.
- **Cankaya, ICML 2026** ([arXiv 2606.00279](https://arxiv.org/abs/2606.00279)): bit-exact recomputation of *float* GPU inference by
  emulation, given the provider's kernel and batch information.
- **EigenAI** ([arXiv 2602.00182](https://arxiv.org/abs/2602.00182)): bit-exact float inference on fixed GPU architectures (~2% overhead).
- **Anthropic FlashProve:** zkSNARK provable inference with integer-based inference (70–95× GPU cost).
- **Integer-only inference** (I-LLM [2405.17849](https://arxiv.org/abs/2405.17849); [2609.16161](https://arxiv.org/abs/2609.16161)),
  **QuaRot** (Hadamard rotation), **Ozaki scheme** (limb splitting), **TOPLOC** and Moonshot's K2-Vendor-Verifier (statistical checks),
  and the Thinking Machines / vLLM / SGLang deterministic modes (same hardware only).

What this repository adds, as far as we know: exact inference at float-level quality that is **measured** to decode at least as fast as an
equally optimised float16 implementation, plus the verification tools above.

## Limits (read before citing)
- **Speed:** only batch-1 decoding is measured; prefill is slower than float16; one GPU type (T4); no comparison with vLLM / TensorRT-LLM.
- **Quality:** 4,096 evaluation tokens per model; perplexity only, no downstream benchmarks yet.
- **Models:** Qwen2.5 dense 0.5B–7B. Mixture-of-experts models are not supported yet.
- **Hardware:** no ARM, Apple or AMD runs yet.
- **Verification:** needs fresh or secret challenges, because a provider could memorise public ones.
- The speed path (custom CUDA kernel + graph) currently lives in `experiments/speed_fair_kaggle.py`, not yet in the library module.

## Failures and bugs (disclosed)
- **Killed by our own gates:** naive 8-bit activations (+8.8% / +11.8%); exact int4 weights (+21% and worse); power-of-two group scales
  (missed their gate).
- **Bugs found by pre-registered checks:**
  - a RoPE sign error;
  - a silently truncated fractional shift (head dim 128);
  - a shared lookup-table constant;
  - a float32-vs-float64 weight-conversion mismatch;
  - a T4 cuBLASLt int8 failure at one row count (worked around by padding);
  - BitNet scaling and RMSNorm-epsilon errors (BitNet experiment still in progress).
- **An earlier speed claim, withdrawn:** "near float16" was measured against HF's slow eager float16. The table above is the fair version.

## Reproduce
- `exact_llm.py`: the numpy reference, including sampling, receipts and the spot-check.
- `exact_torch.py`: the same program as torch ops (CPU/CUDA).
- `experiments/`: the Kaggle scripts as run. Each states its pre-registered gate in its header, and most run on a tiny random model with
  `PREFLIGHT=1`. `reproduce_kaggle.py` rebuilds the core numbers on a fresh machine.

Built by Novain Shibu with AI assistance (Anthropic Claude) for code and analysis. All numbers come from the scripts in this repository,
run on free Kaggle CPU and T4 machines. Models: Qwen2.5 (Apache-2.0); no model weights are redistributed.

Cite: Shibu, N. (2026). *exact-llm: bit-exact LLM inference that is also fast* (v0.4.0). Zenodo. https://doi.org/10.5281/zenodo.24137508 (all versions: https://doi.org/10.5281/zenodo.24137507)

License: Apache-2.0.
