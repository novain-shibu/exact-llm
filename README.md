# exact-llm: bit-exact LLM inference at float16 decode speed

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24162825.svg)](https://doi.org/10.5281/zenodo.24162825)

**Same model + same input → the same output bits on any machine, at float-level quality, decoding at about the speed of float16
on the same GPU: 0.88–1.10× of vLLM float16 (batch 1, NVIDIA T4).**

**New: zero-leak verified inference.** A verifier that holds **no model weights** checks every operation of a served model exactly,
attention included. It catches one changed weight out of 358 million, catches hidden data in sampled text, and can run in a web
browser. Verification is measured on dense models up to 7B, on a native 1-bit model, and on a 14.3B-parameter mixture-of-experts
model (Qwen1.5-MoE-A2.7B, 2.7B active), where it also catches a token sent to the wrong expert.

Floating-point sums give different low-order bits when they are added in a different order. That order changes with the
hardware, kernel, batch and thread count. So today two honest servers return different numbers for the same request, nobody can
re-check a provider's output, and a provider serving a quietly degraded model is hard to catch.

`exact-llm` defines the model as an **integer program**: int8 weights (with integer scales), integer activations, exact integer sums,
and integer tables for the nonlinearities. Every summation order gives the same integer, so every conforming implementation gives the
same bits.

## Zero-leak verified inference (new, November 2026)
A compromised inference server can hide data, for example stolen model weights, inside ordinary sampled text
([arXiv 2511.02620](https://arxiv.org/abs/2511.02620)). A float verifier has to tolerate honest numerical noise, and that tolerance is the
channel. We measured that noise in our test on a T4: with a fixed per-request seed, **6–22% of 64-token outputs from vLLM 0.33 changed
only because of other requests in the same batch** (greedy decoding included). This is expected behaviour for batched float kernels, not a defect. In HF fp16, batch-1 / batch-8 / KV-cache runs **sampled different tokens at
5.9–7.4% of positions**.

With an integer model there is no honest noise: the output is a pure function of (weights, prompt, verifier nonce). On top of it we built a
verifier that **holds no weights**:
- a secret Freivalds key checks every linear layer (r·acc ≡ d·x mod p);
- attention, norms, softmax and the sampler are replayed exactly;
- the sampling seed is fixed by the protocol: SHA-256(prompt ‖ verifier nonce).

| | Qwen2.5-0.5B | Qwen2.5-1.5B | Qwen2.5-7B |
|---|---|---|---|
| verifier key vs int8 weights | 10.6 MB vs 358 MB | 18.1 MB vs 1.31 GB | **35.6 MB vs 7.07 GB (199×)** |
| honest windows verified, logits identical | 4/4 | 4/4 | 4/4 |
| forged accumulators (±1, +p) rejected | 40/40 | 40/40 | 40/40 |
| model with one layer cut to 7 bits | rejected | rejected | rejected |
| consistent fake attention (one layer/head) | 10/10 rejected | 10/10 | 10/10 |
| fake post-attention output, 5 modes × first/last layer¹ | — | 10/10 rejected | 10/10 rejected |
| hidden bits by token substitution / attacker's own seed | 10/10, 5/5 rejected | 10/10, 5/5 | 10/10, 5/5 |

¹ The attack class that passes an audit-only verifier in CommitLLM's red-team notes (`docs/security/redteam_audit_only.md`). Their test
used a toy 2-layer model in their harness, so this compares attack classes, not numbers.

The degraded model was rejected although it changed **no** output token. Because only one token is valid at each position, the bound of
arXiv 2511.02620 (Theorem 4.1) gives **0 bits per token** through the sampled tokens, against our verifier.

**Commit-and-audit** (Qwen2.5-0.5B): a full trace would be 3.65 MB per token. Instead, the prover sends a **32-byte Merkle root** per
128-token window, and the verifier checks random (layer, position) cells.
- Cached K/V rows are committed too, so a cell re-derives only its own row.
- 40/40 honest cells verified, opening about 436 KB and taking 13 ms per cell.
- At **7B**: a 32-byte root replaces a 790 MB trace per 64-token window; 40/40 honest cells verified, opening about 0.8 MB and taking 24 ms per cell; all attack gates pass.
- A degraded layer and fake attention were caught in exactly the cells they touched.
- A forged cached K row was rejected by the cell that owns it.
- A substituted token was caught by the sampler check.
- 20/20 lying openings were rejected.

**In a web browser, without the model:** a JavaScript port of the verifier (exact BigInt arithmetic, Web Crypto SHA-256) checks real
Qwen2.5-0.5B audit cells in 7–70 ms each and **Qwen2.5-7B** audit cells in 6–117 ms each, using only that layer's key slice and the
prover's openings. It matched the Python verifier's verdict, and the failing step, on 8/8 honest and tampered cells for each model. Code: `webverify/`.

**The fast GPU engine is verifiable too:** its trace (group-128 spec, Qwen2.5-0.5B, 64 tokens on a T4) verified on a CPU with a 27.5 MB
weightless key; the verifier's logits equal the GPU's, and 20/20 single-entry tampers were rejected.

**Server cost on a real GPU** (T4, the fast engine, batch-1 decode): the full commitment costs **+2.2% at 7B and +13.8% at 1.5B**, with
identical outputs. That covers capturing every verified vector inside the CUDA graph, copying it to the host, and SHA-256 in 4 CPU threads.
Capture and copy alone measured +4.1% / +7.1% (1.5B / 7B) in one session and +10.2% / +0.5% in another. With hashing on a single
thread, 1.5B could not keep up (+47%).

**Model identity without disclosure:** a verifier keyed on Qwen2.5-0.5B-Instruct (10.6 MB, no weights) accepted the declared model and
rejected the base model swapped in. It also rejected the declared model with **one int8 weight changed by ±1 out of 357,826,560** in 10/10 trials,
each time at the changed matrix, and one normalisation gain changed by about 1% (10/10). This is the check that verification proposals for AI
agreements ask for ("is the deployed model the evaluated one?"), done in software.

**Mixture-of-experts:** Qwen1.5-MoE-A2.7B (14.3B parameters, 60 experts, top-4 + shared expert) runs as an exact integer program (integer
router with deterministic top-4, integer sigmoid gate) at **+0.35% perplexity** vs a float32 reference (8 × 256 tokens), with a stable hash.
The integer and float routers pick the same top-4 expert set for 88–94% of tokens per layer.
The weightless verifier covers this model as well (201 MB key vs 14.0 GB of int8 weights). It replays the integer router itself, so it knows
which tokens each expert must have processed. Honest windows verified with identical logits (2/2), and it rejected 20/20 forged accumulators,
5/5 fake attention, 5/5 hidden-bit substitutions and **5/5 forged routings** (one token at one layer sent to its 5th-best expert instead of
its 4th, everything after it recomputed honestly). CommitLLM's roadmap lists MoE routing verification as planned (item 29); this is
one model on a CPU, with 64-token windows and 5 trials per attack.

**Native 1-bit model:** the same weightless verification on BitNet b1.58 2B4T (17.3 MB key) accepted honest runs and rejected 20/20
forged accumulators, 5/5 fake attention and 5/5 hidden-bit substitutions.

**Limits:**
- The integer model must be the one served; this does not cover existing float deployments.
- Side channels (timing, length, refusals) and the choice of which messages are audited are out of scope, as in 2511.02620.
- The numpy verifier still replays attention and norms (7.7 s vs 46 s for the prover per window at 7B).
- The audit opening size grows with position.
- The key is built from the weights once, so someone trusted must build it (for example the provider in a TEE, or an escrow).
- Prior work:
  - Freivalds checks (Slalom).
  - Weightless commit-and-audit: CommitLLM, and verifiable-gemma (Gemma 4). Both verify with tolerances and list consistent fake
    attention as open.
  - The threat model, and its capacity bound: 2511.02620. A summary of it notes that deterministic inference would remove the
    steganographic channel.
  - Bit-exact float recomputation (Cankaya, ICML 2026).
  - Cross-device bit-exact engines: tritify, samebit, bitexact-llm, int-llm, and idemcore (closed engine, open hash verifier).
  - An integer model verified in a browser with its weights: ModelProof, a tiny MLP.

  What is new here, as far as we found, is zero-tolerance coverage of every operation including attention, with a weightless verifier,
  measured on real models.

Scripts:
- `experiments/zero_leak_verify_kaggle.py`
- `experiments/commit_audit_kaggle.py`
- `experiments/model_identity_kaggle.py`
- `experiments/gpu_engine_trace_verify_kaggle.py`
- `experiments/prover_commit_cost_gpu_kaggle.py`
- `experiments/moe_exact_kaggle.py`
- `experiments/moe_zero_leak_kaggle.py`
- `experiments/bitnet_zero_leak_kaggle.py`
- `experiments/vllm_batching_nondeterminism_kaggle.py`
- `experiments/float_nondeterminism_hf_kaggle.py`

## Results (Qwen2.5, WikiText-2 test; all checks pre-registered before each run)

### Quality: perplexity vs the float32 model
| model | per-channel int8 weights | group-128 int8 weights | 1,024-token context |
|---|---|---|---|
| Qwen2.5-0.5B | +0.55% | **+0.14%** | +0.34% |
| Qwen2.5-1.5B | +0.37% | **−0.07%** (within noise) | +0.33% |
| Qwen2.5-7B | +0.41%¹ | not measured | not measured |

16 × 256 tokens (4 × 1,024 for long context). ¹ Against a float32 reference that matches HF fp32 to 1e-4 logits at 0.5B/1.5B. Spec v0.4
(v0.1 gave +0.42%).

**A model trained with ternary weights:** BitNet b1.58 2B4T (Microsoft's native 1.58-bit model) run as an exact integer program. Its
ternary weights are used exactly as released; only the scales, norms, embeddings and output head are converted to integers. Perplexity
26.38 vs HF float32 26.22 (+0.62%, 8 × 256 tokens), with the same hash across summation orders. **The same hash (b571ee87…) on two
different CPU machines and on an NVIDIA T4 (float64)**. No speed run yet. Scripts: `experiments/bitnet_exact_kaggle.py`,
`experiments/bitnet_cross_device_kaggle.py`.

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
| Qwen2.5-0.5B, session 1 | 153.0 tok/s | 147.3 tok/s | 1.04× |
| Qwen2.5-0.5B, session 2 | 141.4 tok/s | 147.5 tok/s | 0.96× |
| Qwen2.5-0.5B, session 3 | 148.5 tok/s | 153.5 tok/s | 0.97× |
| Qwen2.5-1.5B, session 1 | 62.7 tok/s | 49.8 tok/s | 1.26× |
| Qwen2.5-1.5B, session 2 | 66.2 tok/s | 62.3 tok/s | 1.06× |
| Qwen2.5-1.5B, session 3 | 59.8 tok/s | 49.7 tok/s | 1.20× |
| Qwen2.5-1.5B, session 4 | 64.4 tok/s | 60.6 tok/s | 1.06× |
| Qwen2.5-7B | 19.3 tok/s (11.2 GB) | does not fit in the T4's memory | — |

Each session is a separate Kaggle T4. Across sessions: **0.96–1.04×** at 0.5B (parity) and **1.06–1.26×** at 1.5B.

**Against vLLM** (v0.33.0, float16, CUDA graphs on, same T4 session and prompt tokens as session 3 / session 4 above):

| model | exact | vLLM float16 | exact / vLLM |
|---|---|---|---|
| Qwen2.5-0.5B | 148.5 tok/s | 169.0 tok/s | 0.88× |
| Qwen2.5-1.5B | 64.4 tok/s | 58.7 tok/s | 1.10× |

vLLM's decode rate is 31 / (time for 32 new tokens − time for 1), median of 5. Our own float16 harness ran at 0.91× and 1.03× of vLLM.

Exact decoding keeps up because int8 weights are half the bytes of float16 and batch-1 decoding is memory-bound. The step that
made this possible: an unfused exact step launched 26,656 tiny GPU kernels per token, and fusing them gave 3× the speed with identical bits.

**Scope:** batch-1 decoding on one older GPU. Batched serving, where vLLM is strongest, was not measured; TensorRT-LLM was not tested.
**Prompt processing (prefill) is 2.65–2.8× slower than float16** (256 tokens, compiled, exact: 0.5B 0.112 s vs 0.040 s, 1.5B 0.198 s
vs 0.075 s for HF fp16). It was 3.3–4.0× before switching the int8 weights to column-major in the same session. The rest of the gap is the
small integer element-wise operations between the matrix products.

Correction (2026-11-12): an earlier version of this README said int8 tensor cores (`torch._int_mm`) were slower than float16 matmul on the
T4 (0.44–0.65×). That came from our weight layout: with a row-major weight matrix `_int_mm` is about 2–3× slower (see
pytorch/pytorch#165230). With column-major int8 weights the same T4 test runs at **1.09–2.19× of float16 matmul speed** (1.42× at 256 prompt
tokens for a Qwen2.5-1.5B layer), still exact.

### Applications built on exactness
- **One-request provider check:** a provider that cuts one layer to 7 bits (+0.07% perplexity) changes the output hash on the first
  request. A statistical next-token comparison would need about 3,000 tokens.
- **Receipts for sampled text:** an integer RNG plus an integer sampler make sampling exact. A receipt is the prompt, seed and tokens,
  plus a hash chain over every step's logits. A cheating model whose sampled **text was identical** to the honest one was still rejected.
- **Spot-check (expensive prover, cheap verifier):** the verifier recomputes one random block at one position. That took 1.1 s on a CPU
  for 7B (a full forward pass takes 77.6 s), and it caught a one-layer cheat whenever that layer was picked, with 0 false alarms.
- **Portable LLM compression:** text compressed with one engine decompressed byte-exactly with a different engine. On 61 KB of
  WikiText-2 (15,360 tokens) that was 1.06 bits/byte vs bz2 2.62, xz 2.74 and gzip 2.95. A float version could not decode (230 of 255
  positions differed in a 4.5 KB test). Compression is slow: it runs the model once per token.

Re-run on spec v0.4 (0.5B): receipts passed every check (sampler total variation 0.012; the identical-text cheat was still
caught), the one-request cheat check caught the 7-bit layer, and the spot-check passed 20/20 honest checks at 0.056 s each. The 7B
spot-check timing and the cheat-detection rates are from spec v0.2.

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
  and ARM CI; uses the same float64 exactness bound; reports 11× slower than fp16 on a T4 for a 256-token prompt.
- **bitexact-llm** ([GitHub](https://github.com/mynotwo/bitexact-llm)): integer Llama-3.2-1B and Llama-3.1-8B, bit-identical across
  H800, H100, L20, x86 and Apple M4 Pro, and across batch and KV cache. Its exact mode keeps the bf16 weights unchanged (mean logit
  difference about 5e-5 from fp32); it has an int8 mode and a vLLM plugin, and measures speed against vLLM. Leads on quality.
- **int-llm** ([GitHub](https://github.com/nmicic/int-llm)): integer TinyLlama, bit-identical from servers to microcontrollers.
- **Cankaya, ICML 2026** ([arXiv 2606.00279](https://arxiv.org/abs/2606.00279)): bit-exact recomputation of *float* GPU inference by
  emulation, given the provider's kernel and batch information.
- **EigenAI** ([arXiv 2602.00182](https://arxiv.org/abs/2602.00182)): bit-exact float inference on fixed GPU architectures (~2% overhead).
- **Anthropic FlashProve:** zkSNARK provable inference with integer-based inference (70–95× GPU cost).
- **Integer-only inference** (I-LLM [2405.17849](https://arxiv.org/abs/2405.17849); [2609.16161](https://arxiv.org/abs/2609.16161)),
  **QuaRot** (Hadamard rotation), **Ozaki scheme** (limb splitting), **TOPLOC** and Moonshot's K2-Vendor-Verifier (statistical checks),
  and the Thinking Machines / vLLM / SGLang deterministic modes (same hardware only).

What this repository adds, as far as we know: exact int8 inference at float-level quality that is **measured** to decode at about float16 speed
(0.96–1.26× of our own same-harness float16; 0.88–1.10× of vLLM float16; batch 1, T4), plus the verification tools above.

## Limits (read before citing)
- **Speed:** only batch-1 decoding is measured; prefill is slower than float16; one GPU type (T4); vLLM compared at batch 1 only; no TensorRT-LLM comparison.
- **Quality:** 4,096 evaluation tokens per model; perplexity only, no downstream benchmarks yet.
- **Models:** Qwen2.5 dense 0.5B–7B, BitNet b1.58 2B4T, and Qwen1.5-MoE-A2.7B (CPU reference; no MoE speed run yet).
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
  - BitNet scaling, RMSNorm-epsilon and int64-overflow errors (found by a layer-by-layer comparison with HF).
- **An earlier speed claim, withdrawn:** "near float16" was measured against HF's slow eager float16. The table above is the fair version.

## Reproduce
- `exact_llm.py`: the numpy reference, including sampling, receipts and the spot-check.
- `exact_torch.py`: the same program as torch ops (CPU/CUDA).
- `experiments/`: the Kaggle scripts as run. Each states its pre-registered gate in its header, and most run on a tiny random model with
  `PREFLIGHT=1`. `reproduce_kaggle.py` rebuilds the core numbers on a fresh machine.

Built by Novain Shibu with AI assistance (Anthropic Claude) for code and analysis. All numbers come from the scripts in this repository,
run on free Kaggle CPU and T4 machines. Models: Qwen2.5 (Apache-2.0), BitNet b1.58 2B4T (MIT), Qwen1.5-MoE-A2.7B (Tongyi Qianwen license, research use); no model weights are redistributed.

Cite: Shibu, N. (2026). *exact-llm: bit-exact LLM inference at float16 decode speed, with zero-leak verification* (v0.5.0). Zenodo.
https://doi.org/10.5281/zenodo.24162825 (all versions: https://doi.org/10.5281/zenodo.24137507)

License: Apache-2.0.
