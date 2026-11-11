# exact-llm: bit-exact, re-checkable LLM inference

**Same model, same input → the same output bits on any machine. Anyone can re-check an AI's output by recomputing it.**

Today, an LLM's output depends on the hardware, kernel, batch and thread count. Floating-point sums give different
low-order bits when added in a different order. So nobody can tell an honest model provider from one serving a quietly
degraded model, and no AI output can be re-checked by a third party.

`exact-llm` defines the model as an **integer program**: int8 weights, integer activations, exact integer sums, and
nonlinearities from shipped integer tables. Every summation order gives the same answer, so every conforming
implementation gives the same bits.

## Results (Qwen2.5 family, WikiText-2 test, 16 × 256 tokens)

| model | perplexity, float | perplexity, exact integer | cost |
|---|---|---|---|
| Qwen2.5-0.5B | 22.262 (HF fp32) | 22.384 | +0.55% |
| Qwen2.5-1.5B | 15.783 (HF fp32) | 15.842 | +0.37% |
| Qwen2.5-7B | 9.824 (float32 reference¹) | 9.865 | +0.42% |

¹ HF fp32 7B does not fit in the 30 GB test machine. The float32 reference matches HF fp32 to 1e-4 logits at 0.5B/1.5B,
and is within 0.13% of HF bf16 at 7B.

**Same bits across devices**: integer-logit SHA-256 hashes were identical on

| | 0.5B (4 windows) | 7B (2 windows) |
|---|---|---|
| x86 CPU, numpy float64 BLAS (2 different machines) | ✓ | ✓ |
| NVIDIA T4, cuBLAS float64 | ✓ | ✓ |
| NVIDIA T4, int8 tensor cores (`torch._int_mm`) | ✓ | ✓ |
| 3 summation orders (default / permuted / reversed) | ✓ | ✓ (2 orders) |

In the same runs, ordinary float32 inference gave a different hash for every summation order, and Hugging Face fp32
changed bits between 1 and 4 CPU threads and between CPU and GPU.

**One-request provider check** (0.5B): a "cheating" provider that cuts one layer's weights to 7 bits changes perplexity
by only +0.07%. A statistical next-token comparison would need about 3,000 tokens to notice. Comparing the exact hash
catches it with **one request**.

**Exact sampling and receipts** (0.5B): sampling uses an integer counter-based RNG (SplitMix64 keyed by seed and
position), an integer temperature, the integer softmax, and an integer inverse-CDF. A generation's **receipt** holds the
prompt, seed, temperature and tokens, plus a hash chain over every step's logits. Anyone can verify a receipt by
recomputing it.

| check | result |
|---|---|
| receipts identical under 3 summation orders | yes |
| one recorded token changed | rejected, at exactly the changed position |
| cheating model (one layer cut to 7 bits) | rejected. **Its sampled text was identical to the honest model's**, and only the logits hash chain caught it. A recorder that stores text alone would have passed it. |
| integer sampler vs float softmax (20,000 seeds) | total variation 0.012 (sampling noise ≈ 0.01) |
| real sampling, not greedy | 17 of 32 tokens differ from argmax |

**Spot-check mode: expensive prover, cheap verifier.** The provider commits hashes of every layer's input. The verifier
picks a random (layer, position), and recomputes only that one block at that one position.

| | 0.5B | 7B |
|---|---|---|
| full forward, 128 tokens (CPU, numpy) | 7.4 s | 77.6 s |
| one spot check (CPU, numpy) | **0.06 s** | **1.1 s** |
| one-position shortcut equals the full block, every layer | yes | — |
| honest provider | 30/30 pass | 10/10 pass |

Cheat detection (0.5B, 120 random checks each):
- **Every layer cut to int4:** caught 120/120.
- **One layer cut to 7 bits:** caught every time that layer was picked (4/4), with 0 false alarms in the other 116 checks.
  A single check catches it with probability 1/24, so about 70 checks give 95% detection.

Checking one 7B layer also needs about 7 s to load and convert that layer's weights.

## How it works (spec v0.1, see [SPEC.md](SPEC.md))
- Linear layers: per-row power-of-two quantisation of activations to int16, int8 per-output-channel weights,
  exact integer accumulation, requantisation by an integer multiplier and a rounding shift.
- Exactness rule: an implementation may use int8 tensor cores, int32/int64, or float64 arithmetic **only while every
  partial sum stays below 2^53**. Then every summation order gives the identical integer.
- RMSNorm by integer square root; RoPE, exp and SiLU from shipped integer tables (never libm, whose last bits differ
  across platforms); attention scale as an integer constant; softmax normalised by integer division.
- Conversion from a float checkpoint uses float64 arithmetic. Conformance refers to the integer artifact.

## Relation to prior work
- **Cankaya, ICML 2026 ([arXiv 2606.00279](https://arxiv.org/abs/2606.00279))** recomputes ordinary *float* GPU inference
  bit-exactly by emulating NVIDIA GPU rounding, given the provider's kernel and batch information. It works on existing
  float models. `exact-llm` instead requires the model in an integer format (+0.4–0.6% perplexity), but needs **no**
  knowledge of the provider's kernels, batch or hardware, and is vendor-independent by construction. The two are complementary.
- **Gensyn REE / RepOps**: bitwise reproducibility across CPU/GPU/Apple with fixed-order float operations; proprietary.
- **ReproFuse ([2609.25624](https://arxiv.org/abs/2609.25624)), LLM-42**: bit-identical outputs across GPU architectures (float, GPU only).
- **Integer-only inference** (I-LLM [2405.17849](https://arxiv.org/abs/2405.17849); [2609.16161](https://arxiv.org/abs/2609.16161)):
  shows integer pipelines keep quality. Our quality numbers are consistent with these; the new part here is the
  cross-device identity and the verification use.
- **Statistical provider checks** (Moonshot K2-Vendor-Verifier; Prime Intellect TOPLOC): approximate, need many requests.
- **Deterministic modes** (Thinking Machines batch-invariant kernels; vLLM/SGLang): same bits on the *same* hardware only.

## Limits (read before citing)
- The reference implementation is **slow** (numpy). There is **no speed claim** yet. In the cross-device tests only the
  matrix multiplications ran on the GPU; elementwise integer ops stayed on CPU.
- Tested: Qwen2.5 dense models up to 7B, context 256, 4,096 evaluation tokens. Not yet tested: mixture-of-experts routing
  ties, long-context blockwise attention, AMD/Apple/ARM hardware.
- Verification needs fresh or secret challenge prompts, because a provider could memorise public ones.

## Reproduce
- `exact_llm.py`: the reference module (inference, sampling, receipts, spot-check).
- `experiments/reproduce_kaggle.py`: a single file that rebuilds the headline numbers on a fresh Kaggle CPU machine (about
  15 min). It reproduced 0.5B (hash `ab68ea173cea07fd`, ppl 22.3845), 7B (hash `18a507fa5edaace7`) and the cheat check.
- `experiments/`: the scripts exactly as they ran for cross-device (7B, T4), sampling, spot-check and the training spark.
  Each states its pre-registered pass/fail gates in its header. Run any of them locally on a tiny random model with
  `PREFLIGHT=1`.

## Future work
- A fused GPU kernel for the non-matmul ops (speed is the main adoption blocker).
- Cross-vendor hashes (ARM, Apple silicon, AMD).
- Mixture-of-experts tie-breaking, long-context blockwise attention, and downstream evals (GSM8K/MMLU).
- **Exact training.** A small spark trained a character-level MLP language model with integer forward, backward and SGD:
  held-out loss came within 0.11% of float64, and the weights were bit-identical across 3 summation orders. It is a toy
  (no attention backpropagation), so it is not a claim about transformer training.

## How this was built (honest log)
Every run had a pre-registered pass/fail gate, written before the run. Three bugs of mine were caught that way and fixed,
and are disclosed here:
1. a RoPE sign error (caught by the float-vs-Hugging Face harness check);
2. a fractional shift for head dimension 128, silently truncated, which cost +3.8% perplexity at 1.5B (found by component
   ablation);
3. a shared lookup-table constant that broke one demo variant (excluded from all claims).

A later clean-copy reproduction also showed that converting weights with float32 instead of float64 arithmetic gives a
different (still exact) integer artifact. The spec now pins float64.

Built by Novain Shibu with AI assistance (Anthropic Claude) for code and analysis; all numbers come from the scripts in
this repository. Models: Qwen2.5 (Apache-2.0); no model weights are redistributed.

License: Apache-2.0.
