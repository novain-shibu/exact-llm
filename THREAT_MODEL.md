# Security contracts of the exact-llm verification tools

Each tool below is stated with: what it claims, what it assumes, when commitments and challenges happen, and what it does **not** guarantee.
All tools assume the served model is the integer-defined exact-llm model (spec v0.4). None of them covers existing floating-point deployments.
Side channels (timing, response length, refusals), availability and denial of service are out of scope for every tool.

## 1. One-request conformance check (spec v0.2+)
- **Claim:** for a challenge prompt, a provider serving the declared exact model returns output whose hash equals the reference hash. A changed model, for example one layer cut to 7 bits, changed the hash on the first request in our tests.
- **Assumes:**
  - The challenger can compute the reference output, which requires the weights or a trusted party that has them.
  - **The challenge is fresh or secret.** A provider that has seen a challenge before can cache the right answer and serve a different model to everyone else.
- **Timing:** none needed, beyond the challenge being unpredictable to the provider.
- **Does not guarantee:** anything about requests that were not challenged. It is a conformance test for that request, not a proof about the service.

## 2. Receipts for sampled text (spec v0.2+)
- **Claim:** a receipt (prompt, seed, tokens, and a SHA-256 hash chain over every step's logits) lets anyone who has the weights recompute the run exactly and find the first position where it differs.
- **Assumes:**
  - The verifier has the weights, or a trusted recomputation service.
  - The seed is fixed by the protocol, not chosen by the provider: from v0.5 it is SHA-256(prompt ‖ verifier nonce). Otherwise a provider could search over seeds.
- **Timing:** the nonce must reach the provider before generation and be unpredictable to it.
- **Does not guarantee:** a receipt is **evidence for recomputation, not a standalone proof**. Without recomputing, it proves nothing about which model ran.

## 3. Spot-check with commitments (spec v0.2)
- **Claim:** a verifier that has the weights recomputes one random (layer, position) block. It caught a one-layer cheat whenever that layer was picked, with 0 false alarms.
- **Assumes:**
  - **The provider commits** (hashes of every layer's state) **before** the challenge.
  - The challenge (layer, position) is unpredictable to the provider.
- **Does not guarantee:** detection is probabilistic. A cheat confined to a fraction f of the blocks is caught with probability f per check.

## 4. Zero-leak verified inference, full trace (spec v0.4 integer program, v0.5.0 release)
- **Claim:** a verifier that holds **no weights** checks every operation of a forward pass exactly, attention included:
  - Freivalds digests check every linear layer;
  - norms, RoPE, attention, softmax, SiLU and the sampler are replayed exactly.

  With the protocol seed, every sampled token is determined, so the bound in arXiv 2511.02620 (Thm 4.1) gives 0 bits per token through the tokens.
- **Assumes:**
  - The key (random vectors r mod two 31-bit primes, and digests d = W r) is generated honestly by a party that has the weights (**trusted keygen**: the model maker, a TEE or an escrow).
  - **r stays secret from the provider.** If the provider learns a key, forging accumulators that pass may be possible, because a short vector orthogonal to r can plausibly be found by lattice reduction. Keys are therefore single-verifier secrets, and must never be revealed before the provider's trace is fixed.
  - The provider sends the complete trace (the accumulators of every linear and the embedding rows).
- **Soundness:** a wrong accumulator row passes one prime's check with probability at most about 1/p, so about 2^-62 for both primes, given secret r. All non-linear steps are recomputed exactly, so no tolerance is involved.
- **Does not guarantee:** a correct key, if keygen is dishonest. Who sees which prompts. Anything about requests that were not verified.

## 5. Commit-and-audit (v0.5.0)
- **Claim:**
  - The provider publishes a **32-byte Merkle root** over all integer intermediates of a window (with cached K/V rows committed per layer) **before** any challenge.
  - The verifier then challenges random (layer, position) cells, and each audited cell is checked exactly as in 4.
  - Lying openings fail the Merkle check (20/20 in our tests).
- **Assumes:**
  - The same keygen and key-secrecy assumptions as 4.
  - The root is fixed before the challenge, and the challenge is unpredictable to the provider.
- **Guarantee is probabilistic:**
  - A change to the model itself (a weight, a layer) affects every position of its layer, so auditing one position per layer catches it.
  - A cheat aimed at a single cell is caught only if that cell is audited.
  - A forged cached K/V row is caught only by its owner cell.
- **Does not guarantee:** coverage of cells that were not audited.

## 6. Browser verifier (webverify/)
- **Claim:** the same per-cell check as 5, in JavaScript (exact BigInt, Web Crypto SHA-256). Its verdicts, including the failing step, matched the Python verifier on 8/8 honest and tampered cells for real Qwen2.5-0.5B and 7B cells.
- **Assumes:** the same as 5, plus the page itself being trustworthy (served from a source the user trusts, not by the provider).
- **Does not guarantee:** the JS port has not been independently audited.

## Speed measurements (how to read them)
- **Implementation-level comparison (same harness):**
  - Both sides use torch.compile and one CUDA graph per decode step.
  - Exact rate = 32 / time of 32 graph decode steps after a 64-token prefill, median of 3.
- **Serving-engine comparison (vLLM 0.33.0, torch 2.14.0+cu130):**
  - Settings: dtype float16, CUDA graphs on (enforce_eager=False), max_model_len 512, gpu_memory_utilization 0.5, seed 0, greedy, ignore_eos, the same 64 prompt token ids.
  - Rate = 31 / (t(32 new tokens) − t(1)), median of 5.
- These two boundaries are not identical, so the two comparisons are reported separately. Only medians were saved in the runs reported; raw repetitions will be published with the next re-run.
