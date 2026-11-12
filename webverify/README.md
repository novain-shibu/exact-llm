# Browser verifier (no model weights)

Checks one audited cell of an exact-llm run in the browser, with exact BigInt arithmetic:
- Freivalds digests of that layer instead of its weights;
- exact replay of attention, norms, RoPE and the lookup tables;
- Merkle proofs against the prover's 32-byte commitment.

Run it locally:

    python -m http.server 8765 --directory webverify

Then open http://localhost:8765/ (tiny model test cases). For other case files, use `?f=<cells file>`.

`export_cells.py` is appended to `experiments/commit_audit_kaggle.py` to export cells with the Python verifier's verdicts.
Real Qwen2.5-0.5B cells (12.7 MB) are not included in the repo. Regenerate them with that script (Kaggle CPU).
