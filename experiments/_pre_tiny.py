r = np.random.default_rng(0); c = dict(D=64, H=4, KV=2, hd=16, I=160, V=500, NL=2, T=64, theta=1e4, eps=1e-6, tie=True)
n = lambda *sh: r.standard_normal(sh) * 0.08
E = r.standard_normal((c["V"], c["D"])) * 0.5; GF = 1 + 0.1 * r.standard_normal(c["D"]); T_ = {}
for i in range(c["NL"]):
    d = dict(g1=1 + .1 * r.standard_normal(64), g2=1 + .1 * r.standard_normal(64), wq=n(64, 64) * 2, bq=n(64), wk=n(64, 32) * 2, bk=n(32),
             wv=n(64, 32) * 2, bv=n(32), wo=n(64, 64), wg=n(64, 160) * 2, wu=n(64, 160) * 2, wd=n(160, 64))
    p = PFX.format(i)
    T_.update({p + "input_layernorm.weight": d["g1"], p + "post_attention_layernorm.weight": d["g2"],
        p + "self_attn.q_proj.weight": d["wq"].T, p + "self_attn.q_proj.bias": d["bq"], p + "self_attn.k_proj.weight": d["wk"].T,
        p + "self_attn.k_proj.bias": d["bk"], p + "self_attn.v_proj.weight": d["wv"].T, p + "self_attn.v_proj.bias": d["bv"],
        p + "self_attn.o_proj.weight": d["wo"].T, p + "mlp.gate_proj.weight": d["wg"].T, p + "mlp.up_proj.weight": d["wu"].T,
        p + "mlp.down_proj.weight": d["wd"].T})
T_["model.embed_tokens.weight"] = E; T_["model.norm.weight"] = GF
get = lambda k: T_[k]
r2 = np.random.default_rng(1); windows = [r2.integers(0, c["V"], c["T"]).tolist() for _ in range(4)]
