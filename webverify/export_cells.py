# Appended after gen_audit_tail.py (tiny model, ZL_TINY=1): exports self-contained audit cells (opening + only that layer's key) as JSON,
# with the Python verifier's verdict, so a browser (JavaScript) verifier can be checked against it.
def _vl(vl):
    return {"N": int(vl.N), "K": int(vl.K), "M": [int(v) for v in vl.M], "Mbits": int(vl.Mbits), "b": None if vl.b is None else [int(v) for v in vl.b],
            "keys": [{"p": int(p), "r": [int(v) for v in r], "d": [int(v) for v in d]} for p, r, d in vl.keys]}
def export_cell(prov, cell, label, lie=None, token_check=None, ids_=None):
    ids_ = ids_ or wA; L, t = cell; NL = len(vm.layers)
    names = needed(cell, NL); op = prov.open(names, lie=lie)
    ok, why = verify_cell(vm, c, prov.root, prov.index, cell, op, ids_, token_check)
    o = {"label": label, "cell": [L, t], "root": prov.root.hex(), "python_ok": bool(ok), "python_reason": str(why),
         "cfg": {"H": c["H"], "KV": c["KV"], "hd": c["hd"], "NL": NL, "F": F},
         "opened": [{"name": list(n), "index": int(prov.index[n]), "vals": [[int(x) for x in np.ravel(v)] for v in op[n][0]],
                     "proof": [h_.hex() for h_ in op[n][1]]} for n in names]}
    if L == NL:
        o["head"] = _vl(vm.head); o["gf"] = [int(v) for v in vm.gf]
    else:
        V_ = vm.layers[L]; o["layer"] = {k: _vl(V_[k]) for k in ("q", "k", "v", "o", "gate", "up", "down")}
        o["g1"] = [int(v) for v in V_["g1"]]; o["g2"] = [int(v) for v in V_["g2"]]
        o["cos_t"] = [int(v) for v in vm.cos[t]]; o["sin_t"] = [int(v) for v in vm.sin[t]]
        if L == 0: o["emb"] = [{"p": int(p), "r": [int(v) for v in r], "digest": int(de[ids_[t]])} for p, r, de in vm.emb.keys]
    return o
cases = [export_cell(pv, (0, 7), "honest layer-0 cell"), export_cell(pv, (1, T_A - 1), "honest last-position cell"),
         export_cell(pv, (NLm, 5), "honest final (head) cell"), export_cell(pv, (1, 9), "lying opening", lie=("R", 1, 9)),
         export_cell(pc, (CL, 11), "cheating model at its layer"), export_cell(pf, (FL, T_A - 2), "fake attention (late row)"),
         export_cell(pk, (FL, SS), "forged cached K row, owner cell"), export_cell(pk, (FL, SS + 3), "forged cached K row, later cell")]
tables = {"EXP_T": [int(v) for v in EXP_T], "SILU_T": [int(v) for v in SILU_T], "EXP_STEP": EXP_STEP, "SILU_STEP": SILU_STEP}
out_dir = os.environ.get("WV_OUT", ".")
json.dump({"tables": tables, "cases": cases}, open(os.path.join(out_dir, "cells_tiny.json"), "w"))
P("exported", len(cases), "cases:", [(x["label"], x["python_ok"]) for x in cases])
