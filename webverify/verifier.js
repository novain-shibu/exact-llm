// exact-llm weightless audit verifier, in the browser (2026-11-12). Port of verify_cell() from experiments/commit_audit_kaggle.py.
// The verifier never sees model weights: only Freivalds digests (d = W r mod p) for the audited layer, integer scales, norm gains,
// RoPE rows, lookup tables, and the prover's Merkle openings. All arithmetic is exact (BigInt), with Python's floor semantics.
"use strict";
const B = BigInt;
const mod = (a, p) => { const r = a % p; return r < 0n ? r + p : r; };
const fdiv = (a, b) => { const q = a / b; return (a % b !== 0n && ((a < 0n) !== (b < 0n))) ? q - 1n : q; };
const bitlen = (m) => (m <= 0n ? 0n : B(m.toString(2).length));
const shr = (a, n) => (n <= 0n ? a << (-n) : (a + (1n << (n - 1n))) >> n);
const absmax = (arr) => arr.reduce((m, v) => { const a = v < 0n ? -v : v; return a > m ? a : m; }, 0n);
const big = (arr) => arr.map(B);
function qrow(x, bits = 15n) {
  let s = bitlen(absmax(x)) - bits; if (s < 0n) s = 0n; const lim = (1n << bits) - 1n;
  return [x.map((v) => { const y = shr(v, s); return y > lim ? lim : (y < -lim ? -lim : y); }), s];
}
function isqrt(n) {
  if (n < 2n) return n;
  let x = B(Math.floor(Math.sqrt(Number(n))));
  while (x * x > n) x -= 1n;
  while ((x + 1n) * (x + 1n) <= n) x += 1n;
  return x;
}
function irms(x, g, F) {
  let k = bitlen(absmax(x)) - 24n; if (k < 0n) k = 0n;
  const xs = x.map((v) => shr(v, k)); const ss = xs.reduce((a, v) => a + v * v, 0n);
  const ms = fdiv(ss, B(x.length)) + 1n; let r = isqrt(ms); if (r < 1n) r = 1n;
  return x.map((v, i) => shr(fdiv(v << F, r << k) * g[i], F));
}
function requant(vl, acc, s) {
  let drop = bitlen(absmax(acc)) + vl.Mbits - 62n; if (drop < 0n) drop = 0n;
  return acc.map((a, n) => { const y = shr(shr(a, drop) * vl.M[n], 28n - s - drop); return vl.b ? y + vl.b[n] : y; });
}
function frRow(vl, xq, acc) {
  if (acc.length !== vl.N) return false;
  if (absmax(acc) > 127n * 32767n * B(vl.K)) return false;
  for (const key of vl.keys) {
    const p = key.p; let lhs = 0n, rhs = 0n;
    for (let n = 0; n < acc.length; n++) lhs += mod(mod(acc[n], p) * key.r[n], p);
    for (let k = 0; k < xq.length; k++) rhs += mod(mod(xq[k], p) * key.d[k], p);
    if (mod(lhs, p) !== mod(rhs, p)) return false;
  }
  return true;
}
function rotHalf(z) { const h = z.length / 2; return z.slice(h).map((v) => -v).concat(z.slice(0, h)); }
function rope(z, cos, sin, F) { const rh = rotHalf(z); return z.map((v, j) => shr(v * cos[j] + rh[j] * sin[j], F)); }
function isoftmax(sc, T) {
  const mx = sc.reduce((m, v) => (v > m ? v : m), sc[0]); const step = T.EXP_STEP, n = B(T.EXP_T.length - 2);
  const e = sc.map((v) => {
    const d = mx - v; let i = d >> step; if (i > n) i = n; const fr = d & ((1n << step) - 1n);
    const a = T.EXP_T[Number(i)], b = T.EXP_T[Number(i) + 1];
    const val = a - shr((a - b) * fr, step);
    return d < (40n << 16n) ? val : 0n;
  });
  const tot = e.reduce((a, v) => a + v, 0n);
  return e.map((v) => fdiv(v << 20n, tot));
}
function isilu(x, T, ONE) {
  const step = T.SILU_STEP;
  return x.map((v) => {
    if (v >= 16n * ONE) return v;
    if (v < -16n * ONE) return 0n;
    let t = v; if (t > 16n * ONE - 1n) t = 16n * ONE - 1n; t += 16n * ONE;
    const i = Number(t >> step), fr = t & ((1n << step) - 1n);
    return T.SILU_T[i] + shr((T.SILU_T[i + 1] - T.SILU_T[i]) * fr, step);
  });
}
const hexToBytes = (h) => Uint8Array.from(h.match(/../g).map((b) => parseInt(b, 16)));
const cat = (...a) => { const n = a.reduce((s, x) => s + x.length, 0); const o = new Uint8Array(n); let p = 0; for (const x of a) { o.set(x, p); p += x.length; } return o; };
const sha = async (b) => new Uint8Array(await crypto.subtle.digest("SHA-256", b));
function hb(vals) {             // Python Hb(): int64 little-endian bytes of each array, each followed by "|"
  const parts = [];
  for (const arr of vals) { const buf = new BigInt64Array(arr.length); arr.forEach((v, i) => { buf[i] = v; }); parts.push(new Uint8Array(buf.buffer), new Uint8Array([124])); }
  return cat(...parts);
}
async function merkleOk(rootHex, vals, index, proof) {
  let h = await sha(cat(new Uint8Array([76]), await sha(hb(vals))));
  for (const sibHex of proof) { const sib = hexToBytes(sibHex); h = await sha(cat(new Uint8Array([78]), index % 2 === 0 ? cat(h, sib) : cat(sib, h))); index = Math.floor(index / 2); }
  return h.every((v, i) => v === hexToBytes(rootHex)[i]);
}
function prepVl(v) { return { N: v.N, K: v.K, M: big(v.M), Mbits: B(v.Mbits), b: v.b ? big(v.b) : null, keys: v.keys.map((k) => ({ p: B(k.p), r: big(k.r), d: big(k.d) })) }; }
async function verifyCell(cs, T) {
  const [L, t] = cs.cell, cfg = cs.cfg, F = B(cfg.F), ONE = 1n << F, H = cfg.H, KV = cfg.KV, hd = cfg.hd, NL = cfg.NL;
  const op = {};
  for (const o of cs.opened) {
    const vals = o.vals.map(big);
    if (!(await merkleOk(cs.root, vals, o.index, o.proof))) return [false, "merkle " + o.name.join(",")];
    op[o.name.join(",")] = vals;
  }
  const get = (k, L_, t_) => op[[k, L_, t_].join(",")];
  if (L === NL) {
    const head = prepVl(cs.head), x = get("X", NL, t)[0], hacc = get("HEAD", NL, t)[0];
    const [xq] = qrow(irms(x, big(cs.gf), F));
    return frRow(head, xq, hacc) ? [true, null] : [false, "head"];
  }
  const V = {}; for (const k of Object.keys(cs.layer)) V[k] = prepVl(cs.layer[k]);
  const g1 = big(cs.g1), g2 = big(cs.g2), cos = big(cs.cos_t), sin = big(cs.sin_t);
  const x = get("X", L, t)[0];
  if (L === 0) for (const e of cs.emb) {
    const p = B(e.p); let s = 0n; x.forEach((v, i) => { s += mod(mod(v, p) * B(e.r[i]), p); });
    if (mod(s, p) !== B(e.digest)) return [false, "embedding@" + t];
  }
  const [kacc, vacc] = get("KV", L, t);
  const [xq, sh] = qrow(irms(x, g1, F));
  if (!(frRow(V.k, xq, kacc) && frRow(V.v, xq, vacc))) return [false, "kv@" + t];
  const ko = requant(V.k, kacc, sh), vo = requant(V.v, vacc, sh);
  const kt = []; for (let h = 0; h < KV; h++) kt.push(...rope(ko.slice(h * hd, (h + 1) * hd), cos, sin, F));
  const [krk, krv] = get("KR", L, t);
  if (krk.some((v, i) => v !== kt[i]) || krv.some((v, i) => v !== vo[i])) return [false, "kr@" + t];
  const qacc = get("Q", L, t)[0];
  if (!frRow(V.q, xq, qacc)) return [false, "q"];
  const qo = requant(V.q, qacc, sh);
  const INV = B(Math.round(65536 / Math.sqrt(hd))), out = [];
  for (let h = 0; h < H; h++) {
    const kh = Math.floor(h / (H / KV)); const [qq, sq] = qrow(rope(qo.slice(h * hd, (h + 1) * hd), cos, sin, F));
    const sc = [], vrows = [];
    for (let s = 0; s <= t; s++) {
      const [kr, vr] = get("KR", L, s); const [kk, sk] = qrow(kr.slice(kh * hd, (kh + 1) * hd));
      let dot = 0n; for (let j = 0; j < hd; j++) dot += qq[j] * kk[j];
      sc.push(shr(dot * INV, (F - sq - sk) + 16n)); vrows.push(vr.slice(kh * hd, (kh + 1) * hd));
    }
    const p = isoftmax(sc, T);
    for (let j = 0; j < hd; j++) { let a = 0n; for (let s = 0; s <= t; s++) a += p[s] * vrows[s][j]; out.push(shr(a, 20n)); }
  }
  const [oacc, gacc, uacc, dacc] = get("R", L, t);
  const [oq, so] = qrow(out);
  if (!frRow(V.o, oq, oacc)) return [false, "o"];
  const oo = requant(V.o, oacc, so); const x2 = x.map((v, i) => v + oo[i]);
  const [mq, sm] = qrow(irms(x2, g2, F));
  if (!(frRow(V.gate, mq, gacc) && frRow(V.up, mq, uacc))) return [false, "gate/up"];
  const gs = isilu(requant(V.gate, gacc, sm), T, ONE), us = requant(V.up, uacc, sm);
  const [dq, sd] = qrow(gs.map((v, i) => shr(v * us[i], F)));
  if (!frRow(V.down, dq, dacc)) return [false, "down"];
  const dn = requant(V.down, dacc, sd), xn = get("X", L + 1, t)[0];
  if (x2.some((v, i) => v + dn[i] !== xn[i])) return [false, "residual"];
  return [true, null];
}
async function runAll(url) {
  const data = await (await fetch(url)).json();
  const T = { EXP_T: big(data.tables.EXP_T), SILU_T: big(data.tables.SILU_T), EXP_STEP: B(data.tables.EXP_STEP), SILU_STEP: B(data.tables.SILU_STEP) };
  const rows = [];
  for (const cs of data.cases) {
    const t0 = performance.now(); let ok, why;
    try { [ok, why] = await verifyCell(cs, T); } catch (e) { ok = false; why = "exception: " + e.message; }
    rows.push({ label: cs.label, cell: cs.cell.join(","), js_ok: ok, js_reason: why, python_ok: cs.python_ok, agree: ok === cs.python_ok, ms: +(performance.now() - t0).toFixed(1) });
  }
  return rows;
}
