// ウラガワ 顔の読み取り：計算部分（カメラもAIも使わない純粋な計算だけ。テスト可能）

// ---------- 顔の領域（MediaPipe 468点の番号）----------
// 点の並び順に依存しないよう、使うときは凸包にして塗る
export const ROI = {
  cheekL:   [116, 117, 118, 119, 100, 142, 203, 206, 216, 192, 213, 147, 123, 50, 101, 36, 205, 187],
  cheekR:   [345, 346, 347, 348, 329, 371, 423, 426, 436, 416, 433, 376, 352, 280, 330, 266, 425, 411],
  forehead: [10, 109, 67, 103, 338, 297, 332, 151, 108, 69, 337, 299, 9]
};
export const EYE_L = [33, 160, 158, 133, 153, 144];   // 外端, 上2点, 内端, 下2点
export const EYE_R = [362, 385, 387, 263, 373, 380];

// ---------- 幾何 ----------
export function convexHull(pts) {
  const p = pts.map(q => [q[0], q[1]]).sort((a, b) => a[0] - b[0] || a[1] - b[1]);
  if (p.length < 3) return p;
  const cross = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
  const lo = [], up = [];
  for (const q of p) { while (lo.length >= 2 && cross(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
  for (let i = p.length - 1; i >= 0; i--) { const q = p[i]; while (up.length >= 2 && cross(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
  up.pop(); lo.pop();
  return lo.concat(up);
}
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

// 目の開き（EAR）：縦の開き ÷ 横幅。顔の大きさに左右されない
export function eyeAspect(P) {
  return (dist(P[1], P[5]) + dist(P[2], P[4])) / (2 * dist(P[0], P[3]) + 1e-9);
}

// 顔の向き（度）。MediaPipe の 4x4 変換行列（列優先）から
export function poseFromMatrix(m) {
  const R = (r, c) => m[c * 4 + r];
  const yaw = Math.asin(Math.max(-1, Math.min(1, -R(2, 0)))) * 180 / Math.PI;
  const pitch = Math.atan2(R(2, 1), R(2, 2)) * 180 / Math.PI;
  const roll = Math.atan2(R(1, 0), R(0, 0)) * 180 / Math.PI;
  return { yaw, pitch, roll };
}

// ---------- 色 ----------
// 肌の色を正規化：明るさの影響を消して赤みだけを見る
export function chroma(c) {
  const s = c.R + c.G + c.B + 1e-9;
  return { r: c.R / s, g: c.G / s, b: c.B / s };
}
export const redness = c => { const k = chroma(c); return k.r - k.g; };

// ---------- 統計 ----------
export const mean = a => a.length ? a.reduce((s, v) => s + v, 0) / a.length : null;
export function std(a) { const m = mean(a); return a.length ? Math.sqrt(mean(a.map(v => (v - m) ** 2))) : 0; }
export function quantile(a, q) {
  if (!a.length) return null;
  const s = [...a].sort((x, y) => x - y), i = (s.length - 1) * q, lo = Math.floor(i), hi = Math.ceil(i);
  return s[lo] + (s[hi] - s[lo]) * (i - lo);
}
export function median(a) { return quantile(a, 0.5); }

// ---------- まばたき：その人の目の大きさに合わせて判定 ----------
// 目が細い人・大きい人で固定のしきい値は外れるので、直近の「開いているときの値」を基準にする
export class BlinkTracker {
  constructor({ windowMs = 6000, closeRatio = 0.62, openRatio = 0.78, minMs = 40, maxMs = 1500 } = {}) {
    Object.assign(this, { windowMs, closeRatio, openRatio, minMs, maxMs });
    this.hist = []; this.closedAt = null; this.blinks = []; this.lastT = null;
  }
  openLevel() { return quantile(this.hist.map(h => h.v), 0.85); }
  push(t, ear, reliable = true) {
    if (!reliable) { this.closedAt = null; return null; }   // 顔の向きが大きいときは数えない
    this.hist.push({ t, v: ear });
    while (this.hist.length && t - this.hist[0].t > this.windowMs) this.hist.shift();
    const open = this.openLevel();
    if (this.hist.length < 15 || !open) return null;        // 基準が溜まるまでは判定しない
    const rel = ear / open;
    let ev = null;
    if (this.closedAt === null && rel < this.closeRatio) this.closedAt = t;
    else if (this.closedAt !== null && rel > this.openRatio) {
      const dur = t - this.closedAt;
      if (dur >= this.minMs && dur <= this.maxMs) { ev = { t: this.closedAt, dur }; this.blinks.push(ev); }
      this.closedAt = null;
    }
    this.lastT = t;
    return ev;
  }
  rel(ear) { const o = this.openLevel(); return o ? ear / o : null; }
  reset() { this.blinks = []; }
}

// ---------- 顔の色から心拍（POS法：Wang et al. 2017）----------
// samples: [{t(ms), R, G, B}]。窓の長さは 8〜12秒が目安
export function resample(samples, fs) {
  if (samples.length < 2) return null;
  const t0 = samples[0].t, t1 = samples[samples.length - 1].t;
  const n = Math.floor((t1 - t0) / 1000 * fs);
  const out = { R: new Float64Array(n), G: new Float64Array(n), B: new Float64Array(n) };
  let j = 0;
  for (let i = 0; i < n; i++) {
    const t = t0 + i * 1000 / fs;
    while (j < samples.length - 2 && samples[j + 1].t < t) j++;
    const a = samples[j], b = samples[j + 1], w = b.t === a.t ? 0 : (t - a.t) / (b.t - a.t);
    out.R[i] = a.R + (b.R - a.R) * w; out.G[i] = a.G + (b.G - a.G) * w; out.B[i] = a.B + (b.B - a.B) * w;
  }
  return out;
}

export function posSignal(C, fs, winSec = 1.6) {
  const n = C.R.length, l = Math.round(winSec * fs), H = new Float64Array(n);
  for (let s = 0; s + l <= n; s++) {
    let mR = 0, mG = 0, mB = 0;
    for (let k = s; k < s + l; k++) { mR += C.R[k]; mG += C.G[k]; mB += C.B[k]; }
    mR /= l; mG /= l; mB /= l;
    const S1 = new Float64Array(l), S2 = new Float64Array(l);
    for (let k = 0; k < l; k++) {
      const r = C.R[s + k] / mR, g = C.G[s + k] / mG, b = C.B[s + k] / mB;
      S1[k] = g - b; S2[k] = g + b - 2 * r;
    }
    const a = std(Array.from(S1)) / (std(Array.from(S2)) + 1e-12);
    let hm = 0; const h = new Float64Array(l);
    for (let k = 0; k < l; k++) { h[k] = S1[k] + a * S2[k]; hm += h[k]; }
    hm /= l;
    for (let k = 0; k < l; k++) H[s + k] += h[k] - hm;
  }
  return H;
}

// 0.7〜3.0Hz（42〜180拍/分）でいちばん強い周波数を探す
export function dominantBpm(x, fs, lo = 0.7, hi = 3.0) {
  const n = x.length; if (n < fs * 4) return null;
  const m = mean(Array.from(x)); const y = Array.from(x, v => v - m);
  for (let i = 0; i < n; i++) y[i] *= 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1)); // ハン窓
  let best = 0, bestF = 0, total = 0; const spec = [];
  for (let f = lo; f <= hi + 1e-9; f += 0.01) {
    let re = 0, im = 0; const w = 2 * Math.PI * f / fs;
    for (let i = 0; i < n; i++) { re += y[i] * Math.cos(w * i); im -= y[i] * Math.sin(w * i); }
    const p = re * re + im * im; spec.push(p); total += p;
    if (p > best) { best = p; bestF = f; }
  }
  // 確からしさ：ピークの周り（±0.1Hz）に力がどれだけ集まっているか
  let near = 0; spec.forEach((p, i) => { if (Math.abs(lo + i * 0.01 - bestF) <= 0.1) near += p; });
  return { bpm: bestF * 60, confidence: total ? near / total : 0 };
}

export function heartRate(samples, fs = 30) {
  const C = resample(samples, fs); if (!C || C.R.length < fs * 6) return null;
  return dominantBpm(posSignal(C, fs), fs);
}

// ---------- 笑顔の本気度 ----------
// 本気の笑顔は目の周り（頬の持ち上がり・目の細まり）まで動く。作り笑いは口だけ
export function smileReading(bs) {
  const g = n => bs[n] || 0;
  const mouth = (g("mouthSmileLeft") + g("mouthSmileRight")) / 2;
  const eyes = (g("cheekSquintLeft") + g("cheekSquintRight") + g("eyeSquintLeft") + g("eyeSquintRight")) / 4;
  return { mouth, eyes, smiling: mouth > 0.35, genuine: mouth > 0.35 ? Math.min(1, eyes / (mouth * 0.6)) : null };
}
