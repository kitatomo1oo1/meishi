// ウラガワ 顔の読み取りエンジン（ブラウザ用）
// 使い方：const eng = await createFaceEngine(videoElement); 毎フレーム eng.process() を呼ぶ
import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
import * as C from "./face-core.js";
export { C, FaceLandmarker };

const LIMITS = {
  yaw: 20, pitch: 20,         // これ以上顔が横・上下を向いたら読まない（度）
  dark: 60, bright: 235,      // 額の明るさ（0-255）
  minSize: 0.22,              // 顔の幅が画面の何割以上か
  motion: 0.025,              // 1フレームで動いてよい量（顔の幅に対する割合）
  spec: 0.3,                  // テカリの画素がこの割合を超えたら色を信用しない
  specular: 245, darkPix: 25, // 画素単位の除外
  hrConf: 0.38                // 心拍の確からしさがこれ未満なら「探し中」
};
export { LIMITS };

export async function createFaceEngine(video, { sampleWidth = 240 } = {}) {
  const fileset = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/wasm");
  const make = delegate => FaceLandmarker.createFromOptions(fileset, {
    baseOptions: {
      modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
      delegate },
    runningMode: "VIDEO", numFaces: 1,
    outputFaceBlendshapes: true, outputFacialTransformationMatrixes: true });
  let landmarker, delegate = "GPU";
  try { landmarker = await make("GPU"); }
  catch (e) { console.warn("GPUで失敗、CPUで再試行", e); landmarker = await make("CPU"); delegate = "CPU"; }

  const W = sampleWidth, H = Math.round(sampleWidth * video.videoHeight / video.videoWidth);
  const cv = document.createElement("canvas"); cv.width = W; cv.height = H;
  const cx = cv.getContext("2d", { willReadFrequently: true });

  const blink = new C.BlinkTracker();
  let lastVideoTime = -1, prevNose = null, last = null;
  let hrBuf = [], hrBadSince = null, hrLastCalc = 0, hrHist = [], hr = null;
  const log = [];            // 直近60秒のフレーム記録
  let baseline = null, baseCollect = null;

  // 凸包の中の肌色を平均（テカリと暗すぎる画素は除外）
  function sampleRegion(lm, idx, img) {
    const hull = C.convexHull(idx.map(i => [lm[i].x * W, lm[i].y * H]));
    let x0 = W, y0 = H, x1 = 0, y1 = 0;
    for (const [x, y] of hull) { x0 = Math.min(x0, x); y0 = Math.min(y0, y); x1 = Math.max(x1, x); y1 = Math.max(y1, y); }
    x0 = Math.max(0, Math.floor(x0)); y0 = Math.max(0, Math.floor(y0)); x1 = Math.min(W - 1, Math.ceil(x1)); y1 = Math.min(H - 1, Math.ceil(y1));
    let R = 0, G = 0, B = 0, n = 0, spec = 0, all = 0;
    for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
      if (!inside(hull, x + 0.5, y + 0.5)) continue;
      all++;
      const k = (y * W + x) * 4, r = img[k], g = img[k + 1], b = img[k + 2];
      if (r >= LIMITS.specular || g >= LIMITS.specular || b >= LIMITS.specular) { spec++; continue; }
      if (0.299 * r + 0.587 * g + 0.114 * b < LIMITS.darkPix) continue;
      R += r; G += g; B += b; n++;
    }
    return { hull, n, spec: all ? spec / all : 0, rgb: n ? { R: R / n, G: G / n, B: B / n } : null };
  }
  function inside(poly, x, y) {
    let c = false;
    for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
      const [xi, yi] = poly[i], [xj, yj] = poly[j];
      if ((yi > y) !== (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi) c = !c;
    }
    return c;
  }

  function process() {
    if (video.currentTime === lastVideoTime) return last;
    lastVideoTime = video.currentTime;
    const t = performance.now();
    const res = landmarker.detectForVideo(video, t);
    const lm = res.faceLandmarks && res.faceLandmarks[0];
    if (!lm) {
      prevNose = null; markHrBad(t);
      last = { t, face: false, quality: { ok: false, reason: "顔が見つかりません" } };
      push(last); return last;
    }
    // 顔の向き・大きさ・動き
    const pose = res.facialTransformationMatrixes?.[0] ? C.poseFromMatrix(res.facialTransformationMatrixes[0].data) : { yaw: 0, pitch: 0, roll: 0 };
    const faceW = Math.abs(lm[454].x - lm[234].x);
    const nose = [lm[1].x, lm[1].y];
    const motion = prevNose ? Math.hypot(nose[0] - prevNose[0], nose[1] - prevNose[1]) / (faceW + 1e-9) : 0;
    prevNose = nose;

    // 色
    cx.drawImage(video, 0, 0, W, H);
    const img = cx.getImageData(0, 0, W, H).data;
    const cL = sampleRegion(lm, C.ROI.cheekL, img), cR = sampleRegion(lm, C.ROI.cheekR, img), fh = sampleRegion(lm, C.ROI.forehead, img);
    const avg = (...rs) => { const v = rs.filter(r => r.rgb); if (!v.length) return null;
      const w = v.reduce((s, r) => s + r.n, 0);
      return { R: v.reduce((s, r) => s + r.rgb.R * r.n, 0) / w, G: v.reduce((s, r) => s + r.rgb.G * r.n, 0) / w, B: v.reduce((s, r) => s + r.rgb.B * r.n, 0) / w }; };
    const cheek = avg(cL, cR), skin = avg(cL, cR, fh), fore = fh.rgb;
    const light = fore ? 0.299 * fore.R + 0.587 * fore.G + 0.114 * fore.B : null;
    const spec = (cL.spec + cR.spec + fh.spec) / 3;

    // 品質
    let reason = null;
    if (Math.abs(pose.yaw) > LIMITS.yaw || Math.abs(pose.pitch) > LIMITS.pitch) reason = "顔を正面に向けてください";
    else if (faceW < LIMITS.minSize) reason = "もう少し近づいてください";
    else if (light == null || light < LIMITS.dark) reason = "暗いです";
    else if (light > LIMITS.bright) reason = "明るすぎます";
    else if (motion > LIMITS.motion) reason = "動かないでください";
    else if (spec > LIMITS.spec) reason = "顔のテカリが強いです";
    const ok = !reason;

    // 目
    const P = idx => idx.map(i => [lm[i].x * video.videoWidth, lm[i].y * video.videoHeight]);
    const ear = (C.eyeAspect(P(C.EYE_L)) + C.eyeAspect(P(C.EYE_R))) / 2;
    const poseOk = Math.abs(pose.yaw) <= LIMITS.yaw && Math.abs(pose.pitch) <= LIMITS.pitch;
    const blinkEv = blink.push(t, ear, poseOk);

    // 表情
    const bs = {};
    (res.faceBlendshapes?.[0]?.categories || []).forEach(c => { bs[c.categoryName] = c.score; });
    const smile = C.smileReading(bs);

    // 心拍
    if (ok && skin) { hrBadSince = null; hrBuf.push({ t, ...skin }); }
    else markHrBad(t);
    while (hrBuf.length && t - hrBuf[0].t > 12000) hrBuf.shift();
    if (t - hrLastCalc > 1000) {
      hrLastCalc = t;
      const span = hrBuf.length ? hrBuf[hrBuf.length - 1].t - hrBuf[0].t : 0;
      const est = span >= 8000 ? C.heartRate(hrBuf.filter(s => s.t >= t - 10000)) : null;
      if (est && est.confidence >= LIMITS.hrConf) { hrHist.push(est.bpm); if (hrHist.length > 5) hrHist.shift(); }
      hr = { bpm: hrHist.length >= 2 ? C.median(hrHist) : null, raw: est, warming: span < 8000, span };
    }

    last = {
      t, face: true, lm, pose, faceW, motion, light, spec,
      quality: { ok, reason },
      rgb: { cheek, fore, skin },
      red: {
        cheek: cheek ? C.redness(cheek) : null,
        fore: fore ? C.redness(fore) : null,
        diff: cheek && fore ? C.redness(cheek) - C.redness(fore) : null
      },
      hulls: { cheekL: cL.hull, cheekR: cR.hull, forehead: fh.hull, W, H },
      ear, eyeRel: blink.rel(ear), blinkEv, blinkClosed: blink.closedAt !== null,
      smile, bs, hr
    };
    push(last);
    return last;
  }
  function markHrBad(t) {
    if (hrBadSince === null) hrBadSince = t;
    if (t - hrBadSince > 600) { hrBuf = []; hrHist = []; }   // 0.6秒以上読めなかったら心拍は測り直し
  }
  function push(f) {
    const slim = f.face ? { t: f.t, ok: f.quality.ok, red: f.red, ear: f.ear, light: f.light, smile: f.smile, hr: f.hr?.bpm ?? null, blink: f.blinkEv } : { t: f.t, ok: false };
    log.push(slim);
    while (log.length && f.t - log[0].t > 60000) log.shift();
    if (baseCollect) {
      if (f.t >= baseCollect.until) { baseline = summarize(baseCollect.from, f.t); const r = baseCollect.resolve; baseCollect = null; r(baseline); }
    }
  }

  // 期間のまとめ（品質OKのフレームだけで計算）
  function summarize(from, to = performance.now()) {
    const fr = log.filter(f => f.t >= from && f.t <= to);
    const ok = fr.filter(f => f.ok);
    const pick = fn => ok.map(fn).filter(v => v != null);
    const blinks = fr.filter(f => f.blink).map(f => f.blink.dur);
    const sec = (to - from) / 1000;
    const smiles = ok.filter(f => f.smile.smiling);
    return {
      sec: +sec.toFixed(1),
      okRatio: fr.length ? ok.length / fr.length : 0,
      redCheek: C.median(pick(f => f.red.cheek)),
      redFore: C.median(pick(f => f.red.fore)),
      redDiff: C.median(pick(f => f.red.diff)),
      ear: C.median(pick(f => f.ear)),
      light: C.median(pick(f => f.light)),
      blinkN: blinks.length,
      blinkPerMin: sec > 0 ? blinks.length / sec * 60 : null,
      blinkMs: C.median(blinks),
      smileRatio: ok.length ? smiles.length / ok.length : 0,
      smileGenuine: C.median(smiles.map(f => f.smile.genuine)),
      hr: C.median(pick(f => f.hr))
    };
  }
  function startBaseline(ms = 10000) {
    const now = performance.now();
    return new Promise(resolve => { baseCollect = { from: now, until: now + ms, resolve }; });
  }
  function diff(s, b = baseline) {
    if (!b) return null;
    const d = (k) => s[k] != null && b[k] != null ? s[k] - b[k] : null;
    return { redCheek: d("redCheek"), redDiff: d("redDiff"), ear: d("ear"), blinkPerMin: d("blinkPerMin"), blinkMs: d("blinkMs"), hr: d("hr"), light: d("light") };
  }

  return { process, summarize, startBaseline, diff, get baseline() { return baseline; }, delegate, landmarker, blink };
}
