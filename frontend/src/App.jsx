// ═══════════════════════════════════════════════════════════════════════════
//  SG JEWELS · App.jsx  v8 — Production Grade
//  Earring + Necklace Real-Time Try-On with Smart Recommendations
// ═══════════════════════════════════════════════════════════════════════════

import React, { useEffect, useRef, useState, useCallback } from "react";
import { FaceMesh } from "@mediapipe/face_mesh";
import { Camera } from "@mediapipe/camera_utils";
import "./styles.css";

const BACKEND_URL = "http://localhost:8000/process-image";

// ─── Adaptive smoothing (EMA w/ velocity-based alpha + jump snap) ────────
// Why this replaces the old moving-average buffer:
//   • Old: avg over 18–22 frames + 4 px/frame velocity clamp → jewellery
//     visibly "catches up" to the ear when you turn your head fast.
//   • New: One-Euro-style filter — alpha rises with velocity so fast
//     movement is followed instantly, but tiny jitter is still smoothed.
//   • Big position jumps (new person enters frame, tracking flips) trigger
//     an instant snap instead of slow interpolation.
const JUMP_THRESHOLD = 90;    // px — beyond this, snap to new position
const EMA_MIN        = 0.22;  // smoothing factor when face is nearly still
const EMA_MAX        = 0.70;  // smoothing factor when moving fast
const VEL_FULL       = 26;    // px/frame velocity that yields EMA_MAX

let lState = null, rState = null, nState = null;
let noFaceFrames = 0;

function smoothPoint(state, pt) {
  if (!state) return { x: pt.x, y: pt.y };
  const dx = pt.x - state.x;
  const dy = pt.y - state.y;
  const d  = Math.hypot(dx, dy);

  // Big jump → likely a new face / tracking reset → snap instantly
  if (d > JUMP_THRESHOLD) return { x: pt.x, y: pt.y };

  // Velocity-adaptive alpha
  const t = Math.min(1, d / VEL_FULL);
  const alpha = EMA_MIN + (EMA_MAX - EMA_MIN) * t;
  return {
    x: state.x + dx * alpha,
    y: state.y + dy * alpha,
  };
}

function resetSmoothing() {
  lState = null;
  rState = null;
  nState = null;
}

// ═══════════════════════════════════════════════════════════════════════════
//  GEOMETRY
// ═══════════════════════════════════════════════════════════════════════════
function faceWidth (lm, W) { return Math.abs(lm[454].x * W - lm[234].x * W); }
function faceHeight(lm, H) { return Math.abs(lm[10].y  * H - lm[152].y * H); }

function getHeadTilt(lm, W, H) {
  return Math.atan2(
    lm[263].y * H - lm[33].y * H,
    lm[263].x * W - lm[33].x * W
  );
}

// ─── EAR ANCHOR (anatomically correct, eye-corner based) ────────────────
function earAnchor(lm, side, W, H) {
  const fw      = faceWidth(lm, W);
  const fh      = faceHeight(lm, H);
  const avgEyeY = ((lm[33].y + lm[263].y) / 2) * H;
  const lobeY   = avgEyeY + fh * 0.20;   // earlobe is anatomically here

  if (side === "L") {
    return { x: lm[234].x * W - fw * 0.03, y: lobeY };
  }
  return { x: lm[454].x * W + fw * 0.03, y: lobeY };
}

// ─── NECKLACE ANCHOR (neck top, just below chin) ────────────────────────
function necklaceAnchor(lm, W, H) {
  const fw      = faceWidth(lm, W);
  const fh      = faceHeight(lm, H);
  const chinY   = lm[152].y * H;
  const centreX = (lm[234].x * W + lm[454].x * W) / 2;
  const topY    = chinY + fh * 0.12;     // 12% below chin = neck top
  return { centreX, topY, neckWidthBase: fw * 1.45 };
}

// ─── OPACITY (head rotation occlusion) ──────────────────────────────────
function yawOpacity(lm, W) {
  const noseX  = lm[4].x * W;
  const lx     = lm[234].x * W, rx = lm[454].x * W;
  const center = (lx + rx) / 2;
  const fw     = Math.abs(rx - lx);
  if (fw < 1) return { L: 1, R: 1, N: 1 };
  const norm = (noseX - center) / fw;
  return {
    L: Math.max(0, Math.min(1, (norm + 0.32) / 0.18)),
    R: Math.max(0, Math.min(1, (0.32 - norm) / 0.18)),
    N: 1,
  };
}

function pitchOpacity(lm, H) {
  const topY = lm[10].y * H, noseY = lm[4].y * H, chinY = lm[152].y * H;
  const fh = chinY - topY;
  if (fh < 1) return 1;
  const ratio = (noseY - topY) / fh;
  const tiltDown = Math.max(0, (ratio - 0.60) / 0.10);
  return Math.max(0, 1 - tiltDown);
}

// ═══════════════════════════════════════════════════════════════════════════
//  CANVAS RENDERING (with metallic shine)
// ═══════════════════════════════════════════════════════════════════════════

function drawEarring(ctx, img, cx, cy, w, h, angle, alpha, flipH) {
  if (alpha <= 0.01 || !img) return;
  ctx.save();
  ctx.globalAlpha = Math.min(0.96, alpha);

  // Gold glow shadow
  ctx.shadowColor   = "rgba(255, 200, 60, 0.45)";
  ctx.shadowBlur    = 14;
  ctx.shadowOffsetX = 0;
  ctx.shadowOffsetY = 2;

  ctx.translate(cx, cy);
  ctx.rotate(angle);
  if (flipH) ctx.scale(-1, 1);

  // Pass 1 — base draw, top edge at anchor
  ctx.drawImage(img, -w / 2, 0, w, h);

  // Pass 2 — metallic shine via screen blend
  ctx.globalCompositeOperation = "screen";
  ctx.globalAlpha = Math.min(0.18, alpha * 0.20);
  ctx.shadowBlur = 0;
  ctx.drawImage(img, -w / 2, 0, w, h);

  ctx.globalCompositeOperation = "source-over";
  ctx.restore();
}

function drawNecklace(ctx, img, centreX, topY, neckW, alpha) {
  if (alpha <= 0.01 || !img) return;
  const aspect = img.naturalHeight / img.naturalWidth || 1;
  const neckH = neckW * aspect;

  ctx.save();
  ctx.globalAlpha = Math.min(0.97, alpha);

  ctx.shadowColor   = "rgba(255, 195, 70, 0.55)";
  ctx.shadowBlur    = 18;
  ctx.shadowOffsetY = 3;

  ctx.drawImage(img, centreX - neckW / 2, topY, neckW, neckH);

  // Shine pass
  ctx.globalCompositeOperation = "screen";
  ctx.globalAlpha = Math.min(0.20, alpha * 0.22);
  ctx.shadowBlur = 0;
  ctx.drawImage(img, centreX - neckW / 2, topY, neckW, neckH);

  ctx.globalCompositeOperation = "source-over";
  ctx.restore();
}

// ═══════════════════════════════════════════════════════════════════════════
//  FACE SHAPE — 7 categories with stabilisation
// ═══════════════════════════════════════════════════════════════════════════

function classifyFaceShape(lm, W, H) {
  const fw = faceWidth(lm, W);
  const fh = faceHeight(lm, H);
  if (fw < 10) return null;

  const ratio     = fh / fw;
  const jawW      = Math.abs(lm[172].x * W - lm[397].x * W);
  const foreheadW = Math.abs(lm[103].x * W - lm[332].x * W);
  const cheekW    = fw;
  const chinW     = Math.abs(lm[140].x * W - lm[369].x * W);

  const jawRatio   = jawW / fw;
  const fhRatio    = foreheadW / fw;
  const chinRatio  = chinW / fw;

  // Diamond: cheekbones widest, narrow forehead and jaw
  if (cheekW > foreheadW * 1.10 && cheekW > jawW * 1.10 && ratio > 1.20)
    return "Diamond";

  // Heart: wide forehead, narrow chin
  if (ratio > 1.20 && foreheadW > jawW * 1.08 && chinRatio < 0.55)
    return "Heart";

  // Triangle: narrow forehead, wide jaw (inverse heart)
  if (ratio > 1.20 && jawW > foreheadW * 1.10)
    return "Triangle";

  // Oblong / Rectangle: very long with straight sides
  if (ratio > 1.50 && jawRatio < 0.85)
    return "Oblong";

  // Square: nearly equal H/W with strong jaw
  if (ratio < 1.20 && jawRatio > 0.85)
    return "Square";

  // Round: moderately tall, soft jaw, balanced
  if (ratio < 1.35 && jawRatio < 0.85 && jawRatio > 0.70)
    return "Round";

  // Default: Oval
  return "Oval";
}

// ─── Stabilisation: majority vote over last 15 classifications ──────────
const SHAPE_HISTORY_SIZE = 15;
const SHAPE_VOTE_THRESHOLD = 9;
let shapeHistory = [];

function stableFaceShape(newShape) {
  if (newShape) shapeHistory.push(newShape);
  if (shapeHistory.length > SHAPE_HISTORY_SIZE) shapeHistory.shift();

  const counts = {};
  shapeHistory.forEach(s => { counts[s] = (counts[s] || 0) + 1; });
  let winner = null, maxCount = 0;
  for (const s in counts) {
    if (counts[s] > maxCount) { maxCount = counts[s]; winner = s; }
  }
  return maxCount >= SHAPE_VOTE_THRESHOLD ? winner : null;
}

// ─── Recommendation knowledge base (earring + necklace per shape) ───────
const SHAPE_DATA = {
  Oval: {
    emoji: "🥚",
    color: "#7a9e7e",
    tip: "Balanced proportions — almost any style works for you.",
    earring: "Chandelier or Drop Earrings",
    necklace: "Long Pendant or Princess-length Necklace",
  },
  Round: {
    emoji: "⭕",
    color: "#9e7a7a",
    tip: "Vertical lines elongate your face beautifully.",
    earring: "Long Drop or Linear Earrings",
    necklace: "V-shape Pendant or Long Chain",
  },
  Square: {
    emoji: "◼️",
    color: "#7a879e",
    tip: "Curves soften your strong jawline.",
    earring: "Hoop or Round Drop Earrings",
    necklace: "Round Pendant or Curved Collar",
  },
  Heart: {
    emoji: "🫀",
    color: "#9e7a94",
    tip: "Wider base styles balance your forehead.",
    earring: "Teardrop or Inverted Triangle",
    necklace: "Choker or Short Statement Necklace",
  },
  Diamond: {
    emoji: "💎",
    color: "#94739e",
    tip: "Highlight your cheekbones with elegance.",
    earring: "Stud or Small Hoop Earrings",
    necklace: "Princess-length with Pendant",
  },
  Oblong: {
    emoji: "📏",
    color: "#9e9a7a",
    tip: "Add width with horizontal-emphasis pieces.",
    earring: "Wide Stud or Cluster Earrings",
    necklace: "Choker or Layered Short Necklace",
  },
  Triangle: {
    emoji: "🔻",
    color: "#7a9e94",
    tip: "Top-heavy designs balance your wider jaw.",
    earring: "Chandelier or Wide-top Earrings",
    necklace: "Statement Collar or Bib Necklace",
  },
};

// ═══════════════════════════════════════════════════════════════════════════
//  APP COMPONENT
// ═══════════════════════════════════════════════════════════════════════════

export default function App() {
  const videoRef    = useRef(null);
  const canvasRef   = useRef(null);
  const earringRef  = useRef(null);
  const necklaceRef = useRef(null);
  const scaleRef    = useRef(1.0);
  const necklaceScaleRef = useRef(1.0);
  const offsetYRef  = useRef(0);
  const modeRef     = useRef("earring");

  const [earring,         setEarring]         = useState(null);
  const [necklace,        setNecklace]        = useState(null);
  const [activeMode,      setActiveMode]      = useState("earring");
  const [scale,           setScale]           = useState(1.0);
  const [necklaceScale,   setNecklaceScale]   = useState(1.0);
  const [offsetY,         setOffsetY]         = useState(0);
  const [earStatus,       setEarStatus]       = useState("idle");
  const [neckStatus,      setNeckStatus]      = useState("idle");
  const [earMsg,          setEarMsg]          = useState("");
  const [neckMsg,         setNeckMsg]         = useState("");
  const [faceShape,       setFaceShape]       = useState(null);
  const [captureURL,      setCaptureURL]      = useState(null);
  const [showModal,       setShowModal]       = useState(false);
  const [cameraReady,     setCameraReady]     = useState(false);
  const [faceDetected,    setFaceDetected]    = useState(false);

  useEffect(() => { earringRef.current  = earring;       }, [earring]);
  useEffect(() => { necklaceRef.current = necklace;      }, [necklace]);
  useEffect(() => { scaleRef.current    = scale;         }, [scale]);
  useEffect(() => { necklaceScaleRef.current = necklaceScale; }, [necklaceScale]);
  useEffect(() => { offsetYRef.current  = offsetY;       }, [offsetY]);
  useEffect(() => { modeRef.current     = activeMode;    }, [activeMode]);

  // ─── FaceMesh Loop ─────────────────────────────────────────────
  useEffect(() => {
    const mesh = new FaceMesh({
      locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`,
    });
    mesh.setOptions({
      maxNumFaces: 1,
      refineLandmarks: true,
      minDetectionConfidence: 0.65,
      minTrackingConfidence:  0.65,
    });

    mesh.onResults(res => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");

      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(res.image, 0, 0, canvas.width, canvas.height);

      const hasLM = res.multiFaceLandmarks?.length > 0;
      setFaceDetected(hasLM);

      // No face for >5 frames → reset smoothing so next person snaps in cleanly
      if (!hasLM) {
        noFaceFrames++;
        if (noFaceFrames > 5) resetSmoothing();
        return;
      }
      noFaceFrames = 0;

      const lm = res.multiFaceLandmarks[0];
      const W = canvas.width, H = canvas.height;
      const fw = faceWidth(lm, W);
      const tilt = getHeadTilt(lm, W, H);
      const yaw = yawOpacity(lm, W);
      const pitch = pitchOpacity(lm, H);
      const mode = modeRef.current;

      // Face shape — every 5 frames, with stabilisation
      App._fc = ((App._fc || 0) + 1);
      if (App._fc % 5 === 0) {
        const raw = classifyFaceShape(lm, W, H);
        const stable = stableFaceShape(raw);
        if (stable) setFaceShape(stable);
      }

      // ── EARRINGS ─────────────────────────────────────────
      if ((mode === "earring" || mode === "both") && earringRef.current) {
        const ew = fw * 0.19 * scaleRef.current;
        const eh = ew * 2.2;
        const rawL = earAnchor(lm, "L", W, H);
        const rawR = earAnchor(lm, "R", W, H);
        const ancL = (lState = smoothPoint(lState, rawL));
        const ancR = (rState = smoothPoint(rState, rawR));

        drawEarring(ctx, earringRef.current, ancL.x, ancL.y, ew, eh,
                    tilt, yaw.L * pitch, false);
        drawEarring(ctx, earringRef.current, ancR.x, ancR.y, ew, eh,
                    tilt, yaw.R * pitch, true);
      }

      // ── NECKLACE ─────────────────────────────────────────
      if ((mode === "necklace" || mode === "both") && necklaceRef.current) {
        const { centreX, topY, neckWidthBase } = necklaceAnchor(lm, W, H);
        const adjustedY = topY + offsetYRef.current;
        const rawN = { x: centreX, y: adjustedY };
        const ancN = (nState = smoothPoint(nState, rawN));
        const neckW = neckWidthBase * necklaceScaleRef.current;

        drawNecklace(ctx, necklaceRef.current, ancN.x, ancN.y,
                     neckW, yaw.N * pitch);
      }
    });

    const cam = new Camera(videoRef.current, {
      onFrame: async () => { await mesh.send({ image: videoRef.current }); },
      width: 640, height: 480,
    });
    cam.start().then(() => setCameraReady(true));
    return () => { cam.stop(); mesh.close(); };
  }, []);

  // ─── Upload Handlers ────────────────────────────────────────────
  const uploadJewellery = async (file, type, setStatus, setMsg) => {
    if (!file.type.startsWith("image/")) {
      setStatus("error"); setMsg("Please upload a valid image."); return null;
    }
    setStatus("loading"); setMsg("Processing…");
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("type", type);
      const res = await fetch(BACKEND_URL, { method: "POST", body: form });
      const ct = res.headers.get("content-type") || "";
      if (!res.ok || ct.includes("application/json")) {
        const json = await res.json();
        setStatus("error"); setMsg(json.error || "Server error."); return null;
      }
      const blob = await res.blob();
      const img = new Image();
      return new Promise((resolve) => {
        img.onload  = () => { setStatus("loaded"); setMsg(`${type === "necklace" ? "Necklace" : "Earring"} ready ✓`); resolve(img); };
        img.onerror = () => { setStatus("error"); setMsg("Could not load image."); resolve(null); };
        img.src = URL.createObjectURL(blob);
      });
    } catch {
      setStatus("error"); setMsg("Backend not reachable."); return null;
    }
  };

  const handleEarringFile = useCallback(async (e) => {
    const file = e.target.files?.[0]; if (!file) return;
    const img = await uploadJewellery(file, "earring", setEarStatus, setEarMsg);
    if (img) { lState = null; rState = null; setEarring(img); }
  }, []);

  const handleNecklaceFile = useCallback(async (e) => {
    const file = e.target.files?.[0]; if (!file) return;
    const img = await uploadJewellery(file, "necklace", setNeckStatus, setNeckMsg);
    if (img) { nState = null; setNecklace(img); }
  }, []);

  // ─── Capture & Download ─────────────────────────────────────────
  const handleCapture = useCallback(() => {
    const c = canvasRef.current; if (!c) return;
    const ec = document.createElement("canvas");
    ec.width = 1920; ec.height = 1440;
    const ctx = ec.getContext("2d");
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.drawImage(c, 0, 0, ec.width, ec.height);
    setCaptureURL(ec.toDataURL("image/png", 1.0));
    setShowModal(true);
  }, []);

  const handleDownload = useCallback(() => {
    if (!captureURL) return;
    const a = document.createElement("a");
    a.download = `sgjewels-tryon-${Date.now()}.png`;
    a.href = captureURL; a.click();
  }, [captureURL]);

  // ─── UI Helpers ─────────────────────────────────────────────────
  const shapeData = faceShape ? SHAPE_DATA[faceShape] : null;
  const cfg = (st, msg, fallback) => ({
    idle:    { cls: "",        label: fallback },
    loading: { cls: "loading", label: msg || "Processing…" },
    loaded:  { cls: "success", label: msg },
    error:   { cls: "error",   label: msg },
  })[st];
  const earCfg  = cfg(earStatus,  earMsg,  "Upload an earring");
  const neckCfg = cfg(neckStatus, neckMsg, "Upload a necklace");

  // ════════════════════════════════════════════════════════════════
  return (
    <div className="sg-app">

      <header className="sg-header">
        <div className="sg-header__brand">
          <span className="sg-header__logo">✦</span>
          <div>
            <h1 className="sg-header__title">SG Jewels</h1>
            <p className="sg-header__sub">Virtual Try-On Studio</p>
          </div>
        </div>
        <div className="sg-header__badges">
          <span className={`sg-badge ${cameraReady ? "sg-badge--on" : ""}`}>
            {cameraReady ? "Camera Live" : "Starting…"}
          </span>
          {faceDetected && <span className="sg-badge sg-badge--face">Face Detected</span>}
        </div>
      </header>

      <main className="sg-main">

        <aside className="sg-panel">

          <section className="sg-card sg-card--tabs">
            <p className="sg-card__label">Try-On Mode</p>
            <div className="sg-tabs">
              {[
                { id: "earring",  icon: "💎", label: "Earrings" },
                { id: "necklace", icon: "📿", label: "Necklace" },
                { id: "both",     icon: "✨", label: "Both" },
              ].map(t => (
                <button key={t.id}
                  className={`sg-tab ${activeMode === t.id ? "sg-tab--active" : ""}`}
                  onClick={() => setActiveMode(t.id)}>
                  {t.icon} {t.label}
                </button>
              ))}
            </div>
          </section>

          {(activeMode === "earring" || activeMode === "both") && (
            <section className="sg-card">
              <p className="sg-card__label">Upload Earring</p>
              <label className="sg-upload">
                <input type="file" accept="image/*" onChange={handleEarringFile} />
                <span className="sg-upload__icon">💎</span>
                <span className="sg-upload__text">Choose Earring</span>
              </label>
              {earStatus !== "idle" && (
                <p className={`sg-status sg-status--${earCfg.cls}`}>{earCfg.label}</p>
              )}
              <p className="sg-hint">Single or pair — auto-extracts one</p>
            </section>
          )}

          {(activeMode === "necklace" || activeMode === "both") && (
            <section className="sg-card">
              <p className="sg-card__label">Upload Necklace</p>
              <label className="sg-upload sg-upload--necklace">
                <input type="file" accept="image/*" onChange={handleNecklaceFile} />
                <span className="sg-upload__icon">📿</span>
                <span className="sg-upload__text">Choose Necklace</span>
              </label>
              {neckStatus !== "idle" && (
                <p className={`sg-status sg-status--${neckCfg.cls}`}>{neckCfg.label}</p>
              )}
              <p className="sg-hint">Chain ends auto-removed, design isolated</p>
            </section>
          )}

          {/* ── SIZE CONTROLS ────────────────────────────── */}
          {(activeMode === "earring" || activeMode === "both") && earring && (
            <section className="sg-card">
              <p className="sg-card__label">Earring Size</p>
              <div className="sg-slider-row">
                <span className="sg-slider-val">{scale.toFixed(2)}×</span>
                <input className="sg-slider" type="range"
                  min="0.5" max="1.8" step="0.05"
                  value={scale} onChange={e => setScale(+e.target.value)} />
              </div>
            </section>
          )}

          {(activeMode === "necklace" || activeMode === "both") && necklace && (
            <section className="sg-card">
              <p className="sg-card__label">Necklace Size</p>
              <div className="sg-slider-row">
                <span className="sg-slider-val">{necklaceScale.toFixed(2)}×</span>
                <input className="sg-slider" type="range"
                  min="0.6" max="1.6" step="0.05"
                  value={necklaceScale} onChange={e => setNecklaceScale(+e.target.value)} />
              </div>
              <p className="sg-card__label" style={{ marginTop: 12 }}>Vertical Position</p>
              <div className="sg-slider-row">
                <span className="sg-slider-val">{offsetY > 0 ? "+" : ""}{offsetY}</span>
                <input className="sg-slider" type="range"
                  min="-40" max="40" step="2"
                  value={offsetY} onChange={e => setOffsetY(+e.target.value)} />
              </div>
            </section>
          )}

          {/* ── FACE ANALYSIS ──────────────────────────────── */}
          {shapeData && (
            <section className="sg-card sg-card--shape" style={{ "--accent": shapeData.color }}>
              <p className="sg-card__label">AI Face Analysis</p>
              <div className="sg-shape-row">
                <span className="sg-shape-emoji">{shapeData.emoji}</span>
                <div>
                  <p className="sg-shape-name">{faceShape} Face</p>
                  <p className="sg-shape-tip">{shapeData.tip}</p>
                </div>
              </div>
              <div className="sg-recs">
                <div className="sg-rec">
                  <span className="sg-rec__label">💎 Earrings</span>
                  <span className="sg-rec__val">{shapeData.earring}</span>
                </div>
                <div className="sg-rec">
                  <span className="sg-rec__label">📿 Necklace</span>
                  <span className="sg-rec__val">{shapeData.necklace}</span>
                </div>
              </div>
            </section>
          )}

          <section className="sg-card sg-card--actions">
            <button className="sg-btn sg-btn--primary"
              onClick={handleCapture}
              disabled={!earring && !necklace}>
              📸 Capture Look
            </button>
          </section>

          <section className="sg-card sg-card--tips">
            <p className="sg-card__label">Tips for best results</p>
            <ul className="sg-tips">
              <li>Face camera directly — both ears visible</li>
              <li>Good lighting reduces tracking jitter</li>
              <li>Pair photos auto-extract one earring</li>
              <li>Use vertical slider to fine-tune necklace</li>
            </ul>
          </section>

        </aside>

        <section className="sg-viewer">
          <div className="sg-canvas-wrap">
            <video ref={videoRef} className="sg-hidden" />
            <canvas ref={canvasRef} width={1920} height={1440} className="sg-canvas" />
            {!cameraReady && (
              <div className="sg-overlay">
                <span className="sg-spinner" /><p>Starting camera…</p>
              </div>
            )}
            {cameraReady && !faceDetected && (
              <div className="sg-overlay sg-overlay--hint">
                <p>👤 Position your face in frame</p>
              </div>
            )}
          </div>
          <p className="sg-viewer__hint">Sri Ganesh Jewellers · sgjewels.in</p>
        </section>

      </main>

      {showModal && (
        <div className="sg-modal-backdrop" onClick={() => setShowModal(false)}>
          <div className="sg-modal" onClick={e => e.stopPropagation()}>
            <button className="sg-modal__close" onClick={() => setShowModal(false)}>✕</button>
            <p className="sg-modal__title">Your Look ✨</p>
            <img src={captureURL} alt="Try-on result" className="sg-modal__img" />
            <div className="sg-modal__actions">
              <button className="sg-btn sg-btn--primary" onClick={handleDownload}>
                ⬇ Download HD
              </button>
              <button className="sg-btn sg-btn--ghost" onClick={() => setShowModal(false)}>
                Continue Try-On
              </button>
            </div>
            <p className="sg-modal__brand">Sri Ganesh Jewellers · sgjewels.in</p>
          </div>
        </div>
      )}

    </div>
  );
}