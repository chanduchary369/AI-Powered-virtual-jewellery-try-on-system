import React, { useEffect, useRef, useState, useCallback } from "react";
import { FaceMesh } from "@mediapipe/face_mesh";
import { Camera } from "@mediapipe/camera_utils";
import "./styles.css";

const BACKEND_URL = "http://localhost:8000/process-image";

// ═══════════════════════════════════════════════════════════════════
//  SMOOTHING
// ═══════════════════════════════════════════════════════════════════
const SMOOTH_N  = 14;
const MAX_DELTA = 4;
let lBuf = [], rBuf = [], nBuf = [];   // nBuf = necklace centre

function pushBuf(buf, pt) {
  if (buf.length > 0) {
    const last = buf[buf.length - 1];
    const dx = pt.x - last.x, dy = pt.y - last.y;
    const d  = Math.hypot(dx, dy);
    if (d > MAX_DELTA) { const r = MAX_DELTA / d; pt = { x: last.x + dx*r, y: last.y + dy*r }; }
  }
  buf.push({ x: pt.x, y: pt.y });
  if (buf.length > SMOOTH_N) buf.shift();
  const s = buf.reduce((a, b) => ({ x: a.x + b.x, y: a.y + b.y }), { x: 0, y: 0 });
  return { x: s.x / buf.length, y: s.y / buf.length };
}

// ═══════════════════════════════════════════════════════════════════
//  GEOMETRY
// ═══════════════════════════════════════════════════════════════════
function faceWidth(lm, W)  { return Math.abs(lm[454].x * W - lm[234].x * W); }
function faceHeight(lm, H) { return Math.abs(lm[10].y  * H - lm[152].y * H); }

function getHeadTilt(lm, W, H) {
  return Math.atan2(lm[263].y * H - lm[33].y * H, lm[263].x * W - lm[33].x * W);
}

// ─── EAR ANCHOR ──────────────────────────────────────────────────
function earAnchor(lm, side, W, H) {
  if (side === "L") {
    const p1 = lm[177], p2 = lm[93], p3 = lm[234];
    return { x: (p1.x + p2.x + p3.x) / 3 * W, y: (p1.y + p2.y) / 2 * H };
  } else {
    const p1 = lm[401], p2 = lm[323], p3 = lm[454];
    return { x: (p1.x + p2.x + p3.x) / 3 * W, y: (p1.y + p2.y) / 2 * H };
  }
}

// ─── NECKLACE ANCHOR ─────────────────────────────────────────────
//  Uses neck/chin landmarks to position necklace accurately:
//  lm[152] = chin bottom, lm[200] = chin centre, lm[18] = lower lip bottom
//  lm[8]   = nose bottom, lm[175] = lower throat area
//  Width based on shoulder/neck width using lm[234] and lm[454]
function necklaceAnchor(lm, W, H) {
  const fw   = faceWidth(lm, W);
  const fh   = faceHeight(lm, H);
  const chinY = lm[152].y * H;

  // Centre X = face centre
  const centreX = (lm[234].x * W + lm[454].x * W) / 2;

  // Necklace top sits just below the chin — 8% of face height below chin
  const topY = chinY + fh * 0.08;

  // Necklace width = 1.6× face width to span collar bone area
  const neckW = fw * 1.6;

  return { centreX, topY, neckW };
}

// ─── YAW & PITCH OPACITY ──────────────────────────────────────────
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
    N: Math.max(0.2, 1 - Math.abs(norm) * 1.2),   // necklace fades when turned
  };
}

function pitchOpacity(lm, H) {
  const topY = lm[10].y * H, noseY = lm[4].y * H, chinY = lm[152].y * H;
  const fh   = chinY - topY;
  if (fh < 1) return 1;
  const ratio    = (noseY - topY) / fh;
  const tiltDown = Math.max(0, (ratio - 0.60) / 0.10);
  return Math.max(0, 1 - tiltDown);
}

// ═══════════════════════════════════════════════════════════════════
//  CANVAS DRAWING HELPERS
// ═══════════════════════════════════════════════════════════════════

// ── Draw earring with ENHANCED brightness/contrast/glow ──────────
function drawEarring(ctx, img, cx, cy, w, h, angle, alpha, flipH) {
  if (alpha <= 0.01) return;
  ctx.save();

  // HIGHLIGHT: create glow effect using composite layer
  ctx.globalAlpha   = Math.min(0.95, alpha);
  ctx.shadowColor   = "rgba(255, 215, 0, 0.55)";   // gold glow
  ctx.shadowBlur    = 18;
  ctx.shadowOffsetX = 0;
  ctx.shadowOffsetY = 0;

  ctx.translate(cx, cy);
  ctx.rotate(angle);
  if (flipH) ctx.scale(-1, 1);

  // Draw earring normally
  ctx.drawImage(img, 0, 0, img.width, img.height, -w / 2, -h * 0.15, w, h);

  // SECOND PASS: overlay with lighter composite for brightness boost
  ctx.globalCompositeOperation = "screen";
  ctx.globalAlpha = Math.min(0.18, alpha * 0.2);
  ctx.drawImage(img, 0, 0, img.width, img.height, -w / 2, -h * 0.15, w, h);

  ctx.globalCompositeOperation = "source-over";
  ctx.restore();
}

// ── Draw necklace with highlight & glow ──────────────────────────
function drawNecklace(ctx, img, centreX, topY, neckW, alpha) {
  if (alpha <= 0.01) return;

  // Scale: necklace height is proportional to its natural aspect ratio
  const aspect = img.naturalHeight / img.naturalWidth || 1;
  const neckH  = neckW * aspect;

  ctx.save();

  // HIGHLIGHT: warm gold glow matching jewellery
  ctx.shadowColor   = "rgba(255, 210, 80, 0.6)";
  ctx.shadowBlur    = 22;
  ctx.shadowOffsetX = 0;
  ctx.shadowOffsetY = 0;

  ctx.globalAlpha = Math.min(0.96, alpha);

  // Draw necklace centred horizontally, hanging from topY
  ctx.drawImage(
    img,
    centreX - neckW / 2,
    topY,
    neckW,
    neckH
  );

  // SECOND PASS: screen blend for brightness/shine boost
  ctx.globalCompositeOperation = "screen";
  ctx.globalAlpha = Math.min(0.20, alpha * 0.22);
  ctx.drawImage(img, centreX - neckW / 2, topY, neckW, neckH);

  ctx.globalCompositeOperation = "source-over";
  ctx.restore();
}

// ═══════════════════════════════════════════════════════════════════
//  FACE SHAPE
// ═══════════════════════════════════════════════════════════════════
function detectFaceShape(lm, W, H) {
  const fw = faceWidth(lm, W), fh = faceHeight(lm, H);
  if (fw < 10) return null;
  const ratio     = fh / fw;
  const jawW      = Math.abs(lm[172].x * W - lm[397].x * W);
  const foreheadW = Math.abs(lm[103].x * W - lm[332].x * W);
  if (ratio > 1.5)                             return "Oval";
  if (ratio > 1.3 && foreheadW > jawW * 1.05) return "Heart";
  if (ratio < 1.15 && jawW / fw > 0.82)       return "Square";
  if (ratio > 1.2)                             return "Round";
  return "Rectangle";
}

const SHAPE_DATA = {
  Oval:      { emoji: "🥚", tip: "Any style suits you!",             rec: "Chandelier Earrings",      color: "#7a9e7e" },
  Round:     { emoji: "⭕", tip: "Elongate with vertical earrings.",  rec: "Long Drop Earrings",       color: "#9e7a7a" },
  Square:    { emoji: "◼️", tip: "Soften angles with curves.",        rec: "Hoop Earrings",            color: "#7a879e" },
  Heart:     { emoji: "🫀", tip: "Balance with wider base styles.",   rec: "Teardrop Earrings",        color: "#9e7a94" },
  Rectangle: { emoji: "▬",  tip: "Add width with cluster styles.",    rec: "Cluster / Stud Earrings", color: "#9e9a7a" },
};

// ═══════════════════════════════════════════════════════════════════
//  APP
// ═══════════════════════════════════════════════════════════════════
export default function App() {
  const videoRef    = useRef(null);
  const canvasRef   = useRef(null);
  const earringRef  = useRef(null);
  const necklaceRef = useRef(null);
  const scaleRef    = useRef(1.0);
  const modeRef     = useRef("earring");   // "earring" | "necklace" | "both"

  const [earring,      setEarring]      = useState(null);
  const [necklace,     setNecklace]     = useState(null);
  const [activeMode,   setActiveMode]   = useState("earring");
  const [scale,        setScale]        = useState(1.0);
  const [status,       setStatus]       = useState("idle");
  const [neckStatus,   setNeckStatus]   = useState("idle");
  const [statusMsg,    setStatusMsg]    = useState("");
  const [neckMsg,      setNeckMsg]      = useState("");
  const [faceShape,    setFaceShape]    = useState(null);
  const [captureURL,   setCaptureURL]   = useState(null);
  const [showModal,    setShowModal]    = useState(false);
  const [cameraReady,  setCameraReady]  = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);

  useEffect(() => { earringRef.current  = earring;    }, [earring]);
  useEffect(() => { necklaceRef.current = necklace;   }, [necklace]);
  useEffect(() => { scaleRef.current    = scale;      }, [scale]);
  useEffect(() => { modeRef.current     = activeMode; }, [activeMode]);

  // ─── FACEMESH LOOP ──────────────────────────────────────────────
  useEffect(() => {
    const mesh = new FaceMesh({
      locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}`,
    });
    mesh.setOptions({
      maxNumFaces:            1,
      refineLandmarks:        true,
      minDetectionConfidence: 0.65,
      minTrackingConfidence:  0.65,
    });

    mesh.onResults(res => {
      const canvas = canvasRef.current;
      if (!canvas) return;
      const ctx = canvas.getContext("2d");

      ctx.imageSmoothingEnabled  = true;
      ctx.imageSmoothingQuality  = "high";
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(res.image, 0, 0, canvas.width, canvas.height);

      const hasLM = res.multiFaceLandmarks?.length > 0;
      setFaceDetected(hasLM);
      if (!hasLM) return;

      const lm  = res.multiFaceLandmarks[0];
      const W   = canvas.width;
      const H   = canvas.height;
      const fw  = faceWidth(lm, W);
      const tilt = getHeadTilt(lm, W, H);
      const yaw   = yawOpacity(lm, W);
      const pitch = pitchOpacity(lm, H);
      const mode  = modeRef.current;

      App._fc = ((App._fc || 0) + 1);
      if (App._fc % 8 === 0) setFaceShape(detectFaceShape(lm, W, H));

      // ── EARRINGS ──────────────────────────────────────────────
      if ((mode === "earring" || mode === "both") && earringRef.current) {
        const ew = fw * 0.17 * scaleRef.current;
        const eh = ew * 2.3;
        const rawL = earAnchor(lm, "L", W, H);
        const rawR = earAnchor(lm, "R", W, H);
        const ancL = pushBuf(lBuf, rawL);
        const ancR = pushBuf(rBuf, rawR);
        drawEarring(ctx, earringRef.current,  ancL.x, ancL.y, ew, eh, tilt, yaw.L * pitch, false);
        drawEarring(ctx, earringRef.current,  ancR.x, ancR.y, ew, eh, tilt, yaw.R * pitch, true);
      }

      // ── NECKLACE ──────────────────────────────────────────────
      if ((mode === "necklace" || mode === "both") && necklaceRef.current) {
        const { centreX, topY, neckW } = necklaceAnchor(lm, W, H);
        const rawN = { x: centreX, y: topY };
        const ancN = pushBuf(nBuf, rawN);

        // Recalculate neckW fresh each frame (face may have moved/resized)
        const freshFW   = faceWidth(lm, W);
        const freshNeckW = freshFW * 1.6 * scaleRef.current;

        drawNecklace(ctx, necklaceRef.current, ancN.x, ancN.y, freshNeckW, yaw.N * pitch);
      }
    });

    const cam = new Camera(videoRef.current, {
      onFrame: async () => { await mesh.send({ image: videoRef.current }); },
      width: 640, height: 480,
    });
    cam.start().then(() => setCameraReady(true));
    return () => { cam.stop(); mesh.close(); };
  }, []);

  // ─── UPLOAD EARRING ─────────────────────────────────────────────
  const handleEarringFile = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) { setStatus("error"); setStatusMsg("Please upload a valid image."); return; }
    setStatus("loading"); setStatusMsg("Processing earring…");
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("type", "earring");
      const res = await fetch(BACKEND_URL, { method: "POST", body: form });
      const ct  = res.headers.get("content-type") || "";
      if (!res.ok || ct.includes("application/json")) {
        const json = await res.json();
        setStatus("error"); setStatusMsg(json.error || "Server error."); return;
      }
      const blob = await res.blob();
      const img  = new Image();
      img.onload = () => { lBuf = []; rBuf = []; setEarring(img); setStatus("loaded"); setStatusMsg("Earring ready ✓"); };
      img.src = URL.createObjectURL(blob);
    } catch { setStatus("error"); setStatusMsg("Backend not reachable."); }
  }, []);

  // ─── UPLOAD NECKLACE ────────────────────────────────────────────
  const handleNecklaceFile = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    if (!file.type.startsWith("image/")) { setNeckStatus("error"); setNeckMsg("Please upload a valid image."); return; }
    setNeckStatus("loading"); setNeckMsg("Processing necklace…");
    try {
      const form = new FormData();
      form.append("file", file);
      form.append("type", "necklace");
      const res = await fetch(BACKEND_URL, { method: "POST", body: form });
      const ct  = res.headers.get("content-type") || "";
      if (!res.ok || ct.includes("application/json")) {
        const json = await res.json();
        setNeckStatus("error"); setNeckMsg(json.error || "Server error."); return;
      }
      const blob = await res.blob();
      const img  = new Image();
      img.onload = () => { nBuf = []; setNecklace(img); setNeckStatus("loaded"); setNeckMsg("Necklace ready ✓"); };
      img.src = URL.createObjectURL(blob);
    } catch { setNeckStatus("error"); setNeckMsg("Backend not reachable."); }
  }, []);

  // ─── CAPTURE ────────────────────────────────────────────────────
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

  const shapeData = faceShape ? SHAPE_DATA[faceShape] : null;

  const earStatusConfig = {
    idle:    { cls: "",        label: "Upload an earring image" },
    loading: { cls: "loading", label: statusMsg || "Processing…" },
    loaded:  { cls: "success", label: statusMsg },
    error:   { cls: "error",   label: statusMsg },
  }[status];

  const neckStatusConfig = {
    idle:    { cls: "",        label: "Upload a necklace image" },
    loading: { cls: "loading", label: neckMsg || "Processing…" },
    loaded:  { cls: "success", label: neckMsg },
    error:   { cls: "error",   label: neckMsg },
  }[neckStatus];

  // ═══════════════════════════════════════════════════════════════
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

          {/* ── MODE TABS ─────────────────────────────────── */}
          <section className="sg-card sg-card--tabs">
            <p className="sg-card__label">Try-On Mode</p>
            <div className="sg-tabs">
              {["earring","necklace","both"].map(m => (
                <button
                  key={m}
                  className={`sg-tab ${activeMode === m ? "sg-tab--active" : ""}`}
                  onClick={() => setActiveMode(m)}
                >
                  {m === "earring" ? "💎 Earrings" : m === "necklace" ? "📿 Necklace" : "✨ Both"}
                </button>
              ))}
            </div>
          </section>

          {/* ── EARRING UPLOAD ────────────────────────────── */}
          {(activeMode === "earring" || activeMode === "both") && (
            <section className="sg-card">
              <p className="sg-card__label">Upload Earring</p>
              <label className="sg-upload">
                <input type="file" accept="image/*" onChange={handleEarringFile} />
                <span className="sg-upload__icon">💎</span>
                <span className="sg-upload__text">Choose Earring Image</span>
              </label>
              {status !== "idle" && (
                <p className={`sg-status sg-status--${earStatusConfig.cls}`}>{earStatusConfig.label}</p>
              )}
              <p className="sg-hint">Single or pair — auto-extracts one</p>
            </section>
          )}

          {/* ── NECKLACE UPLOAD ───────────────────────────── */}
          {(activeMode === "necklace" || activeMode === "both") && (
            <section className="sg-card">
              <p className="sg-card__label">Upload Necklace</p>
              <label className="sg-upload sg-upload--necklace">
                <input type="file" accept="image/*" onChange={handleNecklaceFile} />
                <span className="sg-upload__icon">📿</span>
                <span className="sg-upload__text">Choose Necklace Image</span>
              </label>
              {neckStatus !== "idle" && (
                <p className={`sg-status sg-status--${neckStatusConfig.cls}`}>{neckStatusConfig.label}</p>
              )}
              <p className="sg-hint">Product photo with visible chain ends</p>
            </section>
          )}

          {/* ── SIZE SLIDER ───────────────────────────────── */}
          <section className="sg-card">
            <p className="sg-card__label">Jewellery Size</p>
            <div className="sg-slider-row">
              <span className="sg-slider-val">{scale.toFixed(2)}×</span>
              <input className="sg-slider" type="range" min="0.5" max="1.8" step="0.05"
                value={scale} onChange={e => setScale(+e.target.value)} />
            </div>
          </section>

          {/* ── AI FACE SHAPE ─────────────────────────────── */}
          {shapeData && (
            <section className="sg-card sg-card--shape" style={{ "--accent": shapeData.color }}>
              <p className="sg-card__label">AI Analysis</p>
              <div className="sg-shape-row">
                <span className="sg-shape-emoji">{shapeData.emoji}</span>
                <div>
                  <p className="sg-shape-name">{faceShape} Face</p>
                  <p className="sg-shape-tip">{shapeData.tip}</p>
                </div>
              </div>
              <div className="sg-rec">
                <span className="sg-rec__label">Recommended</span>
                <span className="sg-rec__val">{shapeData.rec}</span>
              </div>
            </section>
          )}

          {/* ── CAPTURE ───────────────────────────────────── */}
          <section className="sg-card sg-card--actions">
            <button className="sg-btn sg-btn--primary" onClick={handleCapture}
              disabled={!earring && !necklace}>
              📸 Capture Look
            </button>
          </section>

          {/* ── TIPS ──────────────────────────────────────── */}
          <section className="sg-card sg-card--tips">
            <p className="sg-card__label">Tips for best results</p>
            <ul className="sg-tips">
              <li>Face camera directly for both ears</li>
              <li>Good lighting = stable tracking</li>
              <li>Turn head — far ear auto-hides</li>
              <li>Use necklace product photos with visible chain ends</li>
            </ul>
          </section>

        </aside>

        {/* ── VIEWER ──────────────────────────────────────── */}
        <section className="sg-viewer">
          <div className="sg-canvas-wrap">
            <video ref={videoRef} className="sg-hidden" />
            <canvas ref={canvasRef} width={1920} height={1440} className="sg-canvas" />
            {!cameraReady && (
              <div className="sg-overlay"><span className="sg-spinner" /><p>Starting camera…</p></div>
            )}
            {cameraReady && !faceDetected && (
              <div className="sg-overlay sg-overlay--hint"><p>👤 Position your face in frame</p></div>
            )}
          </div>
          <p className="sg-viewer__hint">Sri Ganesh Jewellers · sgjewels.in</p>
        </section>

      </main>

      {/* ── MODAL ─────────────────────────────────────────── */}
      {showModal && (
        <div className="sg-modal-backdrop" onClick={() => setShowModal(false)}>
          <div className="sg-modal" onClick={e => e.stopPropagation()}>
            <button className="sg-modal__close" onClick={() => setShowModal(false)}>✕</button>
            <p className="sg-modal__title">Your Look ✨</p>
            <img src={captureURL} alt="Try-on result" className="sg-modal__img" />
            <div className="sg-modal__actions">
              <button className="sg-btn sg-btn--primary" onClick={handleDownload}>⬇ Download HD</button>
              <button className="sg-btn sg-btn--ghost" onClick={() => setShowModal(false)}>Continue Try-On</button>
            </div>
            <p className="sg-modal__brand">Sri Ganesh Jewellers · sgjewels.in</p>
          </div>
        </div>
      )}
    </div>
  );
}