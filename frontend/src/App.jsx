
import React, { useEffect, useRef, useState, useCallback } from "react";
import { FaceMesh } from "@mediapipe/face_mesh";
import { Camera } from "@mediapipe/camera_utils";
import "./styles.css";

const BACKEND_URL = "http://localhost:8000/process-image";

// ─── MediaPipe FaceMesh landmark reference (face silhouette) ────────────────
//
//  LEFT side of IMAGE  = person's RIGHT side of face
//  RIGHT side of IMAGE = person's LEFT  side of face
//
//  Key ear landmarks (face silhouette, going cheek → jaw):
//    Left  image side:  234(cheekbone) → 93 → 132(tragus) → 58 → 172(lobe area)
//    Right image side:  454(cheekbone) → 323→ 361(tragus) → 288→ 397(lobe area)
//
//  Strategy:
//    EAR X  → face-edge cheekbone landmark  (234 or 454)  — the widest face point
//    EAR Y  → tragus landmark               (132 or 361)  — where earring sits
//    Earring hangs DOWN from this anchor point (top-of-earring = anchor Y)
// ─────────────────────────────────────────────────────────────────────────────

const SMOOTH_N = 12;
let lBuf = [], rBuf = [];

function pushBuf(buf, pt) {
  buf.push({ x: pt.x, y: pt.y });
  if (buf.length > SMOOTH_N) buf.shift();
  const s = buf.reduce((a, b) => ({ x: a.x + b.x, y: a.y + b.y }), { x: 0, y: 0 });
  return { x: s.x / buf.length, y: s.y / buf.length };
}

// ─── Geometry helpers ─────────────────────────────────────────────────────────
function faceHeight(lm, H) {
  return Math.abs(lm[10].y * H - lm[152].y * H);
}

function faceWidth(lm, W) {
  return Math.abs(lm[454].x * W - lm[234].x * W);
}

function getHeadTilt(lm, W, H) {
  // Eye-line angle for earring rotation
  return Math.atan2(
    lm[263].y * H - lm[33].y * H,
    lm[263].x * W - lm[33].x * W
  );
}

// ─── EAR ANCHOR POSITION ─────────────────────────────────────────────────────
//  Returns the point where the earring hook sits on the ear.
//  X = cheekbone edge (most extreme X on that side of the face)
//  Y = tragus landmark height (132 / 361), shifted down 3% of face-height
//      for the actual lobe where a dangling earring starts.
function earAnchor(lm, side, W, H) {
  // ✅ TRUE EAR LOBE LANDMARKS
  // These are MUCH more accurate for earrings

  if (side === "L") {
    const p1 = lm[177];  // lower ear
    const p2 = lm[93];   // mid ear
    const p3 = lm[234];  // outer edge

    return {
      x: (p1.x + p2.x + p3.x) / 3 * W,
      y: (p1.y + p2.y) / 2 * H
    };
  } else {
    const p1 = lm[401];
    const p2 = lm[323];
    const p3 = lm[454];

    return {
      x: (p1.x + p2.x + p3.x) / 3 * W,
      y: (p1.y + p2.y) / 2 * H
    };
  }
}

// ─── FACE TURN opacity (yaw) ──────────────────────────────────────────────────
//  When nose shifts toward one cheek, that cheek's ear fades out.
function yawOpacity(lm, W) {
  const noseX  = lm[4].x * W;
  const lx     = lm[234].x * W;
  const rx     = lm[454].x * W;
  const center = (lx + rx) / 2;
  const fw     = Math.abs(rx - lx);
  if (fw < 1) return { L: 1, R: 1 };
  // norm negative → nose toward lm[234] side; positive → toward lm[454] side
  const norm = (noseX - center) / fw;
  return {
    L: Math.max(0, Math.min(1, (norm + 0.30) / 0.20)),  // LEFT image side
    R: Math.max(0, Math.min(1, (0.30 - norm) / 0.20)),  // RIGHT image side
  };
}

// ─── HEAD PITCH opacity (tilt up/down) ───────────────────────────────────────
//  When head tilts far down, ears leave the camera plane; earrings fade out.
function pitchOpacity(lm, H) {
  const topY  = lm[10].y  * H;
  const noseY = lm[4].y   * H;
  const chinY = lm[152].y * H;
  const fh    = chinY - topY;
  if (fh < 1) return 1;
  const ratio = (noseY - topY) / fh;         // front-facing ≈ 0.47–0.55
  // Tilt DOWN increases ratio; fade starts at 0.60, gone at 0.70
  const tiltDown = Math.max(0, (ratio - 0.58) / 0.10);
  return Math.max(0, 1 - tiltDown);
}

// ─── Draw rotated + optionally mirrored image ─────────────────────────────────
function drawRotated(ctx, img, cx, cy, w, h, angle, alpha, flipH) {
  if (alpha <= 0.01) return;
  ctx.save();
  ctx.globalAlpha   = Math.min(0.96, alpha);
  ctx.shadowColor   = "rgba(0,0,0,0.20)";
  ctx.shadowBlur = 10;
  ctx.shadowOffsetX = 1;
  ctx.shadowOffsetY = 2;
  ctx.translate(cx, cy);
  ctx.rotate(angle);
  if (flipH) ctx.scale(-1, 1);
  // Draw with top of earring at origin → earring hangs down from anchor
  ctx.drawImage(img, 0, 0, img.width, img.height, -w / 2, -h * 0.15, w, h);
  ctx.restore();
}

// ─── 5-category face shape ────────────────────────────────────────────────────
function detectFaceShape(lm, W, H) {
  const fw = faceWidth(lm, W);
  const fh = faceHeight(lm, H);
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
  Oval:      { emoji: "🥚", tip: "Any style suits you!",            rec: "Chandelier Earrings",      color: "#7a9e7e" },
  Round:     { emoji: "⭕", tip: "Elongate with vertical earrings.", rec: "Long Drop Earrings",       color: "#9e7a7a" },
  Square:    { emoji: "◼️", tip: "Soften angles with curves.",       rec: "Hoop Earrings",            color: "#7a879e" },
  Heart:     { emoji: "🫀", tip: "Balance with wider base styles.",  rec: "Teardrop Earrings",        color: "#9e7a94" },
  Rectangle: { emoji: "▬",  tip: "Add width with cluster styles.",   rec: "Cluster / Stud Earrings", color: "#9e9a7a" },
};

// ═══════════════════════════════════════════════════════════════════════════════
export default function App() {
  const videoRef   = useRef(null);
  const canvasRef  = useRef(null);
  const earringRef = useRef(null);
  const scaleRef   = useRef(1.0);

  const [earring,      setEarring]      = useState(null);
  const [scale,        setScale]        = useState(1.0);
  const [status,       setStatus]       = useState("idle");
  const [statusMsg,    setStatusMsg]    = useState("");
  const [faceShape,    setFaceShape]    = useState(null);
  const [captureURL,   setCaptureURL]   = useState(null);
  const [showModal,    setShowModal]    = useState(false);
  const [cameraReady,  setCameraReady]  = useState(false);
  const [faceDetected, setFaceDetected] = useState(false);

  useEffect(() => { earringRef.current = earring; }, [earring]);
  useEffect(() => { scaleRef.current   = scale;   }, [scale]);

  // ─── FaceMesh — runs once ──────────────────────────────────────────
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

      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = "high";
      ctx.webkitImageSmoothingEnabled = true;
      ctx.mozImageSmoothingEnabled = true;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(res.image, 0, 0, canvas.width, canvas.height);

      const hasLM = res.multiFaceLandmarks?.length > 0;
      setFaceDetected(hasLM);
      if (!hasLM || !earringRef.current) return;

      const lm = res.multiFaceLandmarks[0];
      const W  = canvas.width;
      const H  = canvas.height;

      // Face shape (throttled)
      App._fc = ((App._fc || 0) + 1);
      if (App._fc % 8 === 0) setFaceShape(detectFaceShape(lm, W, H));

      // ── Earring dimensions ──────────────────────────────────
      const fw   = faceWidth(lm, W);
      const ew   = fw * 0.17 * scaleRef.current;
      const eh   = ew * 2.3;
      const tilt = getHeadTilt(lm, W, H);

      // ── Ear anchor positions (FIXED: tragus Y, cheekbone X) ─
      const rawL = earAnchor(lm, "L", W, H);
      const rawR = earAnchor(lm, "R", W, H);
      const ancL = pushBuf(lBuf, rawL);
      const ancR = pushBuf(rBuf, rawR);

      // ── Visibility (yaw = left/right turn) ──────────────────
      const yaw   = yawOpacity(lm, W);
      const pitch = pitchOpacity(lm, H);

      const alphaL = yaw.L * pitch;
      const alphaR = yaw.R * pitch;

      // ── Draw: top of earring sits at anchor point ────────────
      // drawRotated uses ctx.drawImage(img, -w/2, 0, w, h)
      // → earring top = anchor, hangs downward
      drawRotated(ctx, earringRef.current, ancL.x, ancL.y, ew, eh, tilt, alphaL, false);
      drawRotated(ctx, earringRef.current, ancR.x, ancR.y, ew, eh, tilt, alphaR, true);
    });

    const cam = new Camera(videoRef.current, {
      onFrame: async () => { await mesh.send({ image: videoRef.current }); },
      width: 640, height: 480,
    });
    cam.start().then(() => setCameraReady(true));
    return () => { cam.stop(); mesh.close(); };
  }, []);

  // ─── Upload ────────────────────────────────────────────────────────
  const handleFile = useCallback(async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setStatus("error");
      setStatusMsg("Please upload a valid image file.");
      return;
    }

    setStatus("loading");
    setStatusMsg("Processing earring…");

    try {
      const form = new FormData();
      form.append("file", file);
      const res = await fetch(BACKEND_URL, { method: "POST", body: form });
      const ct  = res.headers.get("content-type") || "";

      if (!res.ok || ct.includes("application/json")) {
        const json = await res.json();
        setStatus("error");
        setStatusMsg(json.error || "Server error.");
        return;
      }

      const blob = await res.blob();
      const img  = new Image();
      img.onload = () => {
        lBuf = []; rBuf = [];
        setEarring(img);
        setStatus("loaded");
        setStatusMsg("Earring ready ✓");
      };
      img.onerror = () => {
        setStatus("error");
        setStatusMsg("Could not load processed earring.");
      };
      img.src = URL.createObjectURL(blob);
    } catch {
      setStatus("error");
      setStatusMsg("Backend not reachable. Is the Python server running?");
    }
  }, []);

  const handleCapture = useCallback(() => {
  const canvas = canvasRef.current;
  if (!canvas) return;

  // 🔥 Create HD export canvas
  const exportCanvas = document.createElement("canvas");
  exportCanvas.width = 1920;
  exportCanvas.height = 1440;

  const ctx = exportCanvas.getContext("2d");
  ctx.imageSmoothingEnabled = true;
  ctx.imageSmoothingQuality = "high";

  // Draw original canvas into HD canvas
  ctx.drawImage(canvas, 0, 0, exportCanvas.width, exportCanvas.height);

  const finalImage = exportCanvas.toDataURL("image/png", 1.0);

  setCaptureURL(finalImage);
  setShowModal(true);
}, []);

  const handleDownload = useCallback(() => {
    if (!captureURL) return;
    const a = document.createElement("a");
    a.download = `sgjewels-tryon-${Date.now()}.png`;
    a.href = captureURL;
    a.click();
  }, [captureURL]);

  const shapeData = faceShape ? SHAPE_DATA[faceShape] : null;
  const statusConfig = {
    idle:    { cls: "",        label: "Upload an earring to begin" },
    loading: { cls: "loading", label: statusMsg || "Processing…"  },
    loaded:  { cls: "success", label: statusMsg                   },
    error:   { cls: "error",   label: statusMsg                   },
  }[status];

  // ════════════════════════════════════════════════════════════════════
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

          <section className="sg-card">
            <p className="sg-card__label">Upload Earring</p>
            <label className="sg-upload">
              <input type="file" accept="image/*" onChange={handleFile} />
              <span className="sg-upload__icon">💎</span>
              <span className="sg-upload__text">Choose Image</span>
            </label>
            {status !== "idle" && (
              <p className={`sg-status sg-status--${statusConfig.cls}`}>
                {statusConfig.label}
              </p>
            )}
            <p className="sg-hint">Single earring or pair — auto-extracts one</p>
          </section>

          <section className="sg-card">
            <p className="sg-card__label">Earring Size</p>
            <div className="sg-slider-row">
              <span className="sg-slider-val">{scale.toFixed(2)}×</span>
              <input
                className="sg-slider"
                type="range" min="0.6" max="1.6" step="0.05"
                value={scale}
                onChange={e => setScale(+e.target.value)}
              />
            </div>
          </section>

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

          <section className="sg-card sg-card--actions">
            <button className="sg-btn sg-btn--primary" onClick={handleCapture} disabled={!earring}>
              📸 Capture Look
            </button>
          </section>

          <section className="sg-card sg-card--tips">
            <p className="sg-card__label">Tips for best results</p>
            <ul className="sg-tips">
              <li>Face camera directly for both ears</li>
              <li>Good lighting reduces jitter</li>
              <li>Turn head — far ear auto-hides</li>
              <li>Pair images auto-extract one earring</li>
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
              <button className="sg-btn sg-btn--primary" onClick={handleDownload}>⬇ Download HD</button>
              <button className="sg-btn sg-btn--ghost"   onClick={() => setShowModal(false)}>Continue Try-On</button>
            </div>
            <p className="sg-modal__brand">Sri Ganesh Jewellers · sgjewels.in</p>
          </div>
        </div>
      )}

    </div>
  );
}

