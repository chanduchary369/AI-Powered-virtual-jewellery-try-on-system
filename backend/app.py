# ═══════════════════════════════════════════════════════════════════════════
#  SG JEWELS · Backend  v8 — Production Grade
#  Earring + Necklace Try-On System
#  
#  Pipeline:  Upload → BG Removal → Type-Specific Extraction → Enhancement
#                                  → Lossless PNG Response
# ═══════════════════════════════════════════════════════════════════════════

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import numpy as np
import cv2
from PIL import Image, ImageEnhance, ImageFilter
import io
import traceback
from typing import Optional, Tuple

# ── rembg (preferred — uses U2Net deep learning) ─────────────────────────
try:
    from rembg import remove as _rembg, new_session
    REMBG_AVAILABLE = True
    _rembg_session = new_session("u2net")  # explicit u2net model
    print("[SG Jewels] ✅ rembg loaded with u2net session")
except ImportError:
    REMBG_AVAILABLE = False
    _rembg_session = None
    print("[SG Jewels] ⚠️  rembg unavailable — using GrabCut")

app = FastAPI(title="SG Jewels Try-On API", version="8.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 1 — BACKGROUND REMOVAL (cascaded for max accuracy)
# ═══════════════════════════════════════════════════════════════════════════

def _rembg_remove_with_matting(pil_rgb: Image.Image) -> Optional[np.ndarray]:
    """rembg with alpha matting for fine edge preservation (filigree, stones)."""
    if not REMBG_AVAILABLE:
        return None
    try:
        # Alpha matting recovers fine details around jewellery edges
        out = _rembg(
            pil_rgb,
            session=_rembg_session,
            alpha_matting=True,
            alpha_matting_foreground_threshold=240,
            alpha_matting_background_threshold=10,
            alpha_matting_erode_size=8,
        )
        rgba = np.array(out.convert("RGBA"))
        if rgba[:, :, 3].min() >= 254:
            return None
        return rgba
    except Exception:
        # Fall back to plain rembg without alpha matting
        try:
            out = _rembg(pil_rgb, session=_rembg_session)
            rgba = np.array(out.convert("RGBA"))
            if rgba[:, :, 3].min() >= 254:
                return None
            return rgba
        except Exception:
            traceback.print_exc()
            return None


def _grabcut_remove(pil_rgb: Image.Image) -> np.ndarray:
    """OpenCV GrabCut fallback with feathered edges."""
    ow, oh = pil_rgb.size
    s = min(700 / ow, 700 / oh, 1.0)
    w, h = max(int(ow * s), 100), max(int(oh * s), 100)
    img = np.array(pil_rgb.resize((w, h), Image.LANCZOS).convert("RGB"))
    msk = np.zeros((h, w), np.uint8)
    bg, fg = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    pad = max(int(min(h, w) * 0.06), 8)
    try:
        cv2.grabCut(img, msk, (pad, pad, w - 2 * pad, h - 2 * pad),
                    bg, fg, 15, cv2.GC_INIT_WITH_RECT)
        a = np.where((msk == 2) | (msk == 0), 0, 255).astype(np.uint8)
    except Exception:
        a = np.full((h, w), 255, dtype=np.uint8)

    # Multi-stage cleanup for smooth edges
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))

    # Feather edges with two-pass Gaussian for anti-aliasing
    af = a.astype(np.float32) / 255.0
    af = cv2.GaussianBlur(af, (9, 9), 2.5)
    af = cv2.GaussianBlur(af, (5, 5), 1.0)
    a = (af * 255).clip(0, 255).astype(np.uint8)

    if s < 1.0:
        a = cv2.resize(a, (ow, oh), interpolation=cv2.INTER_LANCZOS4)
    rgba = np.array(pil_rgb.convert("RGBA"))
    rgba[:, :, 3] = a
    return rgba


def remove_background(pil_rgb: Image.Image) -> np.ndarray:
    """Master BG remover — tries rembg with matting first."""
    r = _rembg_remove_with_matting(pil_rgb)
    if r is not None:
        return r
    print("[BG] rembg failed, using GrabCut fallback")
    return _grabcut_remove(pil_rgb)


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 2 — JEWELLERY ANALYSIS UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

class JewelleryError(Exception):
    pass


def _clean_alpha_mask(alpha: np.ndarray) -> np.ndarray:
    """Convert soft alpha → clean binary mask for contour analysis."""
    _, b = cv2.threshold(alpha, 12, 255, cv2.THRESH_BINARY)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    return b


def _significant_contours(binary: np.ndarray, img_area: int):
    """Find foreground contours above noise threshold."""
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    threshold = max(80, img_area * 0.0008)
    cnts = [c for c in cnts if cv2.contourArea(c) > threshold]
    cnts.sort(key=cv2.contourArea, reverse=True)
    return cnts


def _tight_crop(rgba: np.ndarray, pad: int = 14) -> np.ndarray:
    """Crop to tight bounding box of non-transparent pixels."""
    a = rgba[:, :, 3]
    rows = np.any(a > 15, axis=1)
    cols = np.any(a > 15, axis=0)
    if not rows.any() or not cols.any():
        return rgba
    H, W = rgba.shape[:2]
    r1, r2 = np.where(rows)[0][[0, -1]]
    c1, c2 = np.where(cols)[0][[0, -1]]
    return rgba[max(0, r1 - pad):min(H, r2 + pad),
                max(0, c1 - pad):min(W, c2 + pad)]


def _crop_one_contour(rgba: np.ndarray, cnt, pad: int = 16) -> np.ndarray:
    """Keep ONLY this contour visible. Tight crop."""
    H, W = rgba.shape[:2]
    x, y, cw, ch = cv2.boundingRect(cnt)
    x1, y1 = max(0, x - pad), max(0, y - pad)
    x2, y2 = min(W, x + cw + pad), min(H, y + ch + pad)
    solo = np.zeros((H, W), np.uint8)
    cv2.drawContours(solo, [cnt], -1, 255, cv2.FILLED)
    out = rgba.copy()
    out[:, :, 3] = np.minimum(out[:, :, 3], solo)
    return out[y1:y2, x1:x2]


def _earring_score(c) -> float:
    """Higher score = better earring candidate (large + tall)."""
    x, y, cw, ch = cv2.boundingRect(c)
    aspect = ch / cw if cw > 0 else 1.0
    return cv2.contourArea(c) * max(1.0, aspect)


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 3A — EARRING EXTRACTION (single from pair)
# ═══════════════════════════════════════════════════════════════════════════

def _alpha_column_split(alpha: np.ndarray) -> Optional[int]:
    """
    Find the vertical gap column between two earrings using alpha sums.
    Returns split column or None if no clear gap exists.
    """
    H, W = alpha.shape
    lo, hi = int(W * 0.25), int(W * 0.75)
    col_sums = alpha[:, lo:hi].sum(axis=0).astype(np.float32)
    smooth = np.convolve(col_sums, np.ones(9) / 9, mode='same')
    split_local = int(np.argmin(smooth))
    split_col = lo + split_local
    mn, mx = smooth[split_local], smooth.max()
    contrast = 1.0 - (mn / mx) if mx > 0 else 0
    print(f"[earring] alpha-split col={split_col} contrast={contrast:.2f}")
    if contrast < 0.20:
        return None
    return split_col


def _split_two_contours(rgba: np.ndarray):
    """Pair extraction when BG removal cleanly separated the earrings."""
    H, W = rgba.shape[:2]
    binary = _clean_alpha_mask(rgba[:, :, 3])
    cnts = _significant_contours(binary, H * W)
    if len(cnts) < 2:
        return None
    a0, a1 = cv2.contourArea(cnts[0]), cv2.contourArea(cnts[1])
    if a0 == 0 or a1 / a0 < 0.30:
        return None
    (x0, _, w0, _) = cv2.boundingRect(cnts[0])
    (x1, _, w1, _) = cv2.boundingRect(cnts[1])
    if (x0 < x1 + w1) and (x1 < x0 + w0):  # overlap in X
        return None
    best = max(cnts[:2], key=_earring_score)
    return _crop_one_contour(rgba, best)


def extract_earring(rgba: np.ndarray, pil_rgb: Image.Image) -> Tuple[np.ndarray, str]:
    """
    Extract single earring from any uploaded image.

    Decides pair-vs-single up front by counting significant foreground
    contours and inspecting the alpha-mask bounding box aspect:

      • 1 significant contour AND a tall-ish bbox  → SINGLE EARRING.
        Skip all pair-splitting (it would falsely "split" a single
        earring at its narrow waist / chain neck and crop garbage).
      • Otherwise → PAIR: try alpha-column split, then contour split.
    """
    H, W = rgba.shape[:2]
    alpha = rgba[:, :, 3]
    if alpha.max() < 15:
        raise JewelleryError("No earring detected. Use a clearer image.")

    # ── Pre-scan: how many real foreground blobs do we have? ──────────
    binary_pre = _clean_alpha_mask(alpha)
    cnts_pre = _significant_contours(binary_pre, H * W)

    # Overall bbox of all foreground pixels (whole jewellery footprint)
    rows = np.any(alpha > 15, axis=1)
    cols = np.any(alpha > 15, axis=0)
    if rows.any() and cols.any():
        r1, r2 = np.where(rows)[0][[0, -1]]
        c1, c2 = np.where(cols)[0][[0, -1]]
        bbox_w = max(1, c2 - c1)
        bbox_h = max(1, r2 - r1)
        bbox_aspect = bbox_w / bbox_h  # >1 = wider than tall
    else:
        bbox_aspect = 1.0

    # Single-earring heuristic: exactly one significant blob AND the
    # whole footprint is taller than wide (pairs are wider than tall
    # because they sit side-by-side). Use largest contour directly.
    if len(cnts_pre) == 1 and bbox_aspect < 1.30:
        c = cnts_pre[0]
        x, y, cw, ch = cv2.boundingRect(c)
        if cw > 0 and cw / ch > 2.8:
            raise JewelleryError(
                "This looks like a necklace. Switch to Necklace mode."
            )
        print(f"[earring] single-contour direct → {cw}×{ch}")
        return _crop_one_contour(rgba, c), "single"

    # Strategy 1: alpha column gap (works for merged-blob product photos).
    # Only attempt when the overall footprint is wider than tall — a true
    # pair sits side-by-side, so its bbox is landscape-ish.
    if bbox_aspect >= 1.10:
        sc = _alpha_column_split(alpha)
        if sc is not None:
            pad = max(int(W * 0.04), 12)
            cut = min(sc + pad, W)
            # Re-run BG removal on the cropped half for cleaner edges
            left_pil = pil_rgb.crop((0, 0, cut, H))
            rgba_clean = remove_background(left_pil)
            cropped = _tight_crop(rgba_clean)
            if cropped.shape[0] > 20 and cropped.shape[1] > 20:
                print(f"[earring] alpha-split → {cropped.shape[1]}×{cropped.shape[0]}")
                return cropped, "pair"

    # Strategy 2: two clean contours
    result = _split_two_contours(rgba)
    if result is not None and result.shape[0] > 20 and result.shape[1] > 20:
        print(f"[earring] two-contour split")
        return result, "pair"

    # Strategy 3: fall back to largest single contour
    if cnts_pre:
        x, y, cw, ch = cv2.boundingRect(cnts_pre[0])
        if cw > 0 and cw / ch > 2.8:
            raise JewelleryError(
                "This looks like a necklace. Switch to Necklace mode."
            )
        print(f"[earring] fallback single-contour → {cw}×{ch}")
        return _crop_one_contour(rgba, cnts_pre[0]), "single"

    return _tight_crop(rgba), "single"


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 3B — NECKLACE EXTRACTION (chain removal + design isolation)
# ═══════════════════════════════════════════════════════════════════════════

def extract_necklace(rgba: np.ndarray) -> np.ndarray:
    """
    Extract front necklace design only — remove side chains, back loops.
    
    Algorithm:
    1. Find the necklace contour (largest connected region).
    2. Identify the two TOPMOST points (left and right chain attachment).
    3. The cut line is the LOWER of these two Y values.
    4. Below the cut line = wearable front portion (pendant + design).
    5. Above the cut line = chain ends (REMOVE).
    6. Keep only the front portion.
    """
    H, W = rgba.shape[:2]
    alpha = rgba[:, :, 3]

    if alpha.max() < 15:
        raise JewelleryError("No necklace detected. Upload a clearer image.")

    binary = _clean_alpha_mask(alpha)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        return _tight_crop(rgba)

    # Largest contour = necklace
    main = max(cnts, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(main)

    # Validate aspect — necklaces are wider than tall
    aspect = cw / ch if ch > 0 else 1
    if aspect < 0.45:
        raise JewelleryError(
            "This looks like an earring. Switch to Earring mode."
        )

    # Find the two topmost points of the contour
    # Split contour points into left half and right half by X
    pts = main.reshape(-1, 2)  # shape: (N, 2)
    mid_x = x + cw // 2
    
    left_pts  = pts[pts[:, 0] < mid_x]
    right_pts = pts[pts[:, 0] >= mid_x]
    
    if len(left_pts) == 0 or len(right_pts) == 0:
        # Degenerate — fall back to tight crop
        return _crop_one_contour(rgba, main)

    # Topmost = smallest Y
    left_top_y  = int(left_pts[:, 1].min())
    right_top_y = int(right_pts[:, 1].min())

    left_top_x  = int(left_pts[left_pts[:, 1] == left_top_y][0, 0])
    right_top_x = int(right_pts[right_pts[:, 1] == right_top_y][0, 0])

    # Cut line = LOWER of the two top points
    # Above this line, both sides have chains — REMOVE
    cut_y = max(left_top_y, right_top_y)

    # Add safety margin to keep the natural curve at chain attachment
    cut_y = min(cut_y + int(ch * 0.02), y + ch - 20)

    print(f"[necklace] bbox={cw}x{ch}, "
          f"chain L=({left_top_x},{left_top_y}), R=({right_top_x},{right_top_y}), "
          f"cut_y={cut_y}")

    # Build mask: keep only the contour AND only below cut_y
    solo_mask = np.zeros((H, W), np.uint8)
    cv2.drawContours(solo_mask, [main], -1, 255, cv2.FILLED)
    solo_mask[:cut_y, :] = 0  # zero out everything above cut line

    # Apply to alpha
    out = rgba.copy()
    out[:, :, 3] = np.minimum(out[:, :, 3], solo_mask)

    # Find new bounding box of remaining content
    rem = out[:, :, 3]
    rows = np.any(rem > 15, axis=1)
    cols = np.any(rem > 15, axis=0)
    if not rows.any() or not cols.any():
        return _crop_one_contour(rgba, main)

    r1, r2 = np.where(rows)[0][[0, -1]]
    c1, c2 = np.where(cols)[0][[0, -1]]
    pad = 14
    y1, y2 = max(0, r1 - pad), min(H, r2 + pad)
    x1, x2 = max(0, c1 - pad), min(W, c2 + pad)

    cropped = out[y1:y2, x1:x2]

    # Final validation
    if cropped.shape[0] < 20 or cropped.shape[1] < 20:
        return _crop_one_contour(rgba, main)

    print(f"[necklace] final crop: {cropped.shape[1]}×{cropped.shape[0]}")
    return cropped


# ═══════════════════════════════════════════════════════════════════════════
#  STAGE 4 — DESIGN ENHANCEMENT  (premium / luxury-grade pipeline)
# ═══════════════════════════════════════════════════════════════════════════

def enhance_jewellery(rgba: np.ndarray) -> np.ndarray:
    """
    Premium multi-stage jewellery enhancement with HIGH-CONTRAST highlighting.

    Enhancement is applied ONLY to jewellery pixels via the alpha mask, so the
    webcam frame, skin, and background are never touched downstream.

    Pipeline
    ────────
      1. Adaptive contrast enhancement   (CLAHE on L channel of LAB)
      2. Tone curve — deep shadows + bright highlights (S-curve, contrast pop)
      3. Selective brightness            (gamma lift on midtones)
      4. Local sharpening                (unsharp mask)
      5. Edge enhancement                (high-frequency detail boost)
      6. Color richness improvement      (HSV saturation)
      7. Gold-tone enhancement           (warm hue saturation + shine lift)
      8. Specular highlight pass         (bright pixels glow brighter — gold sheen)
      9. Anti-aliasing refinement        (alpha edge smoothing, interior preserved)

    The S-curve (step 2) plus the specular pass (step 8) are what create the
    contrasting "lit-up" appearance against the webcam background — bright facets
    push toward white, deep recesses push toward black, so the design pops on
    skin instead of looking flat and washed-out.
    """
    if rgba is None or rgba.size == 0:
        return rgba

    rgba = rgba.astype(np.uint8)
    if rgba.shape[2] != 4:
        return rgba

    rgb = rgba[:, :, :3].copy()
    alpha = rgba[:, :, 3].copy()

    # Interior mask: only enhance solid jewellery pixels, leave feathered edges
    # to blend naturally during overlay.
    interior_mask = alpha > 30
    if not interior_mask.any():
        return rgba

    # ── 1. Adaptive contrast enhancement (CLAHE on L) ────────────────────
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    L, A_lab, B_lab = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    L_eq = clahe.apply(L)
    # Stronger CLAHE blend than before — recover detail aggressively
    L_blend = cv2.addWeighted(L_eq, 0.80, L, 0.20, 0)
    lab_eq = cv2.merge([L_blend, A_lab, B_lab])
    stage = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2RGB)

    # ── 2. Tone curve — S-curve for contrast pop ─────────────────────────
    # Pulls shadows down and highlights up so the jewellery reads as
    # 3-dimensional rather than flat. This is the main "highlighting" step.
    f = stage.astype(np.float32) / 255.0
    # Parametric S-curve: y = 0.5 - 0.5*cos(pi*x) gives a smooth contrast lift.
    # Blend with identity so we don't crush detail.
    s_curve = 0.5 - 0.5 * np.cos(np.pi * f)
    f = f * 0.35 + s_curve * 0.65
    f = np.clip(f, 0.0, 1.0)
    stage = (f * 255.0).astype(np.uint8)

    # ── 3. Selective brightness — lift midtones ──────────────────────────
    f = stage.astype(np.float32) / 255.0
    f = np.power(f, 0.90)
    stage = (np.clip(f, 0.0, 1.0) * 255.0).astype(np.uint8)

    # ── 4. Local sharpening (unsharp mask) ───────────────────────────────
    blur1 = cv2.GaussianBlur(stage, (0, 0), sigmaX=1.2)
    stage = cv2.addWeighted(stage, 1.70, blur1, -0.70, 0)

    # ── 5. Edge enhancement — high-frequency detail boost ────────────────
    blur2 = cv2.GaussianBlur(stage, (0, 0), sigmaX=2.4)
    stage = cv2.addWeighted(stage, 1.30, blur2, -0.30, 0)

    # ── 6. Color richness (HSV saturation, controlled) ───────────────────
    hsv = cv2.cvtColor(stage, cv2.COLOR_RGB2HSV).astype(np.float32)
    H_ch, S_ch, V_ch = cv2.split(hsv)
    S_ch = np.clip(S_ch * 1.30, 0, 255)
    V_ch = np.clip(V_ch * 1.05, 0, 255)
    hsv = cv2.merge([H_ch, S_ch, V_ch]).astype(np.uint8)
    stage = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

    # ── 7. Gold-tone enhancement — warm hue saturation + shine lift ──────
    hsv2 = cv2.cvtColor(stage, cv2.COLOR_RGB2HSV).astype(np.float32)
    Hg, Sg, Vg = cv2.split(hsv2)
    gold_band = ((Hg >= 8) & (Hg <= 48)).astype(np.float32)
    gold_band = cv2.GaussianBlur(gold_band, (0, 0), sigmaX=2.0)
    Sg = np.clip(Sg + gold_band * 22.0, 0, 255)
    Vg = np.clip(Vg + gold_band * 14.0, 0, 255)
    hsv2 = cv2.merge([Hg, Sg, Vg]).astype(np.uint8)
    stage = cv2.cvtColor(hsv2, cv2.COLOR_HSV2RGB)

    # ── 8. Specular highlight pass — gold "sheen" ────────────────────────
    # Detect the brightest 25 % of luminance inside the jewellery and push it
    # further toward warm-white. Creates the contrasting bright accents that
    # make the design pop against skin / webcam background.
    gray = cv2.cvtColor(stage, cv2.COLOR_RGB2GRAY).astype(np.float32)
    interior_vals = gray[interior_mask]
    if interior_vals.size > 0:
        thresh = float(np.percentile(interior_vals, 75))
        # Soft mask: 0 at threshold, 1 at full bright
        spec = np.clip((gray - thresh) / max(1.0, 255.0 - thresh), 0.0, 1.0)
        spec = cv2.GaussianBlur(spec, (0, 0), sigmaX=1.0)
        # Warm-white tint (R 255, G 244, B 214) for gold sheen
        warm_white = np.array([255.0, 244.0, 214.0], dtype=np.float32)
        s3 = stage.astype(np.float32)
        spec3 = spec[:, :, None]
        # Strength 0.55 = clearly visible highlight, still natural
        s3 = s3 * (1.0 - spec3 * 0.55) + warm_white * (spec3 * 0.55)
        stage = np.clip(s3, 0, 255).astype(np.uint8)

    # ── Compose: replace ONLY interior jewellery pixels ──────────────────
    out_rgb = rgb.copy()
    out_rgb[interior_mask] = stage[interior_mask]

    # ── 9. Anti-aliasing refinement on alpha edges ───────────────────────
    a_f = alpha.astype(np.float32) / 255.0
    a_smooth = cv2.GaussianBlur(a_f, (0, 0), sigmaX=0.7)
    solid = a_f > 0.85
    a_smooth[solid] = a_f[solid]
    out_alpha = (a_smooth * 255.0).clip(0, 255).astype(np.uint8)

    return np.dstack([out_rgb, out_alpha])


# ═══════════════════════════════════════════════════════════════════════════
#  ROUTE — POST /process-image
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    type: Optional[str] = Form("earring"),
):
    try:
        ct = (file.content_type or "").lower()
        if not ct.startswith("image/"):
            return JSONResponse(400, content={"error": "Please upload a valid image (JPG, PNG, WEBP)."})

        raw = await file.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        ow, oh = pil_img.size
        print(f"\n[req] type={type}  {file.filename}  {ow}×{oh}")

        # Step 1 — Background removal
        rgba = remove_background(pil_img)
        if rgba.shape[:2] != (oh, ow):
            rgba = cv2.resize(rgba, (ow, oh), interpolation=cv2.INTER_LANCZOS4)

        # Step 2 — Type-specific extraction
        try:
            if type == "necklace":
                processed = extract_necklace(rgba)
                j_type = "necklace"
            else:
                processed, j_type = extract_earring(rgba, pil_img)
        except JewelleryError as je:
            return JSONResponse(400, content={"error": str(je)})

        # Step 3 — Design enhancement
        processed = enhance_jewellery(processed)

        ph, pw = processed.shape[:2]
        print(f"[out] {pw}×{ph}  type={j_type}")

        # Step 4 — Lossless PNG response
        out = Image.fromarray(processed.astype(np.uint8), "RGBA")
        buf = io.BytesIO()
        out.save(buf, format="PNG", compress_level=1)
        buf.seek(0)

        return Response(
            content=buf.read(),
            media_type="image/png",
            headers={"X-Jewellery-Type": j_type},
        )

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(500, content={"error": f"Server error: {e}"})


@app.get("/health")
async def health():
    return {"status": "ok", "rembg": REMBG_AVAILABLE, "version": "8.0"}