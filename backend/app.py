# ═══════════════════════════════════════════════════════════════════
#  SG Jewels · app.py  v5
#  Key fix: robust pair detection with vertical-split strategy
#  Works for product photos (white bg, earrings side-by-side)
# ═══════════════════════════════════════════════════════════════════

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import numpy as np
import cv2
from PIL import Image
import io
import traceback

try:
    from rembg import remove as _rembg
    REMBG_AVAILABLE = True
    print("[SG Jewels] ✅ rembg loaded")
except ImportError:
    REMBG_AVAILABLE = False
    print("[SG Jewels] ⚠️  rembg not found — pip install rembg onnxruntime")

U2NET_AVAILABLE = False

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════
#  BG REMOVAL
# ═══════════════════════════════════════════════════════════════════

def _u2net_remove(pil_rgb: Image.Image):
    return None   # plug in your U2Net here


def _rembg_remove(pil_rgb: Image.Image):
    if not REMBG_AVAILABLE:
        return None
    try:
        out = _rembg(pil_rgb)
        rgba = np.array(out.convert("RGBA"))
        if rgba[:, :, 3].min() >= 254:
            return None
        return rgba
    except Exception:
        traceback.print_exc()
        return None


def _grabcut_remove(pil_rgb: Image.Image) -> np.ndarray:
    orig_w, orig_h = pil_rgb.size
    MAX_DIM = 600
    scale   = min(MAX_DIM / orig_w, MAX_DIM / orig_h, 1.0)
    w, h    = max(int(orig_w * scale), 100), max(int(orig_h * scale), 100)
    img     = np.array(pil_rgb.resize((w, h), Image.LANCZOS).convert("RGB"))
    mask    = np.zeros((h, w), np.uint8)
    bgM, fgM = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    pad  = max(int(min(h, w) * 0.07), 8)
    rect = (pad, pad, w - 2 * pad, h - 2 * pad)
    try:
        cv2.grabCut(img, mask, rect, bgM, fgM, 12, cv2.GC_INIT_WITH_RECT)
        a = np.where((mask == 2) | (mask == 0), 0, 255).astype(np.uint8)
    except Exception:
        a = np.full((h, w), 255, dtype=np.uint8)
    k5 = np.ones((5, 5), np.uint8)
    a  = cv2.morphologyEx(a, cv2.MORPH_CLOSE, k5)
    a  = cv2.morphologyEx(a, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    af = cv2.GaussianBlur(a.astype(np.float32) / 255.0, (7, 7), 2)
    a  = (af * 255).clip(0, 255).astype(np.uint8)
    if scale < 1.0:
        a = cv2.resize(a, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    rgba         = np.array(pil_rgb.convert("RGBA"))
    rgba[:, :, 3] = a
    return rgba


def remove_background(pil_rgb: Image.Image) -> np.ndarray:
    for fn in [_u2net_remove, _rembg_remove]:
        r = fn(pil_rgb)
        if r is not None:
            return r
    return _grabcut_remove(pil_rgb)


# ═══════════════════════════════════════════════════════════════════
#  SMART PAIR DETECTION — main logic
# ═══════════════════════════════════════════════════════════════════

class JewelleryError(Exception):
    pass


def _clean_mask(alpha: np.ndarray) -> np.ndarray:
    _, b = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    k = np.ones((7, 7), np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN,  np.ones((3, 3), np.uint8))
    return b


def _good_contours(binary: np.ndarray, img_area: int):
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_px   = max(100, img_area * 0.001)
    cnts     = [c for c in cnts if cv2.contourArea(c) > min_px]
    cnts.sort(key=cv2.contourArea, reverse=True)
    return cnts


def _tight_crop(rgba: np.ndarray, pad: int = 14) -> np.ndarray:
    """Crop to tight bounding box of non-transparent pixels."""
    alpha = rgba[:, :, 3]
    rows  = np.any(alpha > 20, axis=1)
    cols  = np.any(alpha > 20, axis=0)
    if not rows.any() or not cols.any():
        return rgba
    H, W  = rgba.shape[:2]
    r1, r2 = np.where(rows)[0][[0, -1]]
    c1, c2 = np.where(cols)[0][[0, -1]]
    return rgba[max(0, r1-pad):min(H, r2+pad), max(0, c1-pad):min(W, c2+pad)]


def _crop_one_contour(rgba: np.ndarray, contour, pad: int = 14) -> np.ndarray:
    """Return RGBA with only this contour visible, tightly cropped."""
    H, W  = rgba.shape[:2]
    x, y, cw, ch = cv2.boundingRect(contour)
    x1, y1 = max(0, x - pad),      max(0, y - pad)
    x2, y2 = min(W, x + cw + pad), min(H, y + ch + pad)

    # Mask: keep ONLY this contour's pixels
    solo = np.zeros((H, W), np.uint8)
    cv2.drawContours(solo, [contour], -1, 255, cv2.FILLED)

    out         = rgba.copy()
    out[:, :, 3] = np.minimum(out[:, :, 3], solo)
    return out[y1:y2, x1:x2]


def _earring_score(c) -> float:
    x, y, cw, ch = cv2.boundingRect(c)
    aspect = ch / cw if cw > 0 else 1.0
    return cv2.contourArea(c) * max(1.0, aspect)


# ─── STRATEGY 1: contour-based pair split ──────────────────────────────
def _split_by_contours(rgba: np.ndarray):
    """
    If the alpha mask has TWO clearly separated blobs of similar size,
    pick the better one (taller aspect = more earring-like).
    Returns (cropped_rgba, 'pair') or None if not clearly two blobs.
    """
    H, W   = rgba.shape[:2]
    binary = _clean_mask(rgba[:, :, 3])
    cnts   = _good_contours(binary, H * W)

    if len(cnts) < 2:
        return None

    a0, a1  = cv2.contourArea(cnts[0]), cv2.contourArea(cnts[1])
    size_ratio = a1 / a0 if a0 > 0 else 0

    if size_ratio < 0.25:
        # Very different sizes — not a symmetric pair
        return None

    # Check horizontal separation (pair = blobs on LEFT and RIGHT halves)
    (x0, y0, w0, h0) = cv2.boundingRect(cnts[0])
    (x1, y1, w1, h1) = cv2.boundingRect(cnts[1])
    cx0 = x0 + w0 / 2
    cx1 = x1 + w1 / 2

    # Ensure they are NOT overlapping horizontally
    overlap = (x0 < x1 + w1) and (x1 < x0 + w0)
    if overlap and abs(cx0 - cx1) < W * 0.15:
        # Too close horizontally — probably NOT a pair side-by-side
        return None

    print(f"[pair] contour split: blob0 cx={cx0:.0f}, blob1 cx={cx1:.0f}")
    best = max(cnts[:2], key=_earring_score)
    return _crop_one_contour(rgba, best), "pair"


# ─── STRATEGY 2: vertical centre split ─────────────────────────────────
def _split_by_midline(pil_rgb: Image.Image) -> tuple:
    """
    For product photos where two earrings sit side-by-side on a white BG
    and rembg/GrabCut merges them into one blob:

    1. Find the vertical column with the LOWEST foreground density
       in the centre 40% of the image — this is the gap between the earrings.
    2. Crop the LEFT half (left earring).
    3. Apply BG removal to just that half.
    4. Return tight crop.
    """
    W, H   = pil_rgb.size

    # Convert to grayscale and threshold to detect fg
    gray  = np.array(pil_rgb.convert("L"))
    # For white-background product images, fg = dark pixels
    # For dark-background product images, fg = light pixels
    # Auto-detect: if mean > 200 it's likely white BG
    is_white_bg = gray.mean() > 160

    if is_white_bg:
        fg_mask = (gray < 220).astype(np.uint8) * 255
    else:
        fg_mask = (gray > 35).astype(np.uint8) * 255

    # Search for split column in centre 30–70% of width
    search_l = int(W * 0.30)
    search_r = int(W * 0.70)

    # Sum foreground pixels per column
    col_sums = fg_mask[:, search_l:search_r].sum(axis=0)

    # Find minimum density column = gap between earrings
    split_col = search_l + int(np.argmin(col_sums))
    print(f"[pair] midline split at x={split_col} (image W={W})")

    # Ensure split is somewhat near centre (not edge artefact)
    if split_col < W * 0.25 or split_col > W * 0.75:
        split_col = W // 2
        print(f"[pair] midline clamped to centre x={split_col}")

    # Crop LEFT earring with 5% padding on split edge
    pad   = max(int(W * 0.03), 8)
    left  = pil_rgb.crop((0, 0, split_col + pad, H))

    # BG removal on just the left half
    rgba_half = remove_background(left)

    # Tight crop to earring content only
    cropped = _tight_crop(rgba_half)

    if cropped.shape[0] < 20 or cropped.shape[1] < 20:
        return None

    print(f"[pair] midline half crop: {cropped.shape[1]}×{cropped.shape[0]}")
    return cropped, "pair"


# ─── STRATEGY 3: full-image fallback ───────────────────────────────────
def _full_image_fallback(rgba: np.ndarray):
    """Return tight crop of the full BG-removed image."""
    cropped = _tight_crop(rgba)
    if cropped.shape[0] < 20 or cropped.shape[1] < 20:
        raise JewelleryError("No jewellery detected. Please use a clearer image.")
    return cropped, "single"


# ─── MASTER ANALYSER ────────────────────────────────────────────────────
def analyze_jewellery(rgba: np.ndarray, pil_rgb: Image.Image):
    """
    Multi-strategy earring extraction:
      1. Contour split  — works when BG removal cleanly separates the two earrings
      2. Midline split  — works when they share a connected foreground blob (product photos)
      3. Single object  — necklace check + return as-is
      4. Full fallback  — always returns something useful
    """
    H, W  = rgba.shape[:2]
    alpha = rgba[:, :, 3]

    print(f"\n[analyze] image {W}×{H}, alpha min={alpha.min()} max={alpha.max()}")

    binary = _clean_mask(alpha)
    cnts   = _good_contours(binary, H * W)
    print(f"[analyze] {len(cnts)} contours found")

    # ── Single clear blob ──────────────────────────────────────────
    if len(cnts) == 1:
        x, y, cw, ch = cv2.boundingRect(cnts[0])
        aspect = cw / ch if ch > 0 else 1.0

        # Wide flat shape = necklace
        if aspect > 2.8:
            raise JewelleryError(
                "This looks like a necklace. Please upload an earring image."
            )

        # Could still be a pair merged into one blob
        # → try midline split first
        result = _split_by_midline(pil_rgb)
        if result is not None:
            print("[analyze] single blob → tried midline split → success")
            return result

        # Genuinely single earring
        return _crop_one_contour(rgba, cnts[0]), "single"

    # ── Two or more blobs ──────────────────────────────────────────
    if len(cnts) >= 2:

        # Try contour split first (cleanest result)
        result = _split_by_contours(rgba)
        if result is not None:
            print("[analyze] two blobs → contour split → success")
            return result

        # Contour split failed (blobs overlap / too unequal)
        # → try midline split on original image
        result = _split_by_midline(pil_rgb)
        if result is not None:
            print("[analyze] two blobs → midline split → success")
            return result

        # Take the best single contour (largest + tallest)
        best = max(cnts, key=_earring_score)
        print("[analyze] two blobs → best contour fallback")
        return _crop_one_contour(rgba, best), "single"

    # ── No contours but some alpha ─────────────────────────────────
    if alpha.max() > 50:
        result = _split_by_midline(pil_rgb)
        if result is not None:
            return result
        return _full_image_fallback(rgba)

    raise JewelleryError(
        "No jewellery detected. Please upload a clearer earring image "
        "with a simple background."
    )


# ═══════════════════════════════════════════════════════════════════
#  ROUTE
# ═══════════════════════════════════════════════════════════════════

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    try:
        ct = (file.content_type or "").lower()
        if not ct.startswith("image/"):
            return JSONResponse(400, content={"error": "Please upload a valid image (JPG, PNG, WEBP)."})

        raw     = await file.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        orig_w, orig_h = pil_img.size
        print(f"[request] {file.filename}  {orig_w}×{orig_h}")

        # Step 1: BG removal on full image
        rgba = remove_background(pil_img)

        # Restore resolution if BG remover resized
        rh, rw = rgba.shape[:2]
        if (rh, rw) != (orig_h, orig_w):
            rgba = cv2.resize(rgba, (orig_w, orig_h), interpolation=cv2.INTER_LANCZOS4)

        # Step 2: Extract single earring (pair-aware)
        try:
            processed, j_type = analyze_jewellery(rgba, pil_img)
        except JewelleryError as je:
            return JSONResponse(400, content={"error": str(je)})

        ph, pw = processed.shape[:2]
        print(f"[output] {pw}×{ph} type={j_type}")

        # Step 3: Lossless PNG
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