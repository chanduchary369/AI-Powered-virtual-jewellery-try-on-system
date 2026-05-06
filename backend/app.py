# ═══════════════════════════════════════════════════════════════════
#  SG Jewels · app.py  v7
#  NEW: necklace processing — BG removal + chain-end crop
#  ENHANCED: earring design highlight via contrast boost on export
# ═══════════════════════════════════════════════════════════════════

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, JSONResponse
import numpy as np
import cv2
from PIL import Image, ImageEnhance
import io
import traceback
from typing import Optional

try:
    from rembg import remove as _rembg
    REMBG_AVAILABLE = True
    print("[SG Jewels] ✅ rembg loaded")
except ImportError:
    REMBG_AVAILABLE = False
    print("[SG Jewels] ⚠️  rembg not found — pip install rembg onnxruntime")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════════════════════
#  BACKGROUND REMOVAL (shared)
# ═══════════════════════════════════════════════════════════════════

def _rembg_remove(pil_rgb: Image.Image):
    if not REMBG_AVAILABLE:
        return None
    try:
        out  = _rembg(pil_rgb)
        rgba = np.array(out.convert("RGBA"))
        if rgba[:, :, 3].min() >= 254:
            return None
        return rgba
    except Exception:
        traceback.print_exc()
        return None


def _grabcut_remove(pil_rgb: Image.Image) -> np.ndarray:
    ow, oh = pil_rgb.size
    s = min(600/ow, 600/oh, 1.0)
    w, h = max(int(ow*s), 100), max(int(oh*s), 100)
    img  = np.array(pil_rgb.resize((w, h), Image.LANCZOS).convert("RGB"))
    msk  = np.zeros((h, w), np.uint8)
    bg, fg = np.zeros((1,65), np.float64), np.zeros((1,65), np.float64)
    pad  = max(int(min(h,w)*0.07), 8)
    try:
        cv2.grabCut(img, msk, (pad,pad,w-2*pad,h-2*pad), bg, fg, 12, cv2.GC_INIT_WITH_RECT)
        a = np.where((msk==2)|(msk==0), 0, 255).astype(np.uint8)
    except Exception:
        a = np.full((h,w), 255, dtype=np.uint8)
    a = cv2.morphologyEx(a, cv2.MORPH_CLOSE, np.ones((5,5), np.uint8))
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
    af = cv2.GaussianBlur(a.astype(np.float32)/255.0, (7,7), 2)
    a  = (af*255).clip(0,255).astype(np.uint8)
    if s < 1.0:
        a = cv2.resize(a, (ow,oh), interpolation=cv2.INTER_LINEAR)
    rgba = np.array(pil_rgb.convert("RGBA"))
    rgba[:,:,3] = a
    return rgba


def remove_background(pil_rgb: Image.Image) -> np.ndarray:
    r = _rembg_remove(pil_rgb)
    if r is not None:
        return r
    return _grabcut_remove(pil_rgb)


# ═══════════════════════════════════════════════════════════════════
#  ENHANCEMENT — contrast + brightness boost for display clarity
#  Applied AFTER BG removal so the jewellery design pops on screen
# ═══════════════════════════════════════════════════════════════════

def enhance_jewellery(rgba: np.ndarray) -> np.ndarray:
    """
    Boost contrast and saturation of the jewellery design.
    This is what makes gold designs look vivid instead of dull.
    Only RGB channels are enhanced; alpha channel is preserved.
    """
    pil = Image.fromarray(rgba.astype(np.uint8), "RGBA")
    r, g, b, a = pil.split()
    rgb = Image.merge("RGB", (r, g, b))

    # Contrast boost — makes design details sharper and more visible
    rgb = ImageEnhance.Contrast(rgb).enhance(1.35)
    # Brightness slight lift — earrings look lit from ambient light
    rgb = ImageEnhance.Brightness(rgb).enhance(1.12)
    # Colour saturation — gold tones pop more
    rgb = ImageEnhance.Color(rgb).enhance(1.30)
    # Sharpness — fine filigree details become crisper
    rgb = ImageEnhance.Sharpness(rgb).enhance(2.0)

    r2, g2, b2 = rgb.split()
    enhanced = Image.merge("RGBA", (r2, g2, b2, a))
    return np.array(enhanced)


# ═══════════════════════════════════════════════════════════════════
#  EARRING EXTRACTION (unchanged from v6)
# ═══════════════════════════════════════════════════════════════════

class JewelleryError(Exception):
    pass


def _clean_mask(alpha: np.ndarray) -> np.ndarray:
    _, b = cv2.threshold(alpha, 10, 255, cv2.THRESH_BINARY)
    k = np.ones((7,7), np.uint8)
    b = cv2.morphologyEx(b, cv2.MORPH_CLOSE, k)
    b = cv2.morphologyEx(b, cv2.MORPH_OPEN,  np.ones((3,3), np.uint8))
    return b


def _good_contours(binary, img_area):
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    t = max(80, img_area * 0.0008)
    cnts = [c for c in cnts if cv2.contourArea(c) > t]
    cnts.sort(key=cv2.contourArea, reverse=True)
    return cnts


def _tight_crop(rgba: np.ndarray, pad=12) -> np.ndarray:
    a = rgba[:,:,3]
    rows = np.any(a > 15, axis=1); cols = np.any(a > 15, axis=0)
    if not rows.any() or not cols.any(): return rgba
    H, W = rgba.shape[:2]
    r1,r2 = np.where(rows)[0][[0,-1]]; c1,c2 = np.where(cols)[0][[0,-1]]
    return rgba[max(0,r1-pad):min(H,r2+pad), max(0,c1-pad):min(W,c2+pad)]


def _crop_one_contour(rgba, cnt, pad=14):
    H, W = rgba.shape[:2]
    x,y,cw,ch = cv2.boundingRect(cnt)
    x1,y1 = max(0,x-pad), max(0,y-pad)
    x2,y2 = min(W,x+cw+pad), min(H,y+ch+pad)
    solo = np.zeros((H,W), np.uint8)
    cv2.drawContours(solo, [cnt], -1, 255, cv2.FILLED)
    out = rgba.copy(); out[:,:,3] = np.minimum(out[:,:,3], solo)
    return out[y1:y2, x1:x2]


def _earring_score(c):
    x,y,cw,ch = cv2.boundingRect(c)
    return cv2.contourArea(c) * max(1.0, ch/cw if cw>0 else 1)


def _find_alpha_split_col(alpha):
    H, W = alpha.shape
    lo, hi = int(W*0.25), int(W*0.75)
    col_sums = alpha[:, lo:hi].sum(axis=0).astype(np.float32)
    smooth   = np.convolve(col_sums, np.ones(7)/7, mode='same')
    split_l  = int(np.argmin(smooth))
    split_c  = lo + split_l
    mn, mx   = smooth[split_l], smooth.max()
    contrast = 1.0 - (mn/mx) if mx > 0 else 0
    print(f"[alpha-split] col={split_c} contrast={contrast:.2f}")
    if contrast < 0.15:
        return None
    return split_c


def _extract_left_half(rgba, split_col):
    H, W = rgba.shape[:2]
    pad  = max(int(W*0.04), 10)
    cut  = min(split_col + pad, W)
    return _tight_crop(rgba[:, :cut].copy())


def extract_earring(rgba: np.ndarray, pil_rgb: Image.Image):
    H, W  = rgba.shape[:2]
    alpha = rgba[:,:,3]
    if alpha.max() < 15:
        raise JewelleryError("No jewellery detected. Please upload a clearer earring image.")

    # Try alpha-column split first
    sc = _find_alpha_split_col(alpha)
    if sc is not None:
        left = _extract_left_half(rgba, sc)
        if left.shape[0] > 20 and left.shape[1] > 20:
            return left, "pair"

    binary = _clean_mask(alpha)
    cnts   = _good_contours(binary, H*W)

    if not cnts:
        cropped = _tight_crop(rgba)
        if cropped.shape[0] > 20: return cropped, "single"
        raise JewelleryError("No earring detected. Try a cleaner background.")

    if len(cnts) == 1:
        x,y,cw,ch = cv2.boundingRect(cnts[0])
        if cw>0 and cw/ch > 2.8:
            raise JewelleryError("This looks like a necklace. Please upload an earring image.")
        return _crop_one_contour(rgba, cnts[0]), "single"

    a0,a1 = cv2.contourArea(cnts[0]), cv2.contourArea(cnts[1])
    if a0>0 and a1/a0 > 0.25:
        best = max(cnts[:2], key=_earring_score)
        return _crop_one_contour(rgba, best), "pair"
    return _crop_one_contour(rgba, cnts[0]), "single"


# ═══════════════════════════════════════════════════════════════════
#  NECKLACE EXTRACTION — NEW
# ═══════════════════════════════════════════════════════════════════

def extract_necklace(rgba: np.ndarray) -> np.ndarray:
    """
    Extract necklace from a BG-removed RGBA image.

    Strategy:
    1. Find all foreground contours in the alpha mask.
    2. The necklace is the LARGEST connected component.
    3. Find its bounding box.
    4. Detect chain ends: scan for the leftmost and rightmost non-zero
       alpha columns at the TOP 30% of the bounding box (chain hangs there).
    5. Crop from left-chain-end to right-chain-end, full height.
    6. This gives a crop that spans exactly from one chain end to the other.
    """
    H, W  = rgba.shape[:2]
    alpha = rgba[:,:,3]

    if alpha.max() < 15:
        raise JewelleryError("No necklace detected. Please upload a clearer image.")

    binary = _clean_mask(alpha)
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not cnts:
        # Fallback: tight crop of entire BG-removed image
        return _tight_crop(rgba)

    # Take the single largest contour = the necklace
    main = max(cnts, key=cv2.contourArea)
    x, y, cw, ch = cv2.boundingRect(main)

    # Validate: necklace is typically WIDE (aspect ratio > 0.8)
    aspect = cw / ch if ch > 0 else 1
    if aspect < 0.4:
        # Too narrow — might be earring uploaded by mistake
        raise JewelleryError(
            "This looks like an earring, not a necklace. "
            "Please upload a necklace image."
        )

    # ── Find chain ends in top 30% of the necklace bounding box ──
    # The chain/clasp top edge is where the necklace is worn from
    top_region_h = max(int(ch * 0.30), 10)
    top_alpha    = alpha[y : y + top_region_h, x : x + cw]

    # Columns in the top region that have ANY foreground pixels
    col_has_fg   = np.any(top_alpha > 20, axis=0)
    fg_cols      = np.where(col_has_fg)[0]

    if len(fg_cols) > 0:
        chain_left  = x + fg_cols[0]          # leftmost chain-end column
        chain_right = x + fg_cols[-1]         # rightmost chain-end column
    else:
        chain_left  = x
        chain_right = x + cw

    # Add padding
    pad = 16
    x1 = max(0, chain_left  - pad)
    x2 = min(W, chain_right + pad)
    y1 = max(0, y - pad)
    y2 = min(H, y + ch + pad)

    print(f"[necklace] bbox x={x} y={y} w={cw} h={ch} aspect={aspect:.2f}")
    print(f"[necklace] chain ends: left={chain_left} right={chain_right}")

    # Zero out pixels outside the main necklace contour
    solo = np.zeros((H, W), np.uint8)
    cv2.drawContours(solo, [main], -1, 255, cv2.FILLED)
    out = rgba.copy()
    out[:,:,3] = np.minimum(out[:,:,3], solo)

    return out[y1:y2, x1:x2]


# ═══════════════════════════════════════════════════════════════════
#  ROUTE
# ═══════════════════════════════════════════════════════════════════

@app.post("/process-image")
async def process_image(
    file: UploadFile = File(...),
    type: Optional[str] = Form("earring")   # "earring" | "necklace"
):
    try:
        ct = (file.content_type or "").lower()
        if not ct.startswith("image/"):
            return JSONResponse(400, content={"error": "Please upload a valid image."})

        raw     = await file.read()
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
        ow, oh  = pil_img.size
        print(f"\n[req] type={type}  {file.filename}  {ow}×{oh}")

        # Step 1: BG removal
        rgba = remove_background(pil_img)
        if rgba.shape[:2] != (oh, ow):
            rgba = cv2.resize(rgba, (ow, oh), interpolation=cv2.INTER_LANCZOS4)

        # Step 2: Extract based on jewellery type
        try:
            if type == "necklace":
                processed = extract_necklace(rgba)
                j_type    = "necklace"
            else:
                processed, j_type = extract_earring(rgba, pil_img)
        except JewelleryError as je:
            return JSONResponse(400, content={"error": str(je)})

        # Step 3: ENHANCE — boost design visibility
        processed = enhance_jewellery(processed)

        ph, pw = processed.shape[:2]
        print(f"[out] {pw}×{ph}  type={j_type}")

        # Step 4: Lossless PNG
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