# server.py
# FastAPI server that accepts an uploaded image, removes background using rembg,
# and returns a transparent PNG. CORS enabled for local frontend testing.

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from rembg import remove
from PIL import Image
import io
import uvicorn

app = FastAPI()

# Allow local dev origins (adjust as needed for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development; in production set explicit origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/process-image")
async def process_image(file: UploadFile = File(...)):
    """
    Accepts file upload (image), strips background using rembg, and returns PNG bytes.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    try:
        content = await file.read()
        input_stream = io.BytesIO(content)

        # Open as PIL image (rembg expects bytes / PIL)
        img = Image.open(input_stream).convert("RGBA")

        # Use rembg to remove background; returns bytes
        # rembg.remove accepts bytes or numpy array / PIL image
        output_bytes = remove(content)  # returns PNG bytes when source has alpha

        # Ensure we return PNG bytes with transparent bg
        out_io = io.BytesIO(output_bytes)
        out_io.seek(0)
        return StreamingResponse(out_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {e}")

if __name__ == "__main__":
    # Run with: python server.py  OR recommended: uvicorn server:app --reload --port 8000
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)
