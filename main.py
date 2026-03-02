from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import os
from pathlib import Path

app = FastAPI(title="Underwater Video Enhancer")

# Create folders
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory="static"), name="static")

def enhance_frame(frame, clip_limit=2.5, color_boost=1.0):
    frame_float = frame.astype(np.float32) / 255.0
    b, g, r = cv2.split(frame_float)

    mean_b = np.mean(b)
    mean_g = np.mean(g)
    mean_r = np.mean(r)
    max_mean = max(mean_b, mean_g, mean_r)

    r_corrected = np.clip(r * (color_boost * max_mean / (mean_r + 1e-6)), 0, 1)
    g_corrected = np.clip(g * (color_boost * max_mean / (mean_g + 1e-6)), 0, 1)
    b_corrected = b

    corrected = cv2.merge([b_corrected, g_corrected, r_corrected])

    corrected_uint8 = (corrected * 255).astype(np.uint8)
    lab = cv2.cvtColor(corrected_uint8, cv2.COLOR_BGR2LAB)
    l, a, b_lab = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)

    lab_enhanced = cv2.merge([l_enhanced, a, b_lab])
    enhanced = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)
    return enhanced

@app.get("/", response_class=HTMLResponse)
async def home():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Underwater Video Enhancer</title>
        <style>
            body { font-family: Arial; text-align: center; background: #0a1f3d; color: white; }
            h1 { color: #00ccff; }
            .container { max-width: 900px; margin: 40px auto; }
            input, button { padding: 10px; margin: 10px; font-size: 16px; }
            button { background: #00ccff; color: black; border: none; border-radius: 8px; cursor: pointer; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🌊 Underwater Video Enhancer</h1>
            <p>Upload a video and enhance it in real-time</p>
            
            <form action="/enhance" enctype="multipart/form-data" method="post">
                <input type="file" name="video" accept="video/*" required><br><br>
                
                <label>CLAHE Clip Limit: <input type="range" name="clip_limit" min="0.5" max="6.0" value="2.5" step="0.1"></label><br>
                <label>Color Boost Factor: <input type="range" name="color_boost" min="0.5" max="3.0" value="1.0" step="0.1"></label><br><br>
                
                <button type="submit">Enhance Video</button>
            </form>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/enhance")
async def enhance(video: UploadFile = File(...), clip_limit: float = Form(2.5), color_boost: float = Form(1.0)):
    # Save uploaded video
    input_path = UPLOAD_DIR / video.filename
    with open(input_path, "wb") as f:
        f.write(await video.read())

    # Process video
    cap = cv2.VideoCapture(str(input_path))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    output_path = OUTPUT_DIR / f"enhanced_{video.filename}"
    out = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*'mp4v'), fps, (width*2, height))

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        enhanced = enhance_frame(frame, clip_limit, color_boost)
        combined = np.hstack((frame, enhanced))
        out.write(combined)
        frame_count += 1

    cap.release()
    out.release()

    return FileResponse(output_path, media_type="video/mp4", filename=f"enhanced_{video.filename}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
