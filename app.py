import gradio as gr
import cv2
import numpy as np
import os
import threading

# ==================== ENHANCEMENT FUNCTION ====================
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

# ==================== VIDEO PROCESSING ====================
def process_video(video, clip_limit, color_boost, pause_event):
    if video is None:
        return None, "Upload a video first", ""

    temp_input = "temp_input.mp4"
    video.save(temp_input)

    cap = cv2.VideoCapture(temp_input)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    out_path = "enhanced_output.mp4"
    out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w*2, h))

    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    while cap.isOpened():
        if pause_event.is_set():
            status = "Paused – click Resume to continue"
            return out_path, status, status

        ret, frame = cap.read()
        if not ret:
            break

        enhanced = enhance_frame(frame, clip_limit, color_boost)
        combined = np.hstack((frame, enhanced))
        out.write(combined)
        frame_count += 1

    cap.release()
    out.release()

    return out_path, f"Success! Processed {frame_count} frames.", "Done! Download below."

# ==================== PAUSE / RESUME LOGIC ====================
pause_event = threading.Event()

def toggle_pause():
    if pause_event.is_set():
        pause_event.clear()
        return "Paused", "Resume"
    else:
        pause_event.set()
        return "Resumed", "Pause"

# ==================== MODERN UI ====================
theme = gr.themes.Soft(
    primary_hue="cyan",
    secondary_hue="teal",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'sans-serif'],
)

with gr.Blocks(theme=theme, css="""
    .gradio-container {background: linear-gradient(to bottom, #0f172a, #1e293b);}
    h1 {color: #67e8f9; text-shadow: 0 0 10px #0ea5e9;}
    .gr-button {background: #0ea5e9 !important; border: none;}
    .gr-button:hover {background: #38bdf8 !important;}
""") as demo:
    gr.Markdown(
        """
        # 🌊 Real-Time Underwater Video Enhancer
        **Simulation Phase** – Restore natural colors & contrast in underwater footage
        """,
        elem_classes="text-center text-4xl font-bold mb-4"
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Controls")
            clip_limit = gr.Slider(0.5, 6.0, 2.5, 0.1, label="CLAHE Clip Limit (contrast)")
            color_boost = gr.Slider(0.5, 3.0, 1.0, 0.1, label="Color Boost Factor (red recovery)")

        with gr.Column(scale=3):
            gr.Markdown("### Upload & Preview")
            input_video = gr.Video(label="Upload Underwater Video", sources="upload", format="mp4")
            output_video = gr.Video(label="Enhanced Video (Original | Enhanced)", interactive=False)

    with gr.Row():
        btn_enhance = gr.Button("Enhance Video", variant="primary", scale=0)
        btn_pause = gr.Button("Pause", variant="secondary")
        download_btn = gr.DownloadButton("Download Enhanced Video", visible=False)

    status = gr.Textbox(label="Status", interactive=False, lines=2)

    # Event handlers
    def start_enhance(video, cl, cb):
        pause_event.clear()  # make sure not paused
        return process_video(video, cl, cb, pause_event)

    btn_enhance.click(
        start_enhance,
        inputs=[input_video, clip_limit, color_boost],
        outputs=[output_video, status]
    )

    btn_pause.click(
        toggle_pause,
        outputs=[status, btn_pause]
    )

    # Show download button when output is ready
    output_video.change(
        lambda v: gr.update(visible=bool(v)),
        inputs=output_video,
        outputs=download_btn
    )

    download_btn.click(
        lambda: "enhanced_output.mp4",
        outputs=gr.File()
    )

demo.launch(server_name="0.0.0.0", server_port=7860)
