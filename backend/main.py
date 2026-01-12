from fastapi import FastAPI, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import cv2
import uuid
import os
import subprocess
import threading
import logging
from ultralytics import YOLO

logging.basicConfig(level=logging.INFO)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
app = FastAPI()

os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "hls"), exist_ok=True)

model = YOLO("yolov8n.pt")
FFMPEG_PATH = "ffmpeg" # ffmpeg issues

def process_to_hls_live(input_path, output_dir):
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    os.makedirs(output_dir, exist_ok=True)

    cmd = [ # ffmpeg required to stream hls (.ts) segments as a true live video
        FFMPEG_PATH,
        "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}",
        "-r", str(fps),
        "-i", "-",  
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-f", "hls",
        "-hls_time", "1",
        "-hls_list_size", "6",
        "-hls_flags", "delete_segments+append_list+independent_segments",
        "-hls_playlist_type", "event",
        os.path.join(output_dir, "index.m3u8")
    ]

    logging.info(f"Starting live HLS FFmpeg: {' '.join(cmd)}") # logging
    process = subprocess.Popen(cmd, stdin=subprocess.PIPE)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        frame = results[0].plot()
        try:
            process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            logging.error("FFmpeg pipe broken") # throw error if stopped
            break

    cap.release()
    process.stdin.close()
    process.wait()
    logging.info(f"Live HLS finished: {output_dir}")

@app.post("/upload")
async def upload_video(file: UploadFile):
    video_id = str(uuid.uuid4())
    input_path = os.path.join(BASE_DIR, "uploads", f"{video_id}.mp4") 
    output_dir = os.path.join(BASE_DIR, "hls", video_id)

    with open(input_path, "wb") as f:
        f.write(await file.read())

    threading.Thread(target=process_to_hls_live, args=(input_path, output_dir), daemon=True).start()
    return JSONResponse({"video_id": video_id})

@app.get("/hls_ready/{video_id}")
def hls_ready(video_id: str):
    playlist = os.path.join(BASE_DIR, "hls", video_id, "index.m3u8")
    return PlainTextResponse("ready" if os.path.exists(playlist) else "not_ready")

app.mount("/hls", StaticFiles(directory=os.path.join(BASE_DIR, "hls")), name="hls")
app.mount("/", StaticFiles(directory=os.path.join(BASE_DIR, "frontend"), html=True), name="frontend")
