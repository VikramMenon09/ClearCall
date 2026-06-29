"""
Clearcall — prototype boundary-call system.

Detects a colored ball in a live video feed, tracks its position, and fires an
"OUT OF BOUNDS" call when the ball crosses a user-defined line. The annotated
video is streamed to any browser on the same network, where tapping/clicking
the video sets the boundary line.

This is a proof-of-concept that runs on a Raspberry Pi 4 using OpenCV color
tracking (no GPU / no PyTorch required). The production Clearcall system uses a
trained vision model on faster edge hardware for full player + ball tracking.
"""

import cv2
import numpy as np
import threading
from flask import Flask, Response, request

# ----------------------------------------------------------------------------
# CONFIG — edit these for your setup
# ----------------------------------------------------------------------------

# Video source. Options:
#   0                                  -> a USB webcam plugged into the Pi
#   "rtsp://USER:PASS@IP:PORT/live"    -> a phone IP-camera app over WiFi
#   "picam"                            -> the Raspberry Pi Camera Module (see README)
VIDEO_SOURCE = "rtsp://10.0.0.27:554/live"

# Ball color range in HSV. Defaults target an ORANGE ball.
# Run calibrate.py to find values for a different ball / lighting.
LOWER_COLOR = np.array([5, 120, 120])
UPPER_COLOR = np.array([20, 255, 255])

# Minimum blob size (pixels) to count as the ball — filters out small specks.
MIN_BALL_AREA = 300

# Starting position of the boundary line (x-pixel). Click the video to move it.
DEFAULT_BOUNDARY_X = 320

# Frame size the system works in.
FRAME_W, FRAME_H = 640, 480

# Web server port.
PORT = 5000

# ----------------------------------------------------------------------------

app = Flask(__name__)
cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG) if isinstance(VIDEO_SOURCE, str) else cv2.VideoCapture(VIDEO_SOURCE)

state = {"frame": None, "boundary_x": DEFAULT_BOUNDARY_X, "out": False}
lock = threading.Lock()


def camera_loop():
    """Continuously grab frames, detect the ball, and decide the call."""
    while True:
        ok, frame = cap.read()
        if not ok:
            continue

        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Mask to only the ball-colored pixels, then clean up noise.
        mask = cv2.inRange(hsv, LOWER_COLOR, UPPER_COLOR)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        out_of_bounds = False
        bx = state["boundary_x"]

        if contours:
            # Largest blob of the target color is assumed to be the ball.
            c = max(contours, key=cv2.contourArea)
            if cv2.contourArea(c) > MIN_BALL_AREA:
                (x, y), radius = cv2.minEnclosingCircle(c)
                cx = int(x)
                cv2.circle(frame, (cx, int(y)), int(radius), (0, 165, 255), 2)
                cv2.putText(frame, "BALL", (cx - 20, int(y) - int(radius) - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
                if cx > bx:                 # crossed to the right of the line
                    out_of_bounds = True

        # Draw the boundary line.
        cv2.line(frame, (bx, 0), (bx, FRAME_H), (0, 0, 255), 3)

        # Fire the call banner.
        if out_of_bounds:
            cv2.rectangle(frame, (0, 0), (FRAME_W, 50), (0, 180, 0), -1)
            cv2.putText(frame, "OUT OF BOUNDS", (20, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 3)

        _, jpeg = cv2.imencode(".jpg", frame)
        with lock:
            state["frame"] = jpeg.tobytes()
            state["out"] = out_of_bounds


def mjpeg():
    """Yield the latest annotated frame as an MJPEG stream."""
    while True:
        with lock:
            f = state["frame"]
        if f is None:
            continue
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + f + b"\r\n")


PAGE = """
<!doctype html><html><head><meta name=viewport content="width=device-width,initial-scale=1">
<title>Clearcall</title>
<style>body{margin:0;background:#111;font-family:sans-serif;text-align:center}
img{width:100%;max-width:640px}p{color:#ccc}</style></head>
<body><p>Tap the video where your line should be</p>
<img id="v" src="/stream" onclick="setLine(event)">
<script>
function setLine(e){
  const img=document.getElementById('v');const r=img.getBoundingClientRect();
  const x=Math.round((e.clientX-r.left)/r.width*640);
  fetch('/setline?x='+x);
}
</script></body></html>
"""


@app.route("/")
def index():
    return PAGE


@app.route("/stream")
def stream():
    return Response(mjpeg(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")


@app.route("/setline")
def setline():
    state["boundary_x"] = int(request.args.get("x", DEFAULT_BOUNDARY_X))
    return "ok"


if __name__ == "__main__":
    threading.Thread(target=camera_loop, daemon=True).start()
    app.run(host="0.0.0.0", port=PORT)
