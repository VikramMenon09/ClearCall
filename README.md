# Clearcall — Prototype

A proof-of-concept for [Clearcall](https://github.com/), an affordable AI-powered
sports officiating system for high school and amateur sports. This prototype runs
on a Raspberry Pi 4 and demonstrates the core officiating loop: **detect the ball,
track its position, and automatically call when it crosses a boundary line.**

The boundary is software-defined — you point the camera at any surface, click to
set the line, and the system enforces it. In a real gym that line is a sideline or
the three-point arc; in a classroom demo it's a strip of tape on a desk. The
detection and the call are identical either way.

> This prototype uses lightweight OpenCV color tracking so it runs on a Pi 4 with
> no GPU and no PyTorch. The production system uses a trained vision model (YOLO-class)
> on faster edge hardware (e.g. NVIDIA Jetson) for full player + ball tracking and
> additional rule types. The prototype proves the concept on minimal hardware.

## Try it in the browser

**Live demo: https://clearcall-delta.vercel.app**

You don't need a Raspberry Pi to try it. The `web/` folder re-implements the same
pipeline in plain JavaScript on a canvas, and everything runs locally in the page:

- **Demo mode** (no camera): a simulated ball moves around a court. Each frame is
  rendered to pixels and the ball is found by color, just like with a real camera.
- **Camera mode**: uses your webcam via `getUserMedia`. Video never leaves your device.
- **Set the line**: click to drop a vertical line (like the Pi version), or drag to
  draw one at any angle. Use *Flip OUT side* to choose which side is out.
- **Calibrate**: press *Calibrate*, then click the ball to sample its color. This
  does the same job as `calibrate.py`. Fine-tune the HSV range with sliders.
- **Show what the computer sees**: toggle the color mask to see the thresholded frame.

Pipeline per frame: RGB → HSV → `inRange` → erode/dilate → largest blob → center →
which side of the line → **IN / OUT** call, with a short trail.

To run it locally, serve the folder (the camera needs `localhost` or HTTPS):

```bash
cd web && python3 -m http.server 8000
```

## How it works

```
camera  ->  Raspberry Pi (OpenCV)  ->  browser
            1. detect ball by color
            2. track its center point
            3. check if it crossed the line
            4. draw the "OUT OF BOUNDS" call
```

A Flask server streams the annotated video to any browser on the same network.
Clicking/tapping the video sets the boundary line.

## Hardware

- Raspberry Pi 4 (4GB)
- A camera — any of:
  - USB webcam (plug into the Pi)
  - Raspberry Pi Camera Module 3 (ribbon cable)
  - A phone running an IP-camera app, streaming over WiFi (used during early testing)
- An orange ball (or any solid-colored ball — recalibrate the color range)

## Setup

```bash
sudo apt update
sudo apt install -y python3-opencv python3-flask
```

(No PyTorch / ultralytics needed — this version is pure OpenCV.)

## Configure

Open `clearcall.py` and set `VIDEO_SOURCE` near the top:

```python
VIDEO_SOURCE = 0                                 # USB webcam
VIDEO_SOURCE = "rtsp://USER:PASS@10.0.0.27:554/live"  # phone IP-camera app
```

If your ball isn't orange, run the calibration helper to find your color values:

```bash
python calibrate.py
```

Hold the ball in the center of the view, read the printed HSV numbers, and set
`LOWER_COLOR` / `UPPER_COLOR` in `clearcall.py` just below / above them.

## Run

```bash
python clearcall.py
```

Then on any device on the same WiFi, open:

```
http://<pi-ip>:5000
```

Find the Pi's IP with `hostname -I`. Tap the video to set the boundary line, then
roll the ball across it — the call fires when the ball's center crosses.

## Notes

- The boundary line is **vertical** by default (detects left/right crossing). To
  detect front/back crossing, check the ball's `y` against a horizontal line instead.
- Frame rate over a network (phone) stream is modest; a directly-attached camera is
  much faster.

## Roadmap

- [ ] Swap color tracking for a trained model (player + ball detection)
- [ ] Multi-camera fusion for occlusion handling
- [ ] Additional rule types (boundary crossings for multiple sports)
- [ ] Optional UWB sensor input for tracking when the view is blocked
- [ ] Edge hardware (Jetson) for real-time frame rates
