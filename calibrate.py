"""
Clearcall — HSV color calibration helper.

Use this to find the LOWER_COLOR / UPPER_COLOR values for your ball under your
actual lighting. It prints the HSV value at the center of the frame, so you hold
your ball in the middle of the camera view and read the numbers.

Run it the same way as clearcall.py (set VIDEO_SOURCE to match), open the stream
in a browser, center the ball, and watch the terminal for the printed HSV value.
Then set LOWER_COLOR slightly below and UPPER_COLOR slightly above those numbers
in clearcall.py.
"""

import cv2
import numpy as np

# Match this to your clearcall.py setup.
VIDEO_SOURCE = "rtsp://10.0.0.27:554/live"

cap = cv2.VideoCapture(VIDEO_SOURCE, cv2.CAP_FFMPEG) if isinstance(VIDEO_SOURCE, str) else cv2.VideoCapture(VIDEO_SOURCE)

print("Hold your ball in the CENTER of the camera view.")
print("Reading the HSV value at the center of each frame...\n")

while True:
    ok, frame = cap.read()
    if not ok:
        continue
    frame = cv2.resize(frame, (640, 480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Sample the center pixel.
    h, s, v = hsv[240, 320]
    print(f"Center HSV -> H:{h:3d}  S:{s:3d}  V:{v:3d}", end="\r")
