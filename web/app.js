/*
 * ClearCall, browser edition.
 *
 * A plain-JavaScript port of clearcall.py / calibrate.py:
 *   frame -> HSV -> inRange -> erode/dilate -> largest blob -> center -> side of line -> call
 *
 * Runs on a live webcam (getUserMedia) or on a simulated court ("Demo"), where a
 * ball is rendered into pixels and then found by the same color pipeline. The demo
 * never reads the ball's true position; it has to detect it like the camera mode does.
 */
(() => {
  "use strict";

  // ---------------------------------------------------------------------------
  // CONFIG (mirrors clearcall.py)
  // ---------------------------------------------------------------------------
  const FRAME_W = 640, FRAME_H = 480;          // display / "camera" frame size
  const SCALE = 2;                              // process at half resolution for speed
  const PW = FRAME_W / SCALE, PH = FRAME_H / SCALE;
  const MIN_BALL_AREA = 300 / (SCALE * SCALE);  // 300 px at full res
  const DEFAULT_LOWER = [5, 120, 120];          // orange ball, OpenCV HSV ranges
  const DEFAULT_UPPER = [20, 255, 255];         // H 0-179, S 0-255, V 0-255
  const TRAIL_LEN = 28;
  const LOST_FRAMES = 12;                       // frames without a ball before "no ball"

  // ---------------------------------------------------------------------------
  // DOM
  // ---------------------------------------------------------------------------
  const $ = (id) => document.getElementById(id);
  const view = $("view");
  const ctx = view.getContext("2d");
  const stage = $("stage");
  const video = $("video");
  const callEl = $("call"), callText = $("call-text"), hintEl = $("hint");

  const proc = document.createElement("canvas");
  proc.width = PW; proc.height = PH;
  const pctx = proc.getContext("2d", { willReadFrequently: true });

  const scene = document.createElement("canvas");  // demo "camera"
  scene.width = FRAME_W; scene.height = FRAME_H;
  const sctx = scene.getContext("2d");

  // ---------------------------------------------------------------------------
  // STATE
  // ---------------------------------------------------------------------------
  const state = {
    mode: "demo",
    lower: DEFAULT_LOWER.slice(),
    upper: DEFAULT_UPPER.slice(),
    line: null,          // {ax, ay, bx, by} in frame coords
    outSign: 1,          // which side of the line is OUT (+1 / -1)
    calibrating: false,
    showMask: false,
    showTrail: true,
    paused: false,
    trail: [],
    call: null,          // "IN" | "OUT" | null
    lostFrames: LOST_FRAMES,
    calls: 0,
    lastImage: null,
    ball: null,
    stream: null,
    t0: performance.now(),
  };

  // Buffers reused every frame.
  const N = PW * PH;
  const mask = new Uint8Array(N), tmp = new Uint8Array(N), clean = new Uint8Array(N);
  const labels = new Int32Array(N), stack = new Int32Array(N);
  const maskImg = new ImageData(PW, PH);

  // ---------------------------------------------------------------------------
  // VISION PIPELINE
  // ---------------------------------------------------------------------------

  // RGB -> OpenCV-style HSV (H 0-179, S 0-255, V 0-255).
  function hsv(r, g, b) {
    const max = r > g ? (r > b ? r : b) : (g > b ? g : b);
    const min = r < g ? (r < b ? r : b) : (g < b ? g : b);
    const d = max - min;
    let h = 0;
    if (d !== 0) {
      if (max === r) h = 60 * ((g - b) / d);
      else if (max === g) h = 60 * ((b - r) / d) + 120;
      else h = 60 * ((r - g) / d) + 240;
      if (h < 0) h += 360;
    }
    return [Math.round(h / 2) % 180, max === 0 ? 0 : Math.round((255 * d) / max), max];
  }

  // cv2.inRange, with hue wrap-around support for red balls (lowH > highH).
  function threshold(data) {
    const [lh, ls, lv] = state.lower, [uh, us, uv] = state.upper;
    const wrap = lh > uh;
    for (let i = 0, p = 0; i < N; i++, p += 4) {
      const [h, s, v] = hsv(data[p], data[p + 1], data[p + 2]);
      const hOk = wrap ? (h >= lh || h <= uh) : (h >= lh && h <= uh);
      mask[i] = hOk && s >= ls && s <= us && v >= lv && v <= uv ? 1 : 0;
    }
  }

  // 3x3 rectangular erode / dilate, done as two separable passes.
  function morph(src, dst, erode) {
    for (let y = 0; y < PH; y++) {
      const row = y * PW;
      for (let x = 0; x < PW; x++) {
        const i = row + x;
        let v = src[i];
        if (x > 0) v = erode ? v & src[i - 1] : v | src[i - 1];
        if (x < PW - 1) v = erode ? v & src[i + 1] : v | src[i + 1];
        tmp[i] = v;
      }
    }
    for (let y = 0; y < PH; y++) {
      for (let x = 0; x < PW; x++) {
        const i = y * PW + x;
        let v = tmp[i];
        if (y > 0) v = erode ? v & tmp[i - PW] : v | tmp[i - PW];
        if (y < PH - 1) v = erode ? v & tmp[i + PW] : v | tmp[i + PW];
        dst[i] = v;
      }
    }
  }

  // Largest 8-connected blob = the ball (like max(contours, key=contourArea)).
  function largestBlob(m) {
    labels.fill(0);
    let best = null, label = 0;
    for (let start = 0; start < N; start++) {
      if (!m[start] || labels[start]) continue;
      label++;
      let sp = 0, area = 0, sx = 0, sy = 0;
      let minx = PW, maxx = 0, miny = PH, maxy = 0;
      stack[sp++] = start; labels[start] = label;
      while (sp) {
        const i = stack[--sp];
        const x = i % PW, y = (i / PW) | 0;
        area++; sx += x; sy += y;
        if (x < minx) minx = x; if (x > maxx) maxx = x;
        if (y < miny) miny = y; if (y > maxy) maxy = y;
        for (let dy = -1; dy <= 1; dy++) {
          const ny = y + dy;
          if (ny < 0 || ny >= PH) continue;
          for (let dx = -1; dx <= 1; dx++) {
            const nx = x + dx;
            if (nx < 0 || nx >= PW) continue;
            const j = ny * PW + nx;
            if (m[j] && !labels[j]) { labels[j] = label; stack[sp++] = j; }
          }
        }
      }
      if (!best || area > best.area) {
        best = { area, cx: sx / area, cy: sy / area, r: Math.max(maxx - minx + 1, maxy - miny + 1) / 2 };
      }
    }
    if (!best || best.area <= MIN_BALL_AREA) return null;
    return {
      x: (best.cx + 0.5) * SCALE, y: (best.cy + 0.5) * SCALE,
      r: best.r * SCALE, area: best.area * SCALE * SCALE,
    };
  }

  function detect() {
    const img = pctx.getImageData(0, 0, PW, PH);
    state.lastImage = img;
    threshold(img.data);
    morph(mask, clean, true);   // erode (2 iterations at full res ~ 1 at half res)
    morph(clean, mask, false);  // dilate
    return largestBlob(mask);
  }

  // ---------------------------------------------------------------------------
  // LINE GEOMETRY
  // ---------------------------------------------------------------------------
  function sideOf(px, py, L = state.line) {
    const c = (L.bx - L.ax) * (py - L.ay) - (L.by - L.ay) * (px - L.ax);
    return c > 0 ? 1 : c < 0 ? -1 : 0;
  }

  function setLine(ax, ay, bx, by) {
    state.line = { ax, ay, bx, by };
    // OUT defaults to the right of the line (like the Pi version), or below a flat line.
    state.outSign = sideOf(FRAME_W, FRAME_H / 2) || sideOf(FRAME_W / 2, FRAME_H) || 1;
    state.trail = [];
    state.call = null;
    state.lostFrames = LOST_FRAMES;
    renderCall(false);
  }

  function defaultLine() {
    if (state.mode === "demo") setLine(COURT.ax, COURT.ay, COURT.bx, COURT.by);
    else setLine(FRAME_W / 2, 0, FRAME_W / 2, FRAME_H);
  }

  // Clip the frame rectangle to the OUT half-plane (one Sutherland–Hodgman step).
  function outPolygon() {
    const pts = [[0, 0], [FRAME_W, 0], [FRAME_W, FRAME_H], [0, FRAME_H]];
    const L = state.line, s = state.outSign;
    const f = ([x, y]) => s * ((L.bx - L.ax) * (y - L.ay) - (L.by - L.ay) * (x - L.ax));
    const res = [];
    for (let i = 0; i < pts.length; i++) {
      const a = pts[i], b = pts[(i + 1) % pts.length];
      const fa = f(a), fb = f(b);
      if (fa >= 0) res.push(a);
      if ((fa >= 0) !== (fb >= 0)) {
        const t = fa / (fa - fb);
        res.push([a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])]);
      }
    }
    return res;
  }

  // ---------------------------------------------------------------------------
  // DEMO COURT (a simulated camera)
  // ---------------------------------------------------------------------------
  const COURT = { ax: 404, ay: 0, bx: 452, by: FRAME_H };   // painted sideline
  const BALL_COLORS = [
    { name: "orange", hi: "#ffc193", mid: "#ff7a1a", lo: "#c4520c" },
    { name: "yellow", hi: "#f4ffa8", mid: "#d4f03c", lo: "#94ab18" },
    { name: "pink",   hi: "#f6b8ff", mid: "#d63cff", lo: "#8f1fb0" },
  ];
  const demo = { x: 180, y: 260, vx: 210, vy: 130, r: 17, color: 0, nextKick: 2 };
  const court = document.createElement("canvas");
  court.width = FRAME_W; court.height = FRAME_H;

  (function paintCourt() {
    const c = court.getContext("2d");
    // In-bounds (blue) and out-of-bounds (green) surfaces, split by the painted line.
    c.fillStyle = "#2c6e4e"; c.fillRect(0, 0, FRAME_W, FRAME_H);
    c.fillStyle = "#1f4e79";
    c.beginPath(); c.moveTo(0, 0); c.lineTo(COURT.ax, COURT.ay); c.lineTo(COURT.bx, COURT.by); c.lineTo(0, FRAME_H); c.fill();
    // Light from the top, a bit of grain, so the frame looks like a camera image.
    const g = c.createLinearGradient(0, 0, 0, FRAME_H);
    g.addColorStop(0, "rgba(255,255,255,0.10)"); g.addColorStop(1, "rgba(0,0,0,0.22)");
    c.fillStyle = g; c.fillRect(0, 0, FRAME_W, FRAME_H);
    const img = c.getImageData(0, 0, FRAME_W, FRAME_H), d = img.data;
    for (let i = 0; i < d.length; i += 4) {
      const n = (Math.random() - 0.5) * 14;
      d[i] += n; d[i + 1] += n; d[i + 2] += n;
    }
    c.putImageData(img, 0, 0);
    // Painted lines.
    c.strokeStyle = "rgba(240,244,248,0.92)"; c.lineCap = "butt";
    c.lineWidth = 7;
    c.beginPath(); c.moveTo(COURT.ax, COURT.ay); c.lineTo(COURT.bx, COURT.by); c.stroke();
    c.lineWidth = 5;
    c.beginPath(); c.moveTo(0, 150); c.lineTo(COURT.ax + (COURT.bx - COURT.ax) * (150 / FRAME_H), 150); c.stroke();
    c.beginPath(); c.arc(0, 330, 120, -Math.PI / 2, Math.PI / 2); c.stroke();
  })();

  function stepDemo(dt) {
    const b = demo;
    if (!state.paused) {
      b.nextKick -= dt;
      if (b.nextKick <= 0) {                 // occasional new "shot" so play keeps changing
        const speed = 170 + Math.random() * 170;
        const towardLine = b.x < COURT.ax ? 1 : -1;
        const ang = (Math.random() - 0.5) * 1.4;
        b.vx = towardLine * Math.cos(ang) * speed;
        b.vy = Math.sin(ang) * speed;
        b.nextKick = 1.6 + Math.random() * 2.2;
      }
      b.x += b.vx * dt; b.y += b.vy * dt;
      if (b.x < b.r) { b.x = b.r; b.vx = Math.abs(b.vx); }
      if (b.x > FRAME_W - b.r) { b.x = FRAME_W - b.r; b.vx = -Math.abs(b.vx); }
      if (b.y < b.r) { b.y = b.r; b.vy = Math.abs(b.vy); }
      if (b.y > FRAME_H - b.r) { b.y = FRAME_H - b.r; b.vy = -Math.abs(b.vy); }
    }
    sctx.drawImage(court, 0, 0);
    // Shadow.
    sctx.fillStyle = "rgba(0,0,0,0.35)";
    sctx.beginPath(); sctx.ellipse(b.x + 6, b.y + b.r * 0.9, b.r * 0.95, b.r * 0.4, 0, 0, Math.PI * 2); sctx.fill();
    // Ball with shading (the highlight falls outside the HSV range, like a real ball).
    const col = BALL_COLORS[b.color];
    const g = sctx.createRadialGradient(b.x - b.r * 0.35, b.y - b.r * 0.4, b.r * 0.1, b.x, b.y, b.r);
    g.addColorStop(0, col.hi); g.addColorStop(0.35, col.mid); g.addColorStop(1, col.lo);
    sctx.fillStyle = g;
    sctx.beginPath(); sctx.arc(b.x, b.y, b.r, 0, Math.PI * 2); sctx.fill();
  }

  // ---------------------------------------------------------------------------
  // SOURCE -> canvases
  // ---------------------------------------------------------------------------
  function drawSource(c, w, h) {
    if (state.mode === "demo") { c.drawImage(scene, 0, 0, w, h); return true; }
    if (!state.stream || video.readyState < 2) return false;
    // Center-crop the camera to 4:3, like cv2.resize into 640x480 but without stretching.
    const vw = video.videoWidth, vh = video.videoHeight, target = 4 / 3;
    let sw = vw, sh = vh;
    if (vw / vh > target) sw = vh * target; else sh = vw / target;
    c.drawImage(video, (vw - sw) / 2, (vh - sh) / 2, sw, sh, 0, 0, w, h);
    return true;
  }

  // ---------------------------------------------------------------------------
  // CALL LOGIC
  // ---------------------------------------------------------------------------
  function updateCall(ball) {
    let next = state.call;
    if (ball) {
      state.lostFrames = 0;
      const out = sideOf(ball.x, ball.y) === state.outSign;   // on the line counts as IN
      next = out ? "OUT" : "IN";
      if (state.showTrail) {
        state.trail.push({ x: ball.x, y: ball.y, out });
        if (state.trail.length > TRAIL_LEN) state.trail.shift();
      }
    } else if (++state.lostFrames >= LOST_FRAMES) {
      next = null;
      state.trail = [];
    }
    if (next !== state.call) {
      if (next === "OUT" && state.call === "IN") logCall("OUT", "crossed the line");
      if (next === "IN" && state.call === "OUT") logCall("IN", "back in play");
      state.call = next;
      renderCall(next === "OUT");
    }
  }

  function renderCall(flash) {
    callEl.classList.toggle("in", state.call === "IN");
    callEl.classList.toggle("out", state.call === "OUT");
    callEl.classList.toggle("none", !state.call);
    callText.textContent = state.call || "NO BALL";
  }

  function logCall(kind, note) {
    const log = $("log");
    if (state.calls === 0 && kind === "IN") return;           // nothing to log yet
    const empty = log.querySelector(".empty");
    if (empty) empty.remove();
    if (kind === "OUT") $("count").textContent = ++state.calls;
    const t = ((performance.now() - state.t0) / 1000).toFixed(1);
    const li = document.createElement("li");
    li.innerHTML = `<span><span class="${kind === "OUT" ? "tagout" : "tagin"}">${kind}</span> <span class="muted">${note}</span></span><span class="mono muted">${t}s</span>`;
    log.prepend(li);
    while (log.children.length > 12) log.lastChild.remove();
  }

  // ---------------------------------------------------------------------------
  // DRAW
  // ---------------------------------------------------------------------------
  let drag = null;

  function drawOverlay(ball) {
    const L = state.line;
    // OUT region tint.
    const poly = outPolygon();
    if (poly.length > 2) {
      ctx.fillStyle = "rgba(239,68,68,0.13)";
      ctx.beginPath(); poly.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.fill();
    }
    // Boundary line, extended across the frame.
    const dx = L.bx - L.ax, dy = L.by - L.ay, len = Math.hypot(dx, dy) || 1, k = 2000 / len;
    ctx.save();
    ctx.strokeStyle = "#ff2d2d"; ctx.lineWidth = 3; ctx.shadowColor = "rgba(0,0,0,.6)"; ctx.shadowBlur = 4;
    ctx.beginPath(); ctx.moveTo(L.ax - dx * k, L.ay - dy * k); ctx.lineTo(L.bx + dx * k, L.by + dy * k); ctx.stroke();
    ctx.restore();
    // Little "OUT" label on the OUT side near the top.
    labelSides();

    // Trail.
    if (state.showTrail && state.trail.length > 1) {
      for (let i = 1; i < state.trail.length; i++) {
        const a = state.trail[i - 1], b = state.trail[i], f = i / state.trail.length;
        ctx.strokeStyle = b.out ? `rgba(255,80,80,${f})` : `rgba(80,230,140,${f})`;
        ctx.lineWidth = 2 + f * 4; ctx.lineCap = "round";
        ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke();
      }
    }
    // Detected ball.
    if (ball) {
      ctx.strokeStyle = "#ffa500"; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.arc(ball.x, ball.y, Math.max(ball.r, 6), 0, Math.PI * 2); ctx.stroke();
      ctx.fillStyle = "#ffa500";
      ctx.beginPath(); ctx.arc(ball.x, ball.y, 3, 0, Math.PI * 2); ctx.fill();
      ctx.font = "600 13px Inter, sans-serif"; ctx.textAlign = "center";
      ctx.fillText("BALL", ball.x, ball.y - Math.max(ball.r, 6) - 7);
    }
    // Line being dragged.
    if (drag && drag.moved) {
      ctx.setLineDash([8, 6]); ctx.strokeStyle = "#fff"; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.moveTo(drag.x0, drag.y0); ctx.lineTo(drag.x1, drag.y1); ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  function labelSides() {
    const L = state.line;
    const mx = (L.ax + L.bx) / 2, my = (L.ay + L.by) / 2;
    let nx = -(L.by - L.ay), ny = L.bx - L.ax;                 // normal
    const n = Math.hypot(nx, ny) || 1; nx /= n; ny /= n;
    if (sideOf(mx + nx, my + ny) !== state.outSign) { nx = -nx; ny = -ny; }
    ctx.font = "700 14px Inter, sans-serif"; ctx.textAlign = "center"; ctx.textBaseline = "middle";
    const put = (txt, color, s) => {
      const x = Math.min(FRAME_W - 30, Math.max(30, mx + nx * 46 * s));
      const y = Math.min(FRAME_H - 40, Math.max(90, my + ny * 46 * s));
      ctx.fillStyle = "rgba(0,0,0,.45)"; ctx.fillRect(x - 24, y - 11, 48, 22);
      ctx.fillStyle = color; ctx.fillText(txt, x, y);
    };
    put("OUT", "#ff8080", 1);
    put("IN", "#7ef0a8", -1);
    ctx.textBaseline = "alphabetic";
  }

  function drawMask() {
    const d = maskImg.data;
    for (let i = 0, p = 0; i < N; i++, p += 4) {
      const v = mask[i] ? 255 : 0;
      d[p] = d[p + 1] = d[p + 2] = v; d[p + 3] = 255;
    }
    pctx.putImageData(maskImg, 0, 0);
    ctx.imageSmoothingEnabled = false;
    ctx.drawImage(proc, 0, 0, FRAME_W, FRAME_H);
    ctx.imageSmoothingEnabled = true;
  }

  // ---------------------------------------------------------------------------
  // MAIN LOOP
  // ---------------------------------------------------------------------------
  let last = performance.now(), fpsAvg = 0, noBallSince = performance.now();

  function frame(now) {
    const dt = Math.min(0.05, (now - last) / 1000);
    last = now;
    if (state.mode === "demo") stepDemo(dt);

    const have = drawSource(pctx, PW, PH);
    let ball = null;
    if (have) {
      ball = detect();
      if (state.showMask) drawMask();
      else drawSource(ctx, FRAME_W, FRAME_H);
      updateCall(ball);
      drawOverlay(ball);
    }

    // Readouts.
    if (dt > 0) fpsAvg = fpsAvg ? fpsAvg * 0.92 + (1 / dt) * 0.08 : 1 / dt;
    $("r-fps").textContent = have ? fpsAvg.toFixed(0) : "—";
    $("r-ball").textContent = ball ? `${ball.x.toFixed(0)}, ${ball.y.toFixed(0)}` : "not found";
    $("r-area").textContent = ball ? `${ball.area.toFixed(0)} px` : "—";

    if (ball) noBallSince = now;
    if (!state.calibrating) {
      hintEl.textContent = have && now - noBallSince > 1200
        ? "No ball found. Calibrate, then click the ball."
        : "Click to set the line · drag to draw it at an angle";
    }
    requestAnimationFrame(frame);
  }

  // ---------------------------------------------------------------------------
  // CALIBRATION (calibrate.py)
  // ---------------------------------------------------------------------------
  function setCalibrating(on) {
    state.calibrating = on;
    stage.classList.toggle("calibrating", on);
    $("calibrate").classList.toggle("on", on);
    $("calibrate").textContent = on ? "Click the ball…" : "Calibrate";
    if (on) hintEl.textContent = "Click the center of the ball to sample its color";
    if (state.mode === "demo") setPaused(on ? true : state.prevPaused);
  }

  function sampleAt(fx, fy) {
    const img = state.lastImage;
    if (!img) return;
    const px = Math.round(fx / SCALE), py = Math.round(fy / SCALE);
    const H = [], S = [], V = [], R = [], G = [], B = [];
    for (let y = py - 2; y <= py + 2; y++) {
      for (let x = px - 2; x <= px + 2; x++) {
        if (x < 0 || y < 0 || x >= PW || y >= PH) continue;
        const p = (y * PW + x) * 4, d = img.data;
        const [h, s, v] = hsv(d[p], d[p + 1], d[p + 2]);
        H.push(h); S.push(s); V.push(v); R.push(d[p]); G.push(d[p + 1]); B.push(d[p + 2]);
      }
    }
    const med = (a) => a.sort((m, n) => m - n)[a.length >> 1];
    const h = med(H), s = med(S), v = med(V);
    // "set LOWER_COLOR slightly below and UPPER_COLOR slightly above those numbers"
    state.lower = [(h - 10 + 180) % 180, Math.max(s - 70, 50), Math.max(v - 80, 50)];
    state.upper = [(h + 10) % 180, 255, 255];
    $("swatch").style.background = `rgb(${med(R)},${med(G)},${med(B)})`;
    syncSliders();
    state.trail = [];
  }

  // ---------------------------------------------------------------------------
  // HSV SLIDERS
  // ---------------------------------------------------------------------------
  const SLIDERS = [
    ["H min", "lower", 0, 179], ["H max", "upper", 0, 179],
    ["S min", "lower", 1, 255], ["S max", "upper", 1, 255],
    ["V min", "lower", 2, 255], ["V max", "upper", 2, 255],
  ];
  const sliderEls = [];
  (function buildSliders() {
    const box = $("sliders");
    SLIDERS.forEach(([label, key, idx, max], i) => {
      const id = `sl${i}`;
      const lab = document.createElement("label"); lab.htmlFor = id; lab.textContent = label;
      const inp = document.createElement("input");
      Object.assign(inp, { type: "range", id, min: 0, max, step: 1 });
      const out = document.createElement("output"); out.htmlFor = id;
      inp.addEventListener("input", () => { state[key][idx] = +inp.value; syncSliders(); });
      box.append(lab, inp, out);
      sliderEls.push({ inp, out, key, idx });
    });
  })();

  function syncSliders() {
    sliderEls.forEach(({ inp, out, key, idx }) => { inp.value = state[key][idx]; out.textContent = state[key][idx]; });
    const [lh, ls, lv] = state.lower, [uh, us, uv] = state.upper;
    $("hsv-text").textContent = `H ${lh}–${uh} · S ${ls}–${us} · V ${lv}–${uv}`;
  }

  // ---------------------------------------------------------------------------
  // INPUT
  // ---------------------------------------------------------------------------
  function toFrame(e) {
    const r = view.getBoundingClientRect();
    return [((e.clientX - r.left) / r.width) * FRAME_W, ((e.clientY - r.top) / r.height) * FRAME_H];
  }

  view.addEventListener("pointerdown", (e) => {
    const [x, y] = toFrame(e);
    view.setPointerCapture(e.pointerId);
    drag = { x0: x, y0: y, x1: x, y1: y, moved: false };
  });
  view.addEventListener("pointermove", (e) => {
    if (!drag) return;
    [drag.x1, drag.y1] = toFrame(e);
    if (Math.hypot(drag.x1 - drag.x0, drag.y1 - drag.y0) > 10 && !state.calibrating) drag.moved = true;
  });
  view.addEventListener("pointerup", () => {
    if (!drag) return;
    const d = drag; drag = null;
    if (state.calibrating) { sampleAt(d.x1, d.y1); setCalibrating(false); return; }
    if (d.moved) setLine(d.x0, d.y0, d.x1, d.y1);
    else setLine(d.x1, 0, d.x1, FRAME_H);          // click = vertical line, like /setline?x=
  });
  view.addEventListener("pointercancel", () => { drag = null; });

  $("flip").onclick = () => { state.outSign *= -1; state.trail = []; state.call = null; renderCall(false); };
  $("reset-line").onclick = defaultLine;
  $("calibrate").onclick = () => setCalibrating(!state.calibrating);
  $("show-mask").onchange = (e) => { state.showMask = e.target.checked; };
  $("show-trail").onchange = (e) => { state.showTrail = e.target.checked; state.trail = []; };
  $("ball-color").onclick = () => {
    demo.color = (demo.color + 1) % BALL_COLORS.length;
    $("ball-color").textContent = `Swap demo ball color (now ${BALL_COLORS[demo.color].name})`;
  };
  function setPaused(p) {
    state.paused = !!p;
    $("pause").textContent = state.paused ? "Play" : "Pause";
  }
  $("pause").onclick = () => { setPaused(!state.paused); state.prevPaused = state.paused; };

  // ---------------------------------------------------------------------------
  // MODES
  // ---------------------------------------------------------------------------
  function stopCamera() {
    if (state.stream) state.stream.getTracks().forEach((t) => t.stop());
    state.stream = null;
    video.srcObject = null;
  }

  async function startCamera() {
    const err = $("cam-err");
    err.textContent = "";
    const btn = $("start-cam");
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      err.textContent = "This browser can't open a camera here. It needs HTTPS and a modern browser.";
      return;
    }
    btn.disabled = true; btn.classList.add("loading"); btn.textContent = "Starting camera";
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: "environment", width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false,
      });
      if (state.mode !== "camera") { stream.getTracks().forEach((t) => t.stop()); return; }
      state.stream = stream;
      video.srcObject = stream;
      await video.play();
      $("cam-gate").hidden = true;
    } catch (e) {
      err.textContent = e && e.name === "NotAllowedError"
        ? "Camera permission was denied. Allow it in your browser's site settings, or try the Demo."
        : `Couldn't start the camera (${(e && e.name) || "error"}). Try the Demo instead.`;
    } finally {
      btn.disabled = false; btn.classList.remove("loading"); btn.textContent = "Start camera";
    }
  }

  function setMode(mode) {
    if (state.calibrating) setCalibrating(false);
    state.mode = mode;
    document.body.classList.toggle("camera", mode === "camera");
    document.querySelectorAll(".mode").forEach((b) => {
      const on = b.dataset.mode === mode;
      b.classList.toggle("active", on);
      b.setAttribute("aria-selected", on);
    });
    if (mode === "camera") {
      $("cam-gate").hidden = !!state.stream;
      ctx.fillStyle = "#000"; ctx.fillRect(0, 0, FRAME_W, FRAME_H);
    } else {
      stopCamera();
      $("cam-gate").hidden = true;
      demo.color = 0;
      $("ball-color").textContent = "Swap demo ball color";
    }
    state.lower = DEFAULT_LOWER.slice(); state.upper = DEFAULT_UPPER.slice();
    $("swatch").style.background = "";
    syncSliders();
    defaultLine();
    renderCall(false);
  }

  document.querySelectorAll(".mode").forEach((b) => (b.onclick = () => setMode(b.dataset.mode)));
  $("start-cam").onclick = startCamera;

  // Debug / test hook (read-only snapshot).
  window.clearcall = { get state() { return { call: state.call, line: state.line, outSign: state.outSign, calls: state.calls, lower: state.lower, upper: state.upper }; } };

  state.prevPaused = false;
  setMode("demo");
  requestAnimationFrame(frame);
})();
