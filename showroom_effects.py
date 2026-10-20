#!/usr/bin/env python3
"""Spatial AI showroom demo for OAK-D Lite.

Multi-person skeleton tracking (YOLO11-pose) + stereo depth from the OAK camera, rendered
as big, readable-from-across-the-room effects. Runs fullscreen and cycles modes by itself.

Keys:
  1-6  pick mode          A  auto-cycle on/off      M  mirror on/off
  F    fullscreen toggle  C  clear light painting   S  screenshot
  Q / Esc  quit

Falls back to a normal webcam (no depth) if no OAK camera is connected.
"""
import argparse
import collections
import math
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort

try:
    import depthai as dai
except ImportError:
    dai = None

HERE = Path(__file__).resolve().parent
W, H = 1280, 720
WIN = "Spatial AI Showroom"

(NOSE, LEYE, REYE, LEAR, REAR, LSHO, RSHO, LELB, RELB,
 LWRI, RWRI, LHIP, RHIP, LKNE, RKNE, LANK, RANK) = range(17)
BONES = [(LSHO, LELB), (LELB, LWRI), (RSHO, RELB), (RELB, RWRI), (LSHO, RSHO), (LSHO, LHIP),
         (RSHO, RHIP), (LHIP, RHIP), (LHIP, LKNE), (LKNE, LANK), (RHIP, RKNE), (RKNE, RANK)]
NEON = [(255, 255, 0), (255, 0, 255), (80, 255, 80), (0, 170, 255),
        (0, 255, 255), (255, 120, 60), (160, 90, 255), (90, 90, 255)]
MODES = [
    ("NEON SKELETON", "AI tracks 17 body joints of every person in real time"),
    ("LIGHT PAINTING", "Wave your hands - paint the air with light"),
    ("HOLOGRAM", "The 3D depth camera cuts you out of the room"),
    ("DEPTH VISION", "What the stereo camera sees: distance as colour"),
    ("ENERGY", "Bring your hands together... then pull them apart!"),
    ("CROWD RADAR", "Everyone's position in the room, seen from above"),
]
FONT = cv2.FONT_HERSHEY_DUPLEX


# ----------------------------------------------------------------------------- camera sources
class OakSource:
    def __init__(self):
        p = dai.Pipeline()
        cam = p.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
        cam.setPreviewSize(W, H)
        cam.setInterleaved(False)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
        cam.setFps(30)

        stereo = p.create(dai.node.StereoDepth)
        stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.DEFAULT)
        stereo.setLeftRightCheck(True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(640, 360)
        for sock, inp in ((dai.CameraBoardSocket.CAM_B, stereo.left), (dai.CameraBoardSocket.CAM_C, stereo.right)):
            m = p.create(dai.node.MonoCamera)
            m.setBoardSocket(sock)
            m.setResolution(dai.MonoCameraProperties.SensorResolution.THE_480_P)
            m.setFps(30)
            m.out.link(inp)

        for name, src in (("rgb", cam.preview), ("depth", stereo.depth)):
            x = p.create(dai.node.XLinkOut)
            x.setStreamName(name)
            x.input.setBlocking(False)
            x.input.setQueueSize(1)
            src.link(x.input)

        self.dev = dai.Device(p)
        self.q_rgb = self.dev.getOutputQueue("rgb", 1, False)
        self.q_depth = self.dev.getOutputQueue("depth", 1, False)
        intr = self.dev.readCalibration().getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, W, H)
        self.fx = intr[0][0]
        self.depth = None
        speed = self.dev.getUsbSpeed().name
        self.info = f"{self.dev.getDeviceName()}  USB {speed}"
        if speed not in ("SUPER", "SUPER_PLUS"):
            print("WARNING: camera is on USB 2 - use a USB 3 port/cable for full frame rate")

    def read(self):
        frame = self.q_rgb.get().getCvFrame()
        d = self.q_depth.tryGet()
        if d is not None:
            self.depth = cv2.resize(d.getFrame(), (W, H), interpolation=cv2.INTER_NEAREST)
        return frame, self.depth

    def close(self):
        self.dev.close()


class WebcamSource:
    def __init__(self, index):
        self.cap = cv2.VideoCapture(index, cv2.CAP_DSHOW if os.name == "nt" else cv2.CAP_ANY)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, W)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, H)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open webcam {index}")
        self.fx = W / (2 * math.tan(math.radians(35)))
        self.info = f"webcam {index} (no depth)"

    def read(self):
        ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("webcam read failed")
        if f.shape[:2] != (H, W):
            f = cv2.resize(f, (W, H))
        return f, None

    def close(self):
        self.cap.release()


# ----------------------------------------------------------------------------- pose model
class PoseModel:
    IN_W, IN_H = 640, 384

    def __init__(self, path, conf=0.4):
        avail = ort.get_available_providers()
        prov = [p for p in ("CUDAExecutionProvider", "DmlExecutionProvider") if p in avail]
        self.sess = ort.InferenceSession(str(path), providers=prov + ["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.conf = conf
        self.scale = self.IN_W / W
        self.pad = (self.IN_H - round(H * self.scale)) // 2
        self.name = Path(path).stem

    def bench(self, n=6):
        x = np.random.rand(1, 3, self.IN_H, self.IN_W).astype(np.float32)
        self.sess.run(None, {self.inp: x})
        t = time.perf_counter()
        for _ in range(n):
            self.sess.run(None, {self.inp: x})
        return (time.perf_counter() - t) / n * 1000

    def infer(self, frame):
        img = cv2.resize(frame, (self.IN_W, round(H * self.scale)))
        bot = self.IN_H - img.shape[0] - self.pad
        img = cv2.copyMakeBorder(img, self.pad, bot, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255
        out = self.sess.run(None, {self.inp: x})[0][0].T  # (anchors, 56)
        out = out[out[:, 4] > self.conf]
        if not len(out):
            return []
        boxes = np.column_stack([out[:, 0] - out[:, 2] / 2, out[:, 1] - out[:, 3] / 2, out[:, 2], out[:, 3]])
        keep = cv2.dnn.NMSBoxes(boxes.tolist(), out[:, 4].tolist(), self.conf, 0.5)
        dets = []
        for i in np.array(keep).flatten():
            bx, by, bw, bh = boxes[i]
            kp = out[i, 5:].reshape(17, 3).copy()
            kp[:, 0] /= self.scale
            kp[:, 1] = (kp[:, 1] - self.pad) / self.scale
            box = np.array([bx, by - self.pad, bx + bw, by + bh - self.pad]) / self.scale
            dets.append({"box": box, "kp": kp, "score": float(out[i, 4])})
        return dets


# ----------------------------------------------------------------------------- people tracking
class Person:
    def __init__(self, pid, det, now):
        self.id = pid
        self.color = NEON[(pid - 1) % len(NEON)]
        self.kp = det["kp"].copy()
        self.box = det["box"]
        self.last = now
        self.z, self.x = None, 0.0
        self.trail = collections.deque(maxlen=50)
        self.wprev = {}
        self.wvel = {LWRI: np.zeros(2), RWRI: np.zeros(2)}
        self.paint_prev = {}
        self.next_fw = 0.0
        self.head = None  # (center, radius)

    def update(self, det, now):
        new, dt = det["kp"], max(now - self.last, 1e-3)
        vis_old = self.kp[:, 2] > 0.35
        both = (new[:, 2] > 0.35) & vis_old
        self.kp[both, :2] = 0.6 * new[both, :2] + 0.4 * self.kp[both, :2]
        self.kp[~vis_old, :2] = new[~vis_old, :2]
        self.kp[:, 2] = new[:, 2]
        for w in (LWRI, RWRI):
            p = self.pt(w)
            if p is not None and w in self.wprev:
                self.wvel[w] = 0.5 * self.wvel[w] + 0.5 * (np.array(p) - self.wprev[w]) / dt
            self.wprev[w] = np.array(p) if p is not None else self.wprev.get(w)
        self.box = det["box"]
        self.last = now

    def pt(self, i, thr=0.35):
        return (int(self.kp[i, 0]), int(self.kp[i, 1])) if self.kp[i, 2] > thr else None

    def shoulder_w(self):
        a, b = self.pt(LSHO), self.pt(RSHO)
        if a and b:
            return max(math.dist(a, b), 10.0)
        return max((self.box[2] - self.box[0]) * 0.45, 10.0)

    def hands_up(self):
        lw, rw = self.pt(LWRI, 0.5), self.pt(RWRI, 0.5)
        if not (lw and rw):
            return False
        nose = self.pt(NOSE)
        top = nose[1] if nose else min(self.kp[LSHO, 1], self.kp[RSHO, 1]) - self.shoulder_w() * 0.6
        return lw[1] < top and rw[1] < top


class Tracker:
    def __init__(self):
        self.people = {}

    def update(self, dets, now):
        pairs = []
        for pid, p in self.people.items():
            pc = ((p.box[0] + p.box[2]) / 2, (p.box[1] + p.box[3]) / 2)
            for j, d in enumerate(dets):
                b = d["box"]
                size = max(b[2] - b[0], b[3] - b[1], 1)
                dist = math.dist(pc, ((b[0] + b[2]) / 2, (b[1] + b[3]) / 2)) / size
                if dist < 0.6:
                    pairs.append((dist, pid, j))
        pairs.sort()
        used_p, used_d = set(), set()
        for _, pid, j in pairs:
            if pid in used_p or j in used_d:
                continue
            self.people[pid].update(dets[j], now)
            used_p.add(pid)
            used_d.add(j)
        for j, d in enumerate(dets):
            if j not in used_d:
                pid = next(i for i in range(1, 1000) if i not in self.people)
                self.people[pid] = Person(pid, d, now)
        for pid in [k for k, p in self.people.items() if now - p.last > 0.5]:
            del self.people[pid]
        return sorted((p for p in self.people.values() if p.last == now), key=lambda p: p.id)


def locate(p, depth, fx):
    """3D position of a person from the depth map (or shoulder width when there is no depth)."""
    pts = [p.kp[i, :2] for i in (LSHO, RSHO, LHIP, RHIP) if p.kp[i, 2] > 0.4]
    if len(pts) >= 2:
        a = np.array(pts)
        x1, y1 = a.min(0)
        x2, y2 = a.max(0)
    else:
        bx1, by1, bx2, by2 = p.box
        x1, x2 = bx1 + (bx2 - bx1) * 0.35, bx2 - (bx2 - bx1) * 0.35
        y1, y2 = by1 + (by2 - by1) * 0.25, by1 + (by2 - by1) * 0.5
    x1, x2 = int(np.clip(x1, 0, W - 8)), int(np.clip(x2, x1 + 8, W))
    y1, y2 = int(np.clip(y1, 0, H - 8)), int(np.clip(y2, y1 + 8, H))
    z = None
    if depth is not None:
        patch = depth[y1:y2:2, x1:x2:2]
        v = patch[(patch > 300) & (patch < 12000)]
        if v.size > 15:
            z = float(np.median(v)) / 1000
    else:
        z = fx * 0.38 / p.shoulder_w()
    if z:
        p.z = z if p.z is None else 0.7 * p.z + 0.3 * z
        p.x = ((x1 + x2) / 2 - W / 2) * p.z / fx
        p.trail.append((p.x, p.z))


# ----------------------------------------------------------------------------- effects
class Particles:
    MAX = 5000

    def __init__(self):
        self.pos = np.zeros((0, 2), np.float32)
        self.vel = np.zeros((0, 2), np.float32)
        self.life = np.zeros(0, np.float32)
        self.maxlife = np.zeros(0, np.float32)
        self.col = np.zeros((0, 3), np.float32)
        self.grav = np.zeros(0, np.float32)

    def emit(self, xy, n, speed, color, base_vel=(0, 0), life=(0.5, 1.2), grav=300.0, white=0.0):
        if n <= 0:
            return
        ang = np.random.uniform(0, 2 * np.pi, n)
        sp = np.random.uniform(0.15, 1.0, n) * speed
        vel = np.column_stack([np.cos(ang) * sp + base_vel[0], np.sin(ang) * sp + base_vel[1]])
        lf = np.random.uniform(*life, n)
        col = np.tile(np.array(color, np.float32), (n, 1))
        if white:
            m = np.random.rand(n) < white
            col[m] = 255
        self.pos = np.vstack([self.pos, np.tile(np.array(xy, np.float32), (n, 1))])[-self.MAX:]
        self.vel = np.vstack([self.vel, vel.astype(np.float32)])[-self.MAX:]
        self.life = np.concatenate([self.life, lf.astype(np.float32)])[-self.MAX:]
        self.maxlife = np.concatenate([self.maxlife, lf.astype(np.float32)])[-self.MAX:]
        self.col = np.vstack([self.col, col])[-self.MAX:]
        self.grav = np.concatenate([self.grav, np.full(n, grav, np.float32)])[-self.MAX:]

    def firework(self, xy, color):
        self.emit(xy, 220, 520, color, life=(0.9, 1.9), grav=200, white=0.25)

    def step(self, dt):
        if not len(self.life):
            return
        self.pos += self.vel * dt
        self.vel[:, 1] += self.grav * dt
        self.vel *= max(0.0, 1 - 1.1 * dt)
        self.life -= dt
        alive = self.life > 0
        for a in ("pos", "vel", "life", "maxlife", "col", "grav"):
            setattr(self, a, getattr(self, a)[alive])

    def draw(self, layer):
        if not len(self.life):
            return
        xy = self.pos.astype(np.int32)
        ok = (xy[:, 0] >= 0) & (xy[:, 0] < W) & (xy[:, 1] >= 0) & (xy[:, 1] < H)
        if not ok.any():
            return
        pl = np.zeros_like(layer)
        f = (self.life[ok] / self.maxlife[ok])[:, None]
        pl[xy[ok, 1], xy[ok, 0]] = (self.col[ok] * np.sqrt(f)).astype(np.uint8)
        pl = cv2.dilate(pl, np.ones((3, 3), np.uint8))
        cv2.add(layer, pl, dst=layer)


def bolt(a, b, rough=0.22, levels=5):
    pts = [np.array(a, float), np.array(b, float)]
    for _ in range(levels):
        out = [pts[0]]
        for p, q in zip(pts, pts[1:]):
            d = q - p
            length = math.hypot(*d) + 1e-6
            m = (p + q) / 2 + np.array([-d[1], d[0]]) / length * random.gauss(0, rough * length)
            out += [m, q]
        pts = out
    return np.int32(pts).reshape(-1, 1, 2)


def bloom(layer, strength=1.8):
    s = cv2.GaussianBlur(cv2.resize(layer, (W // 4, H // 4), interpolation=cv2.INTER_AREA), (0, 0), 4)
    s2 = cv2.GaussianBlur(cv2.resize(s, (W // 16, H // 16), interpolation=cv2.INTER_AREA), (0, 0), 3)
    return cv2.addWeighted(cv2.resize(s, (W, H)), strength, cv2.resize(s2, (W, H)), strength, 0)


def hue_bgr(h, s=255, v=255):
    c = cv2.cvtColor(np.uint8([[[int(h) % 180, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return int(c[0]), int(c[1]), int(c[2])


def lighter(c, k=0.55):
    return tuple(int(v * (1 - k) + 255 * k) for v in c)


def colorize_depth(depth):
    if depth is None:
        return None
    d = np.clip(depth.astype(np.float32), 400, 6000)
    d = (255 - (d - 400) * (255 / 5600)).astype(np.uint8)
    d[depth == 0] = 0
    out = cv2.applyColorMap(d, cv2.COLORMAP_TURBO)
    out[depth == 0] = 0
    return out


def text(img, s, org, scale=0.7, color=(255, 255, 255), thick=1, center=False):
    if center:
        (tw, _), _ = cv2.getTextSize(s, FONT, scale, thick)
        org = (int(org[0] - tw / 2), int(org[1]))
    o = max(2, int(scale * 2.5))
    cv2.putText(img, s, (org[0] + o, org[1] + o), FONT, scale, (0, 0, 0), thick, cv2.LINE_AA)
    cv2.putText(img, s, org, FONT, scale, color, thick, cv2.LINE_AA)


def draw_person(G, C, p, color=None, thick=None):
    col = color or p.color
    core = lighter(col)
    sw = p.shoulder_w()
    t = thick or int(np.clip(sw / 16, 2, 9))
    for a, b in BONES:
        pa, pb = p.pt(a), p.pt(b)
        if pa and pb:
            cv2.line(G, pa, pb, col, t * 2, cv2.LINE_AA)
            cv2.line(C, pa, pb, core, max(1, t // 2), cv2.LINE_AA)
    nose = p.pt(NOSE)
    p.head = None
    if nose:
        le, re = p.pt(LEAR), p.pt(REAR)
        r = int(max(math.dist(le, re) * 0.65 if le and re else sw * 0.32, 8))
        ctr = (nose[0], nose[1] - r // 4)
        p.head = (ctr, r)
        cv2.circle(G, ctr, r, col, t * 2, cv2.LINE_AA)
        cv2.circle(C, ctr, r, core, max(1, t // 2), cv2.LINE_AA)
        ls, rs = p.pt(LSHO), p.pt(RSHO)
        if ls and rs:
            neck = ((ls[0] + rs[0]) // 2, (ls[1] + rs[1]) // 2)
            v = np.array(neck, float) - ctr
            n = np.linalg.norm(v)
            if n > r:
                start = tuple(int(c) for c in np.array(ctr) + v / n * r)
                cv2.line(G, start, neck, col, t * 2, cv2.LINE_AA)
                cv2.line(C, start, neck, core, max(1, t // 2), cv2.LINE_AA)
    for i in range(5, 17):
        q = p.pt(i)
        if q:
            cv2.circle(G, q, t + 3, col, -1, cv2.LINE_AA)
            cv2.circle(C, q, max(2, t // 2 + 1), (255, 255, 255), -1, cv2.LINE_AA)


def person_top(p):
    if p.head:
        return p.head[0][0], p.head[0][1] - p.head[1] - 14
    return int((p.box[0] + p.box[2]) / 2), int(p.box[1]) - 10


def tag(img, p, s, color=None, scale=0.8, placed=None):
    col = color or p.color
    x, y = person_top(p)
    (tw, th), _ = cv2.getTextSize(s, FONT, scale, 2)
    x = int(np.clip(x - tw / 2, 6, W - tw - 6))
    y = int(np.clip(y, th + 100, H - 10))
    if placed is not None:  # push up until it no longer overlaps an earlier tag
        moved = True
        while moved:
            moved = False
            for (ax1, ay1, ax2, ay2) in placed:
                if x - 10 < ax2 and x + tw + 10 > ax1 and y - th - 10 < ay2 and y + 8 > ay1:
                    y = ay1 - 12
                    moved = True
        placed.append((x - 10, y - th - 10, x + tw + 10, y + 8))
    cv2.rectangle(img, (x - 10, y - th - 10), (x + tw + 10, y + 8), col, -1, cv2.LINE_AA)
    cv2.putText(img, s, (x, y), FONT, scale, (15, 15, 15), 2, cv2.LINE_AA)


def dist_label(p):
    return f"#{p.id}  {p.z:.1f} m" if p.z else f"#{p.id}"


def radar(people, size, t, max_z=6.0):
    r = np.zeros((size, size, 3), np.uint8)
    r[:] = (20, 26, 16)
    cx, by = size // 2, size - 16
    sc = (size - 40) / max_z
    for m in range(1, int(max_z) + 1):
        cv2.ellipse(r, (cx, by), (int(m * sc), int(m * sc)), 0, 180, 360, (50, 110, 50), 1, cv2.LINE_AA)
        text(r, f"{m} m", (cx + 5, by - int(m * sc) + 16), 0.45, (90, 180, 90))
    hf = math.radians(34.5)
    for sgn in (-1, 1):
        cv2.line(r, (cx, by), (int(cx + sgn * math.sin(hf) * max_z * sc), int(by - math.cos(hf) * max_z * sc)),
                 (70, 160, 70), 1, cv2.LINE_AA)
    sw = (t * 1.2) % (2 * hf) - hf
    cv2.line(r, (cx, by), (int(cx + math.sin(sw) * size), int(by - math.cos(sw) * size)), (120, 255, 120), 2,
             cv2.LINE_AA)
    cv2.circle(r, (cx, by), 7, (255, 255, 255), -1, cv2.LINE_AA)
    text(r, "CAMERA", (cx - 34, by + 2 - 14), 0.4, (200, 200, 200))
    for p in people:
        if not p.z:
            continue
        pts = [(int(cx + x * sc), int(by - z * sc)) for x, z in p.trail]
        for i in range(1, len(pts)):
            cv2.line(r, pts[i - 1], pts[i], p.color, 2, cv2.LINE_AA)
        q = (int(cx + p.x * sc), int(by - p.z * sc))
        pulse = 12 + int(4 * math.sin(t * 6 + p.id))
        cv2.circle(r, q, pulse + 6, p.color, 1, cv2.LINE_AA)
        cv2.circle(r, q, 11, p.color, -1, cv2.LINE_AA)
        text(r, f"#{p.id} {p.z:.1f}m", (q[0] + 16, q[1] + 6), 0.55, p.color)
    text(r, "TOP-DOWN VIEW", (14, 30), 0.75, (120, 255, 120))
    cv2.rectangle(r, (0, 0), (size - 1, size - 1), (90, 200, 90), 2)
    return r


# ----------------------------------------------------------------------------- app
class Showroom:
    def __init__(self, args, source, model):
        self.a = args
        self.src = source
        self.model = model
        self.tracker = Tracker()
        self.particles = Particles()
        self.paint = np.zeros((H // 2, W // 2, 3), np.float32)
        self.mode = 0
        self.mode_t = time.time()
        self.auto = True
        self.mirror = not args.no_mirror
        self.last_seen = 0.0
        self.fps = 0.0
        self.infer_ms = 0.0

    def set_mode(self, m):
        self.mode = m % len(MODES)
        self.mode_t = time.time()
        if self.mode == 1:
            self.paint[:] = 0

    # -- per-mode backgrounds and layers
    def render(self, frame, depth, people, now, dt):
        G = np.zeros((H, W, 3), np.uint8)  # glowing layer
        C = np.zeros((H, W, 3), np.uint8)  # sharp additive layer
        m = self.mode
        dcol = colorize_depth(depth)

        if m == 0:  # neon skeleton
            out = cv2.convertScaleAbs(frame, alpha=0.4)
            for p in people:
                draw_person(G, C, p)

        elif m == 1:  # light painting
            out = cv2.convertScaleAbs(frame, alpha=0.25)
            self.paint *= 0.985 ** (dt * 60)
            for p in people:
                draw_person(G, C, p, thick=2)
                for k, w in enumerate((LWRI, RWRI)):
                    q = p.pt(w, 0.45)
                    key = w
                    if q is None:
                        p.paint_prev.pop(key, None)
                        continue
                    q2 = (q[0] // 2, q[1] // 2)
                    col = hue_bgr(now * 40 + p.id * 50 + k * 25)
                    prev = p.paint_prev.get(key)
                    if prev is not None and math.dist(prev, q2) < 120:
                        cv2.line(self.paint, prev, q2, col, 7, cv2.LINE_AA)
                        cv2.line(self.paint, prev, q2, (255, 255, 255), 2, cv2.LINE_AA)
                    p.paint_prev[key] = q2
            up = cv2.resize(np.clip(self.paint, 0, 255).astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)
            cv2.add(G, up, dst=G)

        elif m == 2:  # hologram
            out = np.zeros((H, W, 3), np.uint8)
            out[:] = (30, 14, 6)
            if dcol is not None:
                dots = np.zeros_like(out)
                dots[::12, ::12] = dcol[::12, ::12]
                out = cv2.add(out, cv2.dilate(dots, np.ones((2, 2), np.uint8)) // 2)
            mask = np.zeros((H, W), np.uint8)
            for p in people:
                x1, y1, x2, y2 = p.box
                bw, bh = x2 - x1, y2 - y1
                x1, x2 = int(max(x1 - bw * 0.05, 0)), int(min(x2 + bw * 0.05, W))
                y1, y2 = int(max(y1 - bh * 0.05, 0)), int(min(y2 + bh * 0.05, H))
                if x2 <= x1 or y2 <= y1:
                    continue
                if depth is not None and p.z:
                    sub = depth[y1:y2, x1:x2].astype(np.float32) / 1000
                    mask[y1:y2, x1:x2] |= ((np.abs(sub - p.z) < 0.55) & (sub > 0)).astype(np.uint8) * 255
                else:
                    cv2.ellipse(mask, ((x1 + x2) // 2, (y1 + y2) // 2), (int(bw // 2), int(bh // 2)), 0, 0, 360, 255, -1)
            mask = cv2.GaussianBlur(cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8)), (0, 0), 3)
            mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)[1]
            g = cv2.equalizeHist(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            scan = np.where((np.arange(H) + int(now * 90)) % 6 < 3, 255, 100).astype(np.uint8)
            g = cv2.multiply(g, np.broadcast_to(scan[:, None], (H, W)), scale=(0.85 + 0.15 * random.random()) / 255)
            holo = cv2.merge([g, cv2.convertScaleAbs(g, alpha=0.92), cv2.convertScaleAbs(g, alpha=0.3)])
            if random.random() < 0.18:
                y0, hgt = random.randrange(0, H - 40), random.randrange(6, 40)
                sh = random.randrange(-40, 40)
                holo[y0:y0 + hgt] = np.roll(holo[y0:y0 + hgt], sh, axis=1)
                mask[y0:y0 + hgt] = np.roll(mask[y0:y0 + hgt], sh, axis=1)
            cv2.copyTo(holo, mask, out)
            edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8))
            G[edge > 0] = (255, 220, 90)
            for p in people:
                draw_person(G, C, p, color=(255, 230, 120), thick=2)
                feet = [q for q in (p.pt(LANK), p.pt(RANK)) if q]
                fy = max(q[1] for q in feet) + 10 if feet else int(p.box[3])
                if fy < H:
                    cx, rw = int((p.box[0] + p.box[2]) / 2), int((p.box[2] - p.box[0]) * 0.6)
                    for k in range(3):
                        rr = (rw + k * 14 + int(now * 40) % 14, (rw + k * 14) // 5)
                        cv2.ellipse(G, (cx, fy), rr, 0, 0, 360, (255, 200, 60), 2, cv2.LINE_AA)

        elif m == 3:  # depth vision
            out = dcol.copy() if dcol is not None else cv2.cvtColor(cv2.Canny(frame, 60, 140), cv2.COLOR_GRAY2BGR)
            for p in people:
                draw_person(G, C, p, color=(255, 255, 255))
            # legend
            lx, ly, lh = W - 50, 160, 360
            bar = cv2.applyColorMap(np.linspace(255, 0, lh).astype(np.uint8)[:, None].repeat(22, 1), cv2.COLORMAP_TURBO)
            out[ly:ly + lh, lx:lx + 22] = bar
            text(out, "NEAR", (lx - 70, ly + 14), 0.6)
            text(out, "FAR", (lx - 56, ly + lh), 0.6)
            if depth is not None:
                c = depth[H // 2 - 6:H // 2 + 6, W // 2 - 6:W // 2 + 6]
                c = c[c > 0]
                cv2.drawMarker(C, (W // 2, H // 2), (255, 255, 255), cv2.MARKER_CROSS, 34, 2)
                if c.size:
                    text(out, f"{np.median(c) / 1000:.2f} m", (W // 2 + 22, H // 2 - 14), 0.9)

        elif m == 4:  # energy
            out = cv2.convertScaleAbs(frame, alpha=0.3)
            for p in people:
                draw_person(G, C, p, thick=3)
                sw = p.shoulder_w()
                for w in (LWRI, RWRI):
                    q = p.pt(w, 0.45)
                    if q is None:
                        continue
                    v = p.wvel[w]
                    sp = float(np.linalg.norm(v))
                    self.particles.emit(q, min(int(sp / 90) + 1, 14), 60 + sp * 0.15, p.color,
                                        base_vel=v * 0.25, life=(0.4, 1.0), grav=250, white=0.2)
                lw, rw = p.pt(LWRI, 0.45), p.pt(RWRI, 0.45)
                hips = [q[1] for q in (p.pt(LHIP), p.pt(RHIP)) if q]
                raised = hips and lw and rw and max(lw[1], rw[1]) < min(hips) - sw * 0.3
                if raised:  # hands held up in front of the body -> energy ball between them
                    d = math.dist(lw, rw)
                    if d < sw * 3.2:
                        mid = ((lw[0] + rw[0]) // 2, (lw[1] + rw[1]) // 2)
                        k = max(0.0, 1 - d / (sw * 3.2))
                        r = int(6 + k * sw * 0.35 * (1 + 0.15 * math.sin(now * 14)))
                        cv2.circle(G, mid, r, (200, 110, 30), -1, cv2.LINE_AA)
                        cv2.circle(C, mid, r, (255, 230, 160), 2, cv2.LINE_AA)
                        cv2.circle(C, mid, max(2, r // 3), (255, 255, 255), -1, cv2.LINE_AA)
                        for src in (lw, rw):
                            for _ in range(2):
                                pts = bolt(src, mid, 0.2, 4)
                                cv2.polylines(G, [pts], False, (255, 190, 110), 3, cv2.LINE_AA)
                                cv2.polylines(C, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
            # arcs between different people's hands
            for i, a in enumerate(people):
                for b in people[i + 1:]:
                    best = None
                    for wa in (LWRI, RWRI):
                        for wb in (LWRI, RWRI):
                            pa, pb = a.pt(wa, 0.45), b.pt(wb, 0.45)
                            if pa and pb and (best is None or math.dist(pa, pb) < best[0]):
                                best = (math.dist(pa, pb), pa, pb)
                    if best and best[0] < W * 0.4:
                        for _ in range(3):
                            pts = bolt(best[1], best[2], 0.25, 6)
                            cv2.polylines(G, [pts], False, (255, 120, 255), 3, cv2.LINE_AA)
                            cv2.polylines(C, [pts], False, (255, 255, 255), 1, cv2.LINE_AA)
                        for q in best[1:]:
                            self.particles.emit(q, 4, 220, (255, 160, 255), life=(0.2, 0.5), grav=0, white=0.5)

        else:  # crowd radar
            out = cv2.convertScaleAbs(frame, alpha=0.55)
            for p in people:
                draw_person(G, C, p)

        # fireworks when both hands are raised (all modes)
        any_up = False
        for p in people:
            if p.hands_up():
                any_up = True
                if now >= p.next_fw:
                    x, y = person_top(p)
                    sw = p.shoulder_w()
                    pos = (x + random.uniform(-1.5, 1.5) * sw, max(110, y - random.uniform(30, 160)))
                    self.particles.firework(pos, random.choice(NEON))
                    p.next_fw = now + 0.28
        self.particles.step(dt)
        self.particles.draw(G)

        out = cv2.add(out, G)
        out = cv2.add(out, bloom(G))
        out = cv2.add(out, C)

        # name tags
        placed = []
        for p in sorted(people, key=lambda q: -person_top(q)[1]):
            if p.hands_up():
                tag(out, p, "HANDS UP!", (0, 255, 255), 1.0, placed)
            elif m == 3:
                tag(out, p, f"{p.z:.2f} m" if p.z else "?", (255, 255, 255), 1.1, placed)
            else:
                tag(out, p, dist_label(p), placed=placed)
        if m == 5:
            rs = 430
            rd = radar(people, rs, now)
            x0, y0 = W - rs - 20, 110
            out[y0:y0 + rs, x0:x0 + rs] = cv2.addWeighted(out[y0:y0 + rs, x0:x0 + rs], 0.15, rd, 0.85, 0)
        return out, any_up

    def overlay(self, out, people, now, any_up):
        # top bar
        out[:96] = (out[:96] * 0.35).astype(np.uint8)
        text(out, self.a.title, (28, 54), 1.45, (255, 255, 255), 2)
        text(out, self.a.subtitle, (30, 84), 0.62, (190, 190, 190))
        n = len(people)
        text(out, "PEOPLE", (W - 250, 40), 0.7, (200, 200, 200))
        text(out, str(n), (W - 120, 80), 2.6, (0, 255, 255) if n else (120, 120, 120), 4)
        # bottom bar
        out[H - 86:] = (out[H - 86:] * 0.35).astype(np.uint8)
        name, hint = MODES[self.mode]
        if int(now / 6) % 2:
            hint = "Raise both hands for fireworks!"
        text(out, name, (28, H - 44), 1.15, NEON[self.mode % len(NEON)], 2)
        text(out, hint, (30, H - 14), 0.7, (230, 230, 230))
        for i in range(len(MODES)):
            c = (W - 40 - (len(MODES) - 1 - i) * 26, H - 52)
            cv2.circle(out, c, 8 if i == self.mode else 5, (255, 255, 255) if i == self.mode else (110, 110, 110),
                       -1, cv2.LINE_AA)
        info = f"{self.fps:.0f} FPS  AI {self.infer_ms:.0f} ms  {'AUTO' if self.auto else 'MANUAL'}"
        text(out, info, (W - 330, H - 18), 0.5, (150, 150, 150))
        # mode banner
        age = now - self.mode_t
        if age < 2.5:
            a = min(1.0, 4 * (2.5 - age) / 2.5)
            ban = out.copy()
            cv2.rectangle(ban, (0, H // 2 - 70), (W, H // 2 + 60), (0, 0, 0), -1)
            text(ban, name, (W // 2, H // 2 + 12), 2.4, NEON[self.mode % len(NEON)], 4, center=True)
            text(ban, MODES[self.mode][1], (W // 2, H // 2 + 46), 0.8, (230, 230, 230), 1, center=True)
            cv2.addWeighted(ban, a * 0.85, out, 1 - a * 0.85, 0, dst=out)
        if any_up and int(now * 3) % 2:
            text(out, "FIREWORKS!", (W // 2, 150), 1.6, (0, 255, 255), 3, center=True)
        # attract screen
        if now - self.last_seen > 2.0 and age >= 2.5:
            pulse = 0.6 + 0.4 * math.sin(now * 3)
            col = tuple(int(c * pulse) for c in (255, 255, 0))
            text(out, "STEP IN FRONT OF THE CAMERA", (W // 2, H // 2), 1.6, col, 3, center=True)
            text(out, "the AI will find you", (W // 2, H // 2 + 44), 0.8, (210, 210, 210), 1, center=True)
        return out

    def run(self, selftest=0.0):
        if not selftest:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
            if not self.a.windowed:
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.resizeWindow(WIN, W, H)
        pool = ThreadPoolExecutor(1)
        pending = None
        last, fcount, ft = time.time(), 0, time.time()
        start = time.time()
        shots = set()

        def timed_infer(f):
            t = time.perf_counter()
            r = self.model.infer(f)
            return r, (time.perf_counter() - t) * 1000

        while True:
            frame, depth = self.src.read()
            if self.mirror:
                frame = cv2.flip(frame, 1)
                depth = cv2.flip(depth, 1) if depth is not None else None
            job = (pool.submit(timed_infer, frame), frame, depth)
            if pending is None:
                pending = job
                continue
            fut, frame, depth = pending
            pending = job
            dets, ms = fut.result()
            self.infer_ms = 0.8 * self.infer_ms + 0.2 * ms

            now = time.time()
            dt = min(now - last, 0.1)
            last = now
            people = self.tracker.update(dets, now)
            for p in people:
                locate(p, depth, self.src.fx)
            if people:
                self.last_seen = now
            if self.auto and now - self.mode_t > self.a.cycle:
                self.set_mode(self.mode + 1)

            out, any_up = self.render(frame, depth, people, now, dt)
            out = self.overlay(out, people, now, any_up)

            fcount += 1
            if now - ft >= 1:
                self.fps, fcount, ft = fcount / (now - ft), 0, now

            if selftest:
                if now - self.mode_t > 2.7 and self.mode not in shots:
                    shots.add(self.mode)
                    d = HERE / "selftest"
                    d.mkdir(exist_ok=True)
                    cv2.imwrite(str(d / f"mode{self.mode + 1}.jpg"), out)
                    print(f"mode {self.mode + 1} {MODES[self.mode][0]:15s} fps {self.fps:5.1f}  "
                          f"ai {self.infer_ms:4.1f} ms  people {len(people)}")
                    self.set_mode(self.mode + 1)
                if len(shots) == len(MODES) or now - start > selftest:
                    break
                continue

            cv2.imshow(WIN, out)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if ord("1") <= k <= ord(str(len(MODES))):
                self.set_mode(k - ord("1"))
            elif k in (ord("a"), ord("A")):
                self.auto = not self.auto
                self.mode_t = now - 2.5
            elif k in (ord("m"), ord("M")):
                self.mirror = not self.mirror
            elif k in (ord("c"), ord("C")):
                self.paint[:] = 0
            elif k in (ord("f"), ord("F")):
                full = cv2.getWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL if full else cv2.WINDOW_FULLSCREEN)
            elif k in (ord("s"), ord("S")):
                path = HERE / f"shot_{time.strftime('%Y%m%d_%H%M%S')}.jpg"
                cv2.imwrite(str(path), out)
                print("saved", path)
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
        pool.shutdown(wait=False)
        cv2.destroyAllWindows()


class ImageSource:
    """Static test picture, for checking the effects without people in front of the camera."""

    def __init__(self, path):
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"cannot read {path}")
        s = min(W / img.shape[1], H / img.shape[0])
        img = cv2.resize(img, None, fx=s, fy=s)
        self.frame = np.zeros((H, W, 3), np.uint8)
        y, x = (H - img.shape[0]) // 2, (W - img.shape[1]) // 2
        self.frame[y:y + img.shape[0], x:x + img.shape[1]] = img
        self.fx = W / (2 * math.tan(math.radians(35)))
        self.info = f"image {path}"

    def read(self):
        time.sleep(0.03)
        return self.frame.copy(), None

    def close(self):
        pass


def open_source(args):
    if args.image:
        return ImageSource(args.image)
    if not args.webcam_only and dai is not None:
        try:
            if dai.Device.getAllAvailableDevices():
                return OakSource()
            print("No OAK camera found.")
        except Exception as e:
            print("OAK camera failed:", e)
    print(f"Falling back to webcam {args.webcam}")
    return WebcamSource(args.webcam)


def load_model(args):
    mdir = HERE / "models"
    if args.model != "auto":
        return PoseModel(mdir / f"yolo11{args.model}-pose.onnx", args.conf)
    m = PoseModel(mdir / "yolo11s-pose.onnx", args.conf)
    ms = m.bench()
    print(f"yolo11s-pose: {ms:.0f} ms")
    if ms > 45:
        print("computer is slow for the large model, using yolo11n-pose")
        m = PoseModel(mdir / "yolo11n-pose.onnx", args.conf)
    return m


def main():
    ap = argparse.ArgumentParser(description="OAK-D Lite spatial AI showroom demo")
    ap.add_argument("--title", default="SPATIAL AI VISION")
    ap.add_argument("--subtitle", default="OAK-D Lite stereo depth camera  +  real-time multi-person pose AI")
    ap.add_argument("--cycle", type=float, default=25, help="seconds per mode in auto mode")
    ap.add_argument("--model", choices=["auto", "n", "s"], default="auto", help="n = fast, s = more accurate")
    ap.add_argument("--conf", type=float, default=0.4, help="person detection confidence")
    ap.add_argument("--webcam", type=int, default=0, help="webcam index used when there is no OAK")
    ap.add_argument("--webcam-only", action="store_true")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--windowed", action="store_true")
    ap.add_argument("--image", help="use a still picture instead of a camera (testing)")
    ap.add_argument("--selftest", type=float, default=0, help="run N seconds headless, save one image per mode")
    args = ap.parse_args()

    model = load_model(args)
    print("pose model:", model.name, "providers:", model.sess.get_providers()[0])
    src = open_source(args)
    print("camera:", src.info)
    try:
        Showroom(args, src, model).run(args.selftest)
    finally:
        src.close()


if __name__ == "__main__":
    sys.exit(main())
