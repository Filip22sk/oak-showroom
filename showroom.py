#!/usr/bin/env python3
"""Multi-person spatial pose analysis with an OAK-D Lite.

Host (CPU):   YOLO11-pose (ONNX) -> 17 COCO keypoints per person, simple IoU/centroid tracker
OAK device:   stereo depth aligned to RGB, age/gender + emotion CNNs on face crops (Myriad X)
Derived:      3D position from depth, body height, walking speed, posture, head yaw

Default view: fullscreen camera with per-person data. --dashboard shows the analysis layout with plots.

Keys: q/Esc quit   f fullscreen   s save screenshot   m mirror   space pause
"""
import argparse
import collections
import math
import os
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
MODELS = HERE / "models"
W, H = 1280, 720            # processing resolution
VW, VH = 960, 540           # displayed video size
PW = 480                    # side panel width
BH = 270                    # bottom plots height
CW, CH = VW + PW, VH + BH   # window size
WIN = "OAK-D Lite - spatial pose analysis"
FONT = cv2.FONT_HERSHEY_SIMPLEX

(NOSE, LEYE, REYE, LEAR, REAR, LSHO, RSHO, LELB, RELB,
 LWRI, RWRI, LHIP, RHIP, LKNE, RKNE, LANK, RANK) = range(17)
LEFT, RIGHT, MID = (255, 128, 0), (0, 128, 255), (0, 190, 0)
BONES = [(LSHO, LELB, LEFT), (LELB, LWRI, LEFT), (LHIP, LKNE, LEFT), (LKNE, LANK, LEFT),
         (RSHO, RELB, RIGHT), (RELB, RWRI, RIGHT), (RHIP, RKNE, RIGHT), (RKNE, RANK, RIGHT),
         (LSHO, RSHO, MID), (LSHO, LHIP, MID), (RSHO, RHIP, MID), (LHIP, RHIP, MID),
         (NOSE, LEYE, MID), (NOSE, REYE, MID), (LEYE, LEAR, MID), (REYE, REAR, MID)]
# matplotlib tab10, BGR
TAB10 = [(180, 119, 31), (14, 127, 255), (44, 160, 44), (40, 39, 214), (189, 103, 148),
         (75, 86, 140), (194, 119, 227), (127, 127, 127), (34, 189, 188), (207, 190, 23)]
EMOTIONS = ["neutral", "happy", "sad", "surprise", "anger"]
FACE_NETS = {"ag": ("age-gender-recognition-retail-0013", 62), "em": ("emotions-recognition-retail-0003", 64)}
INK, GREY, LIGHT = (25, 25, 25), (110, 110, 110), (225, 225, 225)


# ----------------------------------------------------------------------------- face analysis on the OAK
def add_face_nodes(p):
    for key, (blob, size) in FACE_NETS.items():
        xi = p.create(dai.node.XLinkIn)
        xi.setStreamName(key + "_in")
        xi.setMaxDataSize(size * size * 3)
        nn = p.create(dai.node.NeuralNetwork)
        nn.setBlobPath(str(MODELS / f"{blob}.blob"))
        nn.input.setBlocking(False)
        nn.input.setQueueSize(8)
        xo = p.create(dai.node.XLinkOut)
        xo.setStreamName(key)
        xi.out.link(nn.input)
        nn.out.link(xo.input)


class FaceAnalyzer:
    def __init__(self, dev):
        self.qin = {k: dev.getInputQueue(k + "_in", maxSize=8, blocking=False) for k in FACE_NETS}
        self.qout = {k: dev.getOutputQueue(k, maxSize=16, blocking=False) for k in FACE_NETS}
        self.sent = {}
        self.ms = 0.0

    def send(self, pid, crop):
        for k, (_, s) in FACE_NETS.items():
            img = cv2.resize(crop, (s, s))
            f = dai.ImgFrame()
            f.setType(dai.ImgFrame.Type.BGR888p)
            f.setWidth(s)
            f.setHeight(s)
            f.setSequenceNum(pid)
            f.setData(img.transpose(2, 0, 1).flatten())
            self.qin[k].send(f)
        self.sent[pid] = time.perf_counter()

    def poll(self):
        out = []
        for m in self.qout["ag"].tryGetAll():
            pid = m.getSequenceNum()
            out.append((pid, "ag", (m.getLayerFp16("age_conv3")[0] * 100, m.getLayerFp16("prob")[1])))
            if pid in self.sent:
                self.ms = 0.8 * self.ms + 0.2 * (time.perf_counter() - self.sent[pid]) * 1000
        for m in self.qout["em"].tryGetAll():
            out.append((m.getSequenceNum(), "em", np.array(m.getLayerFp16("prob_emotion"))))
        return out


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
        add_face_nodes(p)

        self.dev = dai.Device(p)
        self.q_rgb = self.dev.getOutputQueue("rgb", 1, False)
        self.q_depth = self.dev.getOutputQueue("depth", 1, False)
        self.faces = FaceAnalyzer(self.dev)
        k = np.array(self.dev.readCalibration().getCameraIntrinsics(dai.CameraBoardSocket.CAM_A, W, H))
        self.fx, self.fy, self.cx, self.cy = k[0, 0], k[1, 1], k[0, 2], k[1, 2]
        self.depth = None
        speed = self.dev.getUsbSpeed().name
        self.info = f"{self.dev.getDeviceName()}, USB {speed}"
        self.depth_info = "stereo 640x480 mono pair, LR-check, aligned to RGB"
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
        self.fx = self.fy = W / (2 * math.tan(math.radians(35)))
        self.cx, self.cy = W / 2, H / 2
        self.faces = None
        self.info = f"webcam {index}"
        self.depth_info = "none (distance from shoulder width)"

    def read(self):
        ok, f = self.cap.read()
        if not ok:
            raise RuntimeError("webcam read failed")
        return (cv2.resize(f, (W, H)) if f.shape[:2] != (H, W) else f), None

    def close(self):
        self.cap.release()


class ImageSource:
    """Still picture instead of a camera, for testing. Uses an OAK for face nets if one is plugged in."""

    def __init__(self, path):
        img = cv2.imread(str(path))
        if img is None:
            raise RuntimeError(f"cannot read {path}")
        s = min(W / img.shape[1], H / img.shape[0])
        img = cv2.resize(img, None, fx=s, fy=s)
        self.frame = np.zeros((H, W, 3), np.uint8)
        y, x = (H - img.shape[0]) // 2, (W - img.shape[1]) // 2
        self.frame[y:y + img.shape[0], x:x + img.shape[1]] = img
        self.fx = self.fy = W / (2 * math.tan(math.radians(35)))
        self.cx, self.cy = W / 2, H / 2
        self.info = f"image {Path(path).name}"
        self.depth_info = "none (distance from shoulder width)"
        self.dev, self.faces = None, None
        if dai is not None and dai.Device.getAllAvailableDevices():
            p = dai.Pipeline()
            add_face_nodes(p)
            self.dev = dai.Device(p)
            self.faces = FaceAnalyzer(self.dev)

    def read(self):
        time.sleep(0.03)
        return self.frame.copy(), None

    def close(self):
        if self.dev:
            self.dev.close()


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
        self.provider = self.sess.get_providers()[0].replace("ExecutionProvider", "")

    def bench(self, n=6):
        x = np.random.rand(1, 3, self.IN_H, self.IN_W).astype(np.float32)
        self.sess.run(None, {self.inp: x})
        best = 1e9
        for _ in range(n):
            t = time.perf_counter()
            self.sess.run(None, {self.inp: x})
            best = min(best, time.perf_counter() - t)
        return best * 1000

    def infer(self, frame):
        img = cv2.resize(frame, (self.IN_W, round(H * self.scale)))
        bot = self.IN_H - img.shape[0] - self.pad
        img = cv2.copyMakeBorder(img, self.pad, bot, 0, 0, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        x = np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))[None].astype(np.float32) / 255
        out = self.sess.run(None, {self.inp: x})[0][0].T  # (anchors, 4 box + 1 conf + 17*3 kpts)
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


# ----------------------------------------------------------------------------- tracking + measurements
class Person:
    def __init__(self, pid, det, now):
        self.id = pid
        self.color = TAB10[(pid - 1) % len(TAB10)]
        self.kp = det["kp"].copy()
        self.box = det["box"]
        self.score = det["score"]
        self.first = self.last = now
        self.z = None
        self.x = self.y = 0.0
        self.trail = collections.deque(maxlen=90)  # (t, x, z)
        self.heights = collections.deque(maxlen=40)
        self.ages = collections.deque(maxlen=30)
        self.males = collections.deque(maxlen=30)
        self.emos = collections.deque(maxlen=12)
        self.face = None
        self.face_note = ""
        self.last_face_req = 0.0

    def update(self, det, now):
        new = det["kp"]
        vis_old = self.kp[:, 2] > 0.35
        both = (new[:, 2] > 0.35) & vis_old
        self.kp[both, :2] = 0.6 * new[both, :2] + 0.4 * self.kp[both, :2]
        self.kp[~vis_old, :2] = new[~vis_old, :2]
        self.kp[:, 2] = new[:, 2]
        self.box, self.score, self.last = det["box"], det["score"], now

    def pt(self, i, thr=0.4):
        return (float(self.kp[i, 0]), float(self.kp[i, 1])) if self.kp[i, 2] > thr else None

    def shoulder_w(self):
        a, b = self.pt(LSHO), self.pt(RSHO)
        return max(math.dist(a, b), 10.0) if a and b else max((self.box[2] - self.box[0]) * 0.45, 10.0)

    # --- derived quantities
    def speed(self):
        if len(self.trail) < 5:
            return None
        t1, x1, z1 = self.trail[-1]
        old = [s for s in self.trail if t1 - s[0] >= 0.8]
        t0, x0, z0 = old[-1] if old else self.trail[0]
        return math.hypot(x1 - x0, z1 - z0) / max(t1 - t0, 1e-3)

    def height(self):
        return float(np.median(self.heights)) if len(self.heights) >= 3 else None

    def age(self):
        if len(self.ages) < 3:
            return None
        a = np.array(self.ages)
        return float(np.median(a)), float(a.std()), len(a)

    def gender(self):
        if len(self.males) < 3:
            return None
        m = float(np.mean(self.males))
        return ("male", m) if m >= 0.5 else ("female", 1 - m)

    def emotion(self):
        if not self.emos:
            return None
        e = np.mean(self.emos, axis=0)
        i = int(np.argmax(e))
        return EMOTIONS[i], float(e[i])

    def posture(self):
        tags = []
        nose, le, re = self.pt(NOSE), self.pt(LEYE), self.pt(REYE)
        lw, rw = self.pt(LWRI), self.pt(RWRI)
        ls, rs = self.pt(LSHO), self.pt(RSHO)
        if lw and rw and nose and lw[1] < nose[1] and rw[1] < nose[1]:
            tags.append("both hands up")
        elif (lw and ls and lw[1] < ls[1]) or (rw and rs and rw[1] < rs[1]):
            tags.append("hand raised")
        hips = [p for p in (self.pt(LHIP), self.pt(RHIP)) if p]
        knees = [p for p in (self.pt(LKNE), self.pt(RKNE)) if p]
        if hips and ls and rs:
            dx = np.mean([p[0] for p in hips]) - (ls[0] + rs[0]) / 2
            dy = np.mean([p[1] for p in hips]) - (ls[1] + rs[1]) / 2
            if abs(dx) > abs(dy):
                tags.insert(0, "lying")
            elif knees:
                thigh = np.mean([p[1] for p in knees]) - np.mean([p[1] for p in hips])
                tags.insert(0, "sitting" if thigh < 0.45 * abs(dy) else "standing")
        if nose and le and re:
            tags.append("facing camera")
        elif not nose and self.pt(LEAR) and self.pt(REAR):
            tags.append("facing away")
        elif nose:
            tags.append("side view")
        return ", ".join(tags) if tags else "-"

    def head_yaw(self):
        nose, le, re = self.pt(NOSE, 0.5), self.pt(LEYE, 0.5), self.pt(REYE, 0.5)
        if not (nose and le and re):
            return None
        d = abs(le[0] - re[0])
        if d < 4:
            return None
        r = (nose[0] - (le[0] + re[0]) / 2) / d
        return math.degrees(math.atan(2 * r))

    def face_crop_box(self):
        nose, le, re = self.pt(NOSE, 0.5), self.pt(LEYE, 0.5), self.pt(REYE, 0.5)
        if not (nose and le and re):
            return None, "face not visible"
        la, ra = self.pt(LEAR, 0.5), self.pt(REAR, 0.5)
        size = 1.25 * math.dist(la, ra) if la and ra else 2.6 * math.dist(le, re)
        if size < 28:
            return None, "face too small (too far)"
        cx = (nose[0] + le[0] + re[0]) / 3
        cy = (nose[1] + le[1] + re[1]) / 3 + 0.08 * size
        x1, y1 = int(cx - size / 2), int(cy - size / 2)
        x2, y2 = int(cx + size / 2), int(cy + size / 2)
        if x1 < 0 or y1 < 0 or x2 > W or y2 > H:
            return None, "face at image edge"
        return (x1, y1, x2, y2), ""


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
        for pid in [k for k, p in self.people.items() if now - p.last > 1.0]:
            del self.people[pid]
        return sorted((p for p in self.people.values() if p.last == now), key=lambda p: p.id)


def measure(p, depth, src, now, mirrored):
    """Distance / 3D position from the depth map (torso median), plus body height."""
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
    p.roi = (x1, y1, x2, y2)
    z = None
    if depth is not None:
        patch = depth[y1:y2:2, x1:x2:2]
        v = patch[(patch > 300) & (patch < 12000)]
        if v.size > 15:
            z = float(np.median(v)) / 1000
    else:
        z = src.fx * 0.38 / p.shoulder_w()  # average shoulder width ~0.38 m
    if not z:
        return
    p.z = z if p.z is None else 0.7 * p.z + 0.3 * z
    cx = W - src.cx if mirrored else src.cx
    p.x = ((x1 + x2) / 2 - cx) * p.z / src.fx
    p.y = -((y1 + y2) / 2 - src.cy) * p.z / src.fy
    p.trail.append((now, p.x, p.z))
    # height: nose-to-ankle in pixels -> metres, + nose-to-crown and ankle-to-floor offsets
    nose = p.pt(NOSE, 0.5)
    ank = [q for q in (p.pt(LANK, 0.5), p.pt(RANK, 0.5)) if q]
    if nose and ank:
        h = (max(q[1] for q in ank) - nose[1]) * p.z / src.fy + 0.12 + 0.08
        if 0.9 < h < 2.3:
            p.heights.append(h)


# ----------------------------------------------------------------------------- drawing helpers
def put(img, s, org, scale=0.45, color=INK, thick=1):
    cv2.putText(img, s, (int(org[0]), int(org[1])), FONT, scale, color, thick, cv2.LINE_AA)


def put_right(img, s, org, scale=0.45, color=INK, thick=1):
    (tw, _), _ = cv2.getTextSize(s, FONT, scale, thick)
    put(img, s, (org[0] - tw, org[1]), scale, color, thick)


class Axes:
    """Minimal matplotlib-looking axes on a white image."""

    def __init__(self, w, h, title, xlim, ylim, xticks, yticks, xlabel="", ylabel="", ml=46, mb=34, mt=26, mr=14):
        self.img = np.full((h, w, 3), 255, np.uint8)
        self.l, self.r, self.t, self.b = ml, w - mr, mt, h - mb
        self.xlim, self.ylim = xlim, ylim
        im = self.img
        for x in xticks:
            px = self.px(x, ylim[0])[0]
            cv2.line(im, (px, self.t), (px, self.b), (232, 232, 232), 1)
            cv2.line(im, (px, self.b), (px, self.b + 4), INK, 1)
            (tw, _), _ = cv2.getTextSize(f"{x:g}", FONT, 0.36, 1)
            put(im, f"{x:g}", (px - tw / 2, self.b + 16), 0.36)
        for y in yticks:
            py = self.px(xlim[0], y)[1]
            cv2.line(im, (self.l, py), (self.r, py), (232, 232, 232), 1)
            cv2.line(im, (self.l - 4, py), (self.l, py), INK, 1)
            put_right(im, f"{y:g}", (self.l - 7, py + 4), 0.36)
        cv2.rectangle(im, (self.l, self.t), (self.r, self.b), INK, 1)
        (tw, _), _ = cv2.getTextSize(title, FONT, 0.45, 1)
        put(im, title, ((self.l + self.r - tw) / 2, self.t - 9), 0.45)
        if xlabel:
            (tw, _), _ = cv2.getTextSize(xlabel, FONT, 0.38, 1)
            put(im, xlabel, ((self.l + self.r - tw) / 2, h - 5), 0.38, GREY)
        if ylabel:
            put(im, ylabel, (4, self.t - 9), 0.38, GREY)

    def px(self, x, y):
        (x0, x1), (y0, y1) = self.xlim, self.ylim
        return (int(round(self.l + (x - x0) / (x1 - x0) * (self.r - self.l))),
                int(round(self.b - (y - y0) / (y1 - y0) * (self.b - self.t))))

    def inside(self, q):
        return self.l <= q[0] <= self.r and self.t <= q[1] <= self.b


def draw_skeleton(img, p):
    for a, b, col in BONES:
        pa, pb = p.pt(a), p.pt(b)
        if pa and pb:
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])), col, 3, cv2.LINE_AA)
    for i in range(17):
        q = p.pt(i)
        if q:
            cv2.circle(img, (int(q[0]), int(q[1])), 4, (0, 0, 230), -1, cv2.LINE_AA)
            cv2.circle(img, (int(q[0]), int(q[1])), 4, (255, 255, 255), 1, cv2.LINE_AA)


FW, FH = 1920, 1080  # fullscreen render size


def draw_fullscreen(frame, people, info):
    """Camera view only, upscaled to 1080p, with each person's data drawn next to them."""
    s = FW / W
    img = cv2.resize(frame, (FW, FH), interpolation=cv2.INTER_LINEAR)
    for p in people:
        for a, b, col in BONES:
            pa, pb = p.pt(a), p.pt(b)
            if pa and pb:
                cv2.line(img, (int(pa[0] * s), int(pa[1] * s)), (int(pb[0] * s), int(pb[1] * s)), col, 4, cv2.LINE_AA)
        for i in range(17):
            q = p.pt(i)
            if q:
                c = (int(q[0] * s), int(q[1] * s))
                cv2.circle(img, c, 6, (0, 0, 230), -1, cv2.LINE_AA)
                cv2.circle(img, c, 6, (255, 255, 255), 1, cv2.LINE_AA)
        x1, y1, x2, y2 = [int(v * s) for v in p.box]
        cv2.rectangle(img, (x1, y1), (x2, y2), p.color, 2)
        if p.face:
            fx1, fy1, fx2, fy2 = [int(v * s) for v in p.face]
            cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (255, 255, 255), 1)

    placed = []
    for p in sorted(people, key=lambda q: q.box[0]):
        ag, gd, em, ht = p.age(), p.gender(), p.emotion(), p.height()
        title = f"ID {p.id}   {p.z:.2f} m" if p.z else f"ID {p.id}"
        lines = [
            f"age {ag[0]:.0f}" + (f"   {gd[0]}" if gd else "") if ag else f"age: {p.face_note or 'waiting...'}",
            f"emotion {em[0]}" if em else None,
            "   ".join(x for x in (f"height {ht:.2f} m" if ht else "", p.posture()) if x and x != "-"),
        ]
        lines = [l for l in lines if l]
        ts, ls, lh = 0.9, 0.7, 30
        tw = max([cv2.getTextSize(title, FONT, ts, 2)[0][0]] + [cv2.getTextSize(l, FONT, ls, 1)[0][0] for l in lines])
        bw, bh = tw + 24, 40 + lh * len(lines) + 8
        bx = int(np.clip(p.box[0] * s, 0, FW - bw))
        by = int(p.box[1] * s) - bh - 4
        if by < 0:
            by = int(p.box[1] * s) + 4
        for (ax1, ay1, ax2, ay2) in placed:  # keep boxes from covering each other
            if bx < ax2 and bx + bw > ax1 and by < ay2 and by + bh > ay1:
                by = ay2 + 6
        by = int(np.clip(by, 0, FH - bh))
        placed.append((bx, by, bx + bw, by + bh))
        roi = img[by:by + bh, bx:bx + bw]
        roi[:] = (roi * 0.35).astype(np.uint8)
        cv2.rectangle(img, (bx, by), (bx + bw, by + 38), p.color, -1)
        put(img, title, (bx + 12, by + 28), ts, (255, 255, 255), 2)
        for i, l in enumerate(lines):
            put(img, l, (bx + 12, by + 38 + lh * (i + 1) - 6), ls, (255, 255, 255))

    (tw, th), _ = cv2.getTextSize(info, FONT, 0.7, 1)
    cv2.rectangle(img, (0, 0), (tw + 24, th + 22), (0, 0, 0), -1)
    put(img, info, (12, th + 11), 0.7, (255, 255, 255))
    return img


def draw_video(frame, people, info):
    img = frame
    for p in people:
        x1, y1, x2, y2 = [int(v) for v in p.box]
        cv2.rectangle(img, (x1, y1), (x2, y2), p.color, 2)
        if p.face:
            fx1, fy1, fx2, fy2 = p.face
            cv2.rectangle(img, (fx1, fy1), (fx2, fy2), (255, 255, 255), 1)
        draw_skeleton(img, p)
    placed = []
    for p in sorted(people, key=lambda q: q.box[0]):
        x1, y1 = int(p.box[0]), int(p.box[1])
        parts = [f"ID {p.id}"]
        if p.z:
            parts.append(f"{p.z:.2f} m")
        a = p.age()
        if a:
            parts.append(f"age ~{a[0]:.0f}")
        s = "  ".join(parts)
        (tw, th), _ = cv2.getTextSize(s, FONT, 0.65, 2)
        ty = max(y1, th + 10)
        for (ax1, ay1, ax2, ay2) in placed:  # stack labels that would overlap
            if x1 < ax2 and x1 + tw + 10 > ax1 and ty - th - 10 < ay2 and ty > ay1:
                ty = ay2 + th + 10
        placed.append((x1, ty - th - 10, x1 + tw + 10, ty))
        cv2.rectangle(img, (x1, ty - th - 10), (x1 + tw + 10, ty), p.color, -1)
        put(img, s, (x1 + 5, ty - 6), 0.65, (255, 255, 255), 2)
    s = info
    (tw, th), _ = cv2.getTextSize(s, FONT, 0.6, 1)
    cv2.rectangle(img, (0, 0), (tw + 16, th + 16), (0, 0, 0), -1)
    put(img, s, (8, th + 8), 0.6, (255, 255, 255))
    return cv2.resize(img, (VW, VH), interpolation=cv2.INTER_AREA)


def depth_view(depth, people, w, h):
    img = np.full((h, w, 3), 255, np.uint8)
    put(img, "Stereo depth (aligned to RGB)", (12, 18), 0.45)
    iw, ih = w - 92, int((w - 92) * 9 / 16)
    ox, oy = 12, 28
    if depth is None:
        cv2.rectangle(img, (ox, oy), (ox + iw, oy + ih), (200, 200, 200), -1)
        put(img, "no depth stream", (ox + iw // 2 - 70, oy + ih // 2), 0.5, GREY)
    else:
        d = cv2.resize(depth, (iw, ih), interpolation=cv2.INTER_NEAREST).astype(np.float32)
        n = np.clip((d - 400) / (6000 - 400), 0, 1)
        col = cv2.applyColorMap((n * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        col[d == 0] = (40, 40, 40)
        sx, sy = iw / W, ih / H
        for p in people:
            x1, y1, x2, y2 = p.box
            cv2.rectangle(col, (int(x1 * sx), int(y1 * sy)), (int(x2 * sx), int(y2 * sy)), (255, 255, 255), 1)
            if hasattr(p, "roi"):
                r = p.roi
                cv2.rectangle(col, (int(r[0] * sx), int(r[1] * sy)), (int(r[2] * sx), int(r[3] * sy)), (0, 0, 255), 1)
            put(col, str(p.id), (x1 * sx + 3, y1 * sy + 14), 0.45, (255, 255, 255))
        img[oy:oy + ih, ox:ox + iw] = col
    cv2.rectangle(img, (ox, oy), (ox + iw, oy + ih), INK, 1)
    # colour bar
    bx = ox + iw + 12
    bar = cv2.applyColorMap(np.linspace(255, 0, ih).astype(np.uint8)[:, None].repeat(14, 1), cv2.COLORMAP_VIRIDIS)
    img[oy:oy + ih, bx:bx + 14] = bar
    cv2.rectangle(img, (bx, oy), (bx + 14, oy + ih), INK, 1)
    for m in (0.4, 2, 4, 6):
        y = int(oy + ih - (m - 0.4) / 5.6 * ih)
        cv2.line(img, (bx + 14, y), (bx + 18, y), INK, 1)
        put(img, f"{m:g} m", (bx + 21, y + 4), 0.36)
    put(img, "red box = depth sampling ROI (torso median)", (ox, oy + ih + 20), 0.36, GREY)
    return img


def top_view(people, w, h):
    ax = Axes(w, h, "Top view (camera at origin)", (-4, 4), (0, 8), range(-4, 5), range(0, 9, 2),
              xlabel="X [m]", ylabel="Z [m]")
    im = ax.img
    o = ax.px(0, 0)
    hf = math.radians(34.5)  # half horizontal FOV of the RGB camera
    for s in (-1, 1):
        e = ax.px(s * math.tan(hf) * 8, 8)
        for k in range(0, 40, 2):  # dashed FOV lines
            a = (int(o[0] + (e[0] - o[0]) * k / 40), int(o[1] + (e[1] - o[1]) * k / 40))
            b = (int(o[0] + (e[0] - o[0]) * (k + 1) / 40), int(o[1] + (e[1] - o[1]) * (k + 1) / 40))
            if ax.inside(a) and ax.inside(b):
                cv2.line(im, a, b, (150, 150, 150), 1, cv2.LINE_AA)
    cv2.drawMarker(im, o, INK, cv2.MARKER_TRIANGLE_UP, 12, 2)
    for p in people:
        if not p.z:
            continue
        pts = [ax.px(x, z) for _, x, z in p.trail]
        pts = [q for q in pts if ax.inside(q)]
        if len(pts) > 1:
            cv2.polylines(im, [np.int32(pts)], False, p.color, 1, cv2.LINE_AA)
        q = ax.px(p.x, p.z)
        if ax.inside(q):
            cv2.circle(im, q, 6, p.color, -1, cv2.LINE_AA)
            cv2.circle(im, q, 6, INK, 1, cv2.LINE_AA)
            put(im, f"{p.id}", (q[0] + 8, q[1] + 4), 0.42)
    return im


def history_plot(hist, w, h, now):
    ys = [n for _, n, _ in hist]
    ymax = max(4, max(ys) + 1 if ys else 4)
    ax = Axes(w, h, "Persons in view / FPS, last 60 s", (-60, 0), (0, ymax), range(-60, 1, 10),
              range(0, ymax + 1, max(1, ymax // 4)), xlabel="t [s]")
    im = ax.img
    if len(hist) > 1:
        pts = []
        for i, (t, n, _) in enumerate(hist):  # step line
            q = ax.px(t - now, n)
            if pts:
                pts.append((q[0], pts[-1][1]))
            pts.append(q)
        cv2.polylines(im, [np.int32(pts)], False, TAB10[0], 2, cv2.LINE_AA)
        fpts = [ax.px(t - now, f / 30 * ymax) for t, _, f in hist]
        cv2.polylines(im, [np.int32(fpts)], False, TAB10[1], 1, cv2.LINE_AA)
    cv2.line(im, (ax.r - 150, ax.t + 12), (ax.r - 130, ax.t + 12), TAB10[0], 2)
    put(im, "persons", (ax.r - 125, ax.t + 16), 0.36)
    cv2.line(im, (ax.r - 150, ax.t + 28), (ax.r - 130, ax.t + 28), TAB10[1], 1)
    put(im, "FPS (0-30)", (ax.r - 125, ax.t + 32), 0.36)
    return im


def side_panel(people, sysinfo, hist, now):
    img = np.full((CH, PW, 3), 245, np.uint8)
    cv2.line(img, (0, 0), (0, CH), (190, 190, 190), 1)
    y = 26
    put(img, "SYSTEM", (14, y), 0.5, INK, 1)
    y += 8
    for k, v in sysinfo:
        y += 19
        put(img, k, (14, y), 0.42, GREY)
        put(img, v, (120, y), 0.42)
    y += 16
    cv2.line(img, (10, y), (PW - 10, y), (200, 200, 200), 1)
    y += 24
    put(img, f"PERSONS  ({len(people)} tracked)", (14, y), 0.5)
    y += 8
    card_h = 128
    room = (CH - 200 - y) // card_h
    for p in people[:room]:
        y0 = y + 6
        cv2.rectangle(img, (10, y0), (PW - 10, y0 + card_h - 8), (255, 255, 255), -1)
        cv2.rectangle(img, (10, y0), (PW - 10, y0 + card_h - 8), (210, 210, 210), 1)
        cv2.rectangle(img, (10, y0), (16, y0 + card_h - 8), p.color, -1)
        put(img, f"ID {p.id}", (26, y0 + 20), 0.5, INK, 1)
        put_right(img, f"in view {now - p.first:5.1f} s   det {p.score:.2f}", (PW - 20, y0 + 20), 0.38, GREY)
        sp, ht, ag, gd, em, yaw = p.speed(), p.height(), p.age(), p.gender(), p.emotion(), p.head_yaw()
        rows = [
            ("distance", f"{p.z:.2f} m" if p.z else "-",
             f"X {p.x:+.2f}  Y {p.y:+.2f} m" if p.z else ""),
            ("height", f"{ht:.2f} m" if ht else "need full body", f"speed {sp:.2f} m/s" if sp is not None else ""),
            ("age", f"{ag[0]:.0f} y  (sd {ag[1]:.1f}, n={ag[2]})" if ag else (p.face_note or "waiting..."),
             f"{gd[0]} {gd[1]:.2f}" if gd else ""),
            ("emotion", f"{em[0]} {em[1]:.2f}" if em else "-",
             f"head yaw {yaw:+.0f} deg" if yaw is not None else ""),
            ("posture", p.posture(), ""),
        ]
        ry = y0 + 22
        for k, v1, v2 in rows:
            ry += 18
            put(img, k, (26, ry), 0.4, GREY)
            put(img, v1, (104, ry), 0.42)
            if v2:
                put(img, v2, (300, ry), 0.4)
        y += card_h
    if len(people) > room:
        put(img, f"+ {len(people) - room} more", (26, y + 18), 0.42, GREY)
    if not people:
        put(img, "no person detected", (26, y + 26), 0.45, GREY)
    hp = history_plot(hist, PW - 10, 170, now)
    img[CH - 196:CH - 26, 6:PW - 4] = hp
    put(img, "q quit   s screenshot   m mirror   space pause", (14, CH - 8), 0.38, GREY)
    return img


# ----------------------------------------------------------------------------- app
class App:
    def __init__(self, args, src, model):
        self.a, self.src, self.model = args, src, model
        self.tracker = Tracker()
        self.mirror = not args.no_mirror
        self.fps = 0.0
        self.infer_ms = 0.0
        self.hist = collections.deque(maxlen=600)
        self.frame_no = 0

    def faces(self, frame, people, now):
        fa = self.src.faces
        for p in people:
            p.face, note = p.face_crop_box()
            p.face_note = note if fa else "n/a (needs OAK device)"
            if fa and p.face and now - p.last_face_req > 0.12:
                x1, y1, x2, y2 = p.face
                fa.send(p.id, frame[y1:y2, x1:x2])
                p.last_face_req = now
        if fa:
            byid = {p.id: p for p in self.tracker.people.values()}
            for pid, kind, val in fa.poll():
                p = byid.get(pid)
                if p is None:
                    continue
                if kind == "ag":
                    p.ages.append(val[0])
                    p.males.append(val[1])
                else:
                    p.emos.append(val)

    def compose(self, frame, depth, people, now):
        if not self.a.dashboard:
            return draw_fullscreen(frame, people, f"OAK-D Lite   {self.fps:4.1f} FPS   {len(people)} person(s)")
        info = f"{self.fps:4.1f} FPS   {len(people)} person(s)"
        video = draw_video(frame.copy(), people, info)
        sysinfo = [
            ("camera", self.src.info),
            ("depth", self.src.depth_info),
            ("pose model", f"{self.model.name} ({self.model.provider}) {self.infer_ms:.0f} ms"),
            ("face models", "age/gender + emotion on Myriad X" + (f"  {self.src.faces.ms:.0f} ms"
                                                                  if self.src.faces else " (off)")),
            ("frame", f"#{self.frame_no}   {self.fps:.1f} FPS   mirror {'on' if self.mirror else 'off'}"),
        ]
        canvas = np.full((CH, CW, 3), 255, np.uint8)
        canvas[:VH, :VW] = video
        canvas[VH:, :VW // 2] = depth_view(depth, people, VW // 2, BH)
        canvas[VH:, VW // 2:VW] = top_view(people, VW // 2, BH)
        cv2.line(canvas, (VW // 2, VH), (VW // 2, CH), (210, 210, 210), 1)
        canvas[:, VW:] = side_panel(people, sysinfo, self.hist, now)
        return canvas

    def run(self, selftest=0.0):
        if not selftest:
            cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
            if self.a.dashboard:
                cv2.resizeWindow(WIN, CW, CH)
            else:
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        pool = ThreadPoolExecutor(1)
        pending, paused, canvas = None, False, None
        fcount, ft, start = 0, time.time(), time.time()

        def timed(f):
            t = time.perf_counter()
            r = self.model.infer(f)
            return r, (time.perf_counter() - t) * 1000

        while True:
            if not paused:
                frame, depth = self.src.read()
                if self.mirror:
                    frame = cv2.flip(frame, 1)
                    depth = cv2.flip(depth, 1) if depth is not None else None
                job = (pool.submit(timed, frame), frame, depth)
                if pending is None:
                    pending = job
                    continue
                (fut, frame, depth), pending = pending, job
                dets, ms = fut.result()
                self.infer_ms = 0.8 * self.infer_ms + 0.2 * ms
                now = time.time()
                people = self.tracker.update(dets, now)
                for p in people:
                    measure(p, depth, self.src, now, self.mirror)
                self.faces(frame, people, now)
                self.frame_no += 1
                fcount += 1
                if now - ft >= 1:
                    self.fps, fcount, ft = fcount / (now - ft), 0, now
                if not self.hist or now - self.hist[-1][0] > 0.1:
                    self.hist.append((now, len(people), self.fps))
                while self.hist and now - self.hist[0][0] > 60:
                    self.hist.popleft()
                canvas = self.compose(frame, depth, people, now)

            if selftest:
                if time.time() - start > selftest:
                    out = HERE / "selftest.jpg"
                    cv2.imwrite(str(out), canvas)
                    print(f"selftest: {self.fps:.1f} FPS, pose {self.infer_ms:.0f} ms, "
                          f"{len(people)} people, ages {[p.age() and round(p.age()[0]) for p in people]} -> {out}")
                    break
                continue

            cv2.imshow(WIN, canvas)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord(" "):
                paused = not paused
            elif k in (ord("m"), ord("M")):
                self.mirror = not self.mirror
            elif k in (ord("f"), ord("F")):
                full = cv2.getWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
                cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN,
                                      cv2.WINDOW_NORMAL if full else cv2.WINDOW_FULLSCREEN)
            elif k in (ord("s"), ord("S")):
                path = HERE / f"shot_{time.strftime('%Y%m%d_%H%M%S')}.png"
                cv2.imwrite(str(path), canvas)
                print("saved", path)
            if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:
                break
        pool.shutdown(wait=False)
        cv2.destroyAllWindows()


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
    if args.model != "auto":
        return PoseModel(MODELS / f"yolo11{args.model}-pose.onnx", args.conf)
    m = PoseModel(MODELS / "yolo11s-pose.onnx", args.conf)
    ms = m.bench()
    print(f"yolo11s-pose: {ms:.0f} ms")
    if ms > 60:
        print("slow CPU, using yolo11n-pose")
        m = PoseModel(MODELS / "yolo11n-pose.onnx", args.conf)
    return m


def main():
    ap = argparse.ArgumentParser(description="OAK-D Lite multi-person spatial pose analysis")
    ap.add_argument("--model", choices=["auto", "n", "s"], default="auto", help="n = faster, s = more accurate")
    ap.add_argument("--conf", type=float, default=0.4, help="person detection confidence")
    ap.add_argument("--webcam", type=int, default=0, help="webcam index used when there is no OAK")
    ap.add_argument("--webcam-only", action="store_true")
    ap.add_argument("--no-mirror", action="store_true")
    ap.add_argument("--dashboard", action="store_true", help="analysis layout with depth map, top view and data panel")
    ap.add_argument("--image", help="use a still picture instead of a camera (testing)")
    ap.add_argument("--selftest", type=float, default=0, help="run N seconds headless and save selftest.jpg")
    args = ap.parse_args()

    model = load_model(args)
    src = open_source(args)
    print("camera:", src.info, "| pose:", model.name, model.provider)
    try:
        App(args, src, model).run(args.selftest)
    finally:
        src.close()


if __name__ == "__main__":
    sys.exit(main())
