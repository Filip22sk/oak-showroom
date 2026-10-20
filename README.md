# OAK-D Lite – multi-person spatial pose demo

Fullscreen camera view that finds every person, draws their skeleton and shows per person:
distance (stereo depth), estimated age / gender / emotion (face CNNs running on the OAK chip),
body height and posture.

- **Pose:** YOLO11-pose (ONNX, runs on the PC CPU / GPU)
- **Depth + face models:** run on the OAK-D Lite (Myriad X)
- Falls back to a normal webcam if no OAK is connected (no depth, no age).

## Run on a new PC (Windows)

1. Install [Python 3.10–3.12](https://www.python.org/downloads/) (tick *Add python to PATH*) and [Git](https://git-scm.com/download/win).
2. Get the code:
   ```
   git clone https://github.com/Filip22sk/oak-showroom.git
   ```
   (or on GitHub: **Code → Download ZIP** and unzip)
3. Double-click `setup.bat` (installs packages once, needs internet).
4. Plug the OAK-D Lite into a **USB 3** port.
5. Double-click `run.bat`.

After setup no internet is needed – all models are in `models/`.

## Keys

| key | action |
|---|---|
| Q / Esc | quit |
| F | fullscreen on/off |
| M | mirror on/off |
| S | screenshot |
| Space | pause |

## Options

```
run.bat --dashboard     analysis layout: depth map, top-down view, data panel, plots
run.bat --model n       faster pose model for slow PCs (s = more accurate, default auto)
run.bat --webcam-only   use a normal webcam
run.bat --no-mirror
```

`showroom_effects.py` is an older flashy version (neon skeletons, light painting, fireworks).

## Notes

- Age/gender/emotion need the face reasonably large in the image (≈ under 4 m away). They are rough estimates.
- Height needs the full body (head to ankles) visible.
- Linux: add the udev rule once:
  `echo 'SUBSYSTEM=="usb", ATTRS{idVendor}=="03e7", MODE="0666"' | sudo tee /etc/udev/rules.d/80-movidius.rules && sudo udevadm control --reload-rules`

## Model licences

- `yolo11*-pose.onnx` – Ultralytics YOLO11, AGPL-3.0
- `age-gender-recognition-retail-0013`, `emotions-recognition-retail-0003` – Intel Open Model Zoo, Apache-2.0
