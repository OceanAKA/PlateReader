"""Synthetic plates for testing.

Lets you measure what the pipeline can and cannot recover, because you know
the ground truth. The degradations are applied in the order a real camera
applies them: geometry, then optics, then sensor sampling, then noise, then
compression.
"""
from __future__ import annotations

import random
import string

import cv2
import numpy as np

from .ocr import find_fonts
from .util import clip8

PLATE_STYLES = {
    "us": dict(size=(440, 140), bg=(245, 245, 245), fg=(28, 28, 34), border=(60, 60, 66)),
    "eu": dict(size=(520, 110), bg=(250, 250, 245), fg=(20, 20, 20), border=(40, 40, 40)),
    "dark": dict(size=(440, 140), bg=(32, 34, 40), fg=(238, 238, 230), border=(90, 90, 96)),
}


def render_plate(text: str, style: str = "us", font_path: str | None = None,
                 banner: str | None = None) -> np.ndarray:
    """Draw a plate. `banner` is the jurisdiction name printed above the
    characters, as most US states do - it is a real segmentation hazard, since
    it is text that is not part of the registration."""
    from PIL import Image, ImageDraw, ImageFont

    st = PLATE_STYLES.get(style, PLATE_STYLES["us"])
    w, h = st["size"]
    img = Image.new("RGB", (w, h), st["bg"][::-1])
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([2, 2, w - 3, h - 3], radius=int(h * 0.12),
                        outline=st["border"][::-1], width=max(2, h // 40))

    fonts = [font_path] if font_path else find_fonts(limit=1)
    if not fonts:
        raise RuntimeError("no TrueType font available to render a plate")

    band_top = 0
    if banner:
        bh = int(h * 0.20)
        bfont = ImageFont.truetype(fonts[0], max(8, int(bh * 0.85)))
        bbox = d.textbbox((0, 0), banner, font=bfont)
        d.text(((w - (bbox[2] - bbox[0])) // 2 - bbox[0], int(h * 0.045)),
               banner, fill=st["fg"][::-1], font=bfont)
        band_top = int(h * 0.22)

    avail_h = h - band_top
    size = int(avail_h * 0.66)
    while size > 8:
        font = ImageFont.truetype(fonts[0], size)
        box = d.textbbox((0, 0), text, font=font)
        if (box[2] - box[0]) <= w * 0.86 and (box[3] - box[1]) <= avail_h * 0.78:
            break
        size -= 2
    font = ImageFont.truetype(fonts[0], size)
    box = d.textbbox((0, 0), text, font=font)
    tx = (w - (box[2] - box[0])) // 2 - box[0]
    ty = band_top + (avail_h - (box[3] - box[1])) // 2 - box[1]
    d.text((tx, ty), text, fill=st["fg"][::-1], font=font)

    if banner:            # mounting bolts, like the real thing
        for bx in (int(w * 0.30), int(w * 0.70)):
            d.ellipse([bx - 5, 6, bx + 5, 16], fill=(120, 120, 120))
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def vehicle_rear(plate: np.ndarray, body=(232, 232, 236), kind: str = "truck",
                 size=(1280, 960), plate_frac: float = 0.22,
                 seed: int | None = None) -> np.ndarray:
    """Compose the plate onto a plausible vehicle rear.

    The point is not photorealism, it is giving the detector the distractors a
    real photo has: tail lights, a badge, tailgate seams and a bumper, all of
    which produce exactly the strong horizontal and vertical edges that a naive
    plate detector latches onto.
    """
    rng = np.random.default_rng(seed)
    W, H = size
    img = np.full((H, W, 3), 150, np.uint8)

    # background: ground plane and some vertical clutter
    cv2.rectangle(img, (0, 0), (W, int(H * 0.42)), (128, 132, 130), -1)
    cv2.rectangle(img, (0, int(H * 0.42)), (W, H), (96, 96, 98), -1)
    for _ in range(14):
        x = int(rng.integers(0, W))
        y = int(rng.integers(0, int(H * 0.45)))
        cv2.rectangle(img, (x, y), (x + int(rng.integers(30, 180)),
                                    y + int(rng.integers(40, 200))),
                      tuple(int(c) for c in rng.integers(70, 165, 3)), -1)
    img = cv2.GaussianBlur(img, (0, 0), 2.2)

    # body
    bx0, bx1 = int(W * 0.06), int(W * 0.94)
    by0, by1 = int(H * 0.18), int(H * 0.86)
    cv2.rectangle(img, (bx0, by0), (bx1, by1), body, -1)
    cv2.rectangle(img, (bx0, by0), (bx1, by1), tuple(int(c * 0.72) for c in body), 3)

    # rear window
    wy1 = by0 + int((by1 - by0) * (0.34 if kind == "truck" else 0.40))
    cv2.rectangle(img, (bx0 + int(W * 0.05), by0 + int(H * 0.02)),
                  (bx1 - int(W * 0.05), wy1), (46, 52, 56), -1)

    # tailgate seam + handle
    seam = wy1 + int((by1 - wy1) * 0.10)
    cv2.line(img, (bx0 + 6, seam), (bx1 - 6, seam), tuple(int(c * 0.80) for c in body), 3)
    hx = (bx0 + bx1) // 2
    cv2.rectangle(img, (hx - int(W * 0.05), seam + int(H * 0.02)),
                  (hx + int(W * 0.05), seam + int(H * 0.05)), (60, 62, 66), -1)

    # tail lights
    ly0, ly1 = wy1 + int((by1 - wy1) * 0.22), wy1 + int((by1 - wy1) * 0.55)
    for lx0, lx1 in ((bx0 + 8, bx0 + int(W * 0.11)), (bx1 - int(W * 0.11), bx1 - 8)):
        cv2.rectangle(img, (lx0, ly0), (lx1, ly1), (36, 40, 190), -1)
        cv2.rectangle(img, (lx0, ly0), (lx1, ly0 + (ly1 - ly0) // 3), (190, 200, 210), -1)

    # badge
    cv2.ellipse(img, (hx, seam + int(H * 0.10)), (int(W * 0.045), int(H * 0.018)),
                0, 0, 360, tuple(int(c * 0.66) for c in body), -1)

    # bumper
    cv2.rectangle(img, (bx0, by1 - int(H * 0.10)), (bx1, by1), (70, 72, 76), -1)

    # mount the plate on the bumper
    pw = int(W * plate_frac)
    ph = max(8, int(pw * plate.shape[0] / plate.shape[1]))
    p = cv2.resize(plate, (pw, ph), interpolation=cv2.INTER_AREA)
    px = hx - pw // 2
    py = by1 - int(H * 0.10) + (int(H * 0.10) - ph) // 2
    py = max(0, min(H - ph, py))
    px = max(0, min(W - pw, px))
    cv2.rectangle(img, (px - 3, py - 3), (px + pw + 3, py + ph + 3), (30, 30, 32), -1)
    img[py:py + ph, px:px + pw] = p
    return img


def perspective(img: np.ndarray, strength: float, side: str = "right") -> np.ndarray:
    """Simulate viewing the plate from an angle."""
    if abs(strength) < 1e-3:
        return img
    h, w = img.shape[:2]
    d = float(np.clip(strength, 0, 0.9)) * w * 0.32
    v = float(np.clip(strength, 0, 0.9)) * h * 0.30
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], np.float32)
    if side == "right":
        dst = np.array([[0, 0], [w - d, v], [w - d, h - v], [0, h]], np.float32)
    elif side == "left":
        dst = np.array([[d, v], [w, 0], [w, h], [d, h - v]], np.float32)
    else:  # from below
        dst = np.array([[d, 0], [w - d, 0], [w, h], [0, h]], np.float32)
    m = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, m, (w, h), borderMode=cv2.BORDER_REPLICATE)


def rotate(img: np.ndarray, angle: float) -> np.ndarray:
    if abs(angle) < 1e-3:
        return img
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), borderMode=cv2.BORDER_REPLICATE)


def motion_blur(img: np.ndarray, length: float, angle: float) -> np.ndarray:
    if length < 1.5:
        return img
    from .restore import motion_psf
    psf = motion_psf(length, angle)
    return clip8(cv2.filter2D(img.astype(np.float64), -1, psf))


def defocus(img: np.ndarray, sigma: float) -> np.ndarray:
    return img if sigma <= 0.05 else cv2.GaussianBlur(img, (0, 0), sigma)


def downscale(img: np.ndarray, factor: float) -> np.ndarray:
    if factor >= 0.999:
        return img
    h, w = img.shape[:2]
    return cv2.resize(img, (max(8, int(w * factor)), max(4, int(h * factor))),
                      interpolation=cv2.INTER_AREA)


def screen_moire(img: np.ndarray, period: float = 3.2, angle: float = 11.0,
                 strength: float = 0.30) -> np.ndarray:
    """Simulate photographing a monitor: the display's pixel grid, beating.

    Two crossed sinusoids at a slight angle to the sensor grid, applied
    multiplicatively - which is what a re-photographed screen looks like, and
    what the descreen filter is meant to remove.
    """
    if strength <= 0:
        return img
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    th = np.deg2rad(angle)
    u = xx * np.cos(th) + yy * np.sin(th)
    v = -xx * np.sin(th) + yy * np.cos(th)
    grid = (0.5 + 0.5 * np.cos(2 * np.pi * u / period)) * \
           (0.5 + 0.5 * np.cos(2 * np.pi * v / max(period * 1.15, 0.5)))
    gain = (1.0 - strength) + strength * 2.0 * grid
    if img.ndim == 3:
        gain = gain[..., None]
    return clip8(img.astype(np.float64) * gain)


def add_noise(img: np.ndarray, sigma: float, seed: int | None = None) -> np.ndarray:
    """Additive sensor noise.

    Seeded deliberately: an unseeded generator here makes every measurement
    run-to-run noisy, and a benchmark that moves by a plate or two between
    identical runs cannot tell a real regression from the dice.
    """
    if sigma <= 0:
        return img
    rng = np.random.default_rng(seed)
    return clip8(img.astype(np.float64) + rng.normal(0, sigma, img.shape))


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    if quality <= 0 or quality >= 100:
        return img
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    return cv2.imdecode(buf, cv2.IMREAD_COLOR) if ok else img


def place_in_scene(plate: np.ndarray, scene_size=(1024, 640), seed: int | None = None) -> np.ndarray:
    """Drop the plate into a cluttered background so detection has work to do."""
    rng = np.random.default_rng(seed)
    W, H = scene_size
    scene = np.full((H, W, 3), 90, np.uint8)
    scene = clip8(scene.astype(np.float64) + rng.normal(0, 18, scene.shape))
    for _ in range(28):
        x, y = int(rng.integers(0, W)), int(rng.integers(0, H))
        w, h = int(rng.integers(20, 220)), int(rng.integers(10, 130))
        col = tuple(int(c) for c in rng.integers(35, 190, 3))
        cv2.rectangle(scene, (x, y), (x + w, y + h), col, -1)
    scene = cv2.GaussianBlur(scene, (0, 0), 1.6)

    ph, pw = plate.shape[:2]
    if pw >= W or ph >= H:
        plate = cv2.resize(plate, (min(pw, W - 20), min(ph, H - 20)))
        ph, pw = plate.shape[:2]
    x = int(rng.integers(10, max(11, W - pw - 10)))
    y = int(rng.integers(10, max(11, H - ph - 10)))
    scene[y:y + ph, x:x + pw] = plate
    return scene


def make_sample(text: str = "ABC1234", style: str = "us", rot: float = 0.0,
                persp: float = 0.0, motion: float = 0.0, motion_angle: float = 0.0,
                blur: float = 0.0, scale: float = 1.0, noise: float = 0.0,
                quality: int = 0, scene: bool = False, seed: int | None = None,
                moire: float = 0.0) -> np.ndarray:
    """Full degradation chain, camera order."""
    img = render_plate(text, style)
    img = perspective(img, persp)
    img = rotate(img, rot)
    img = motion_blur(img, motion, motion_angle)
    img = defocus(img, blur)
    img = downscale(img, scale)
    if moire > 0:
        img = screen_moire(img, strength=moire)
    img = add_noise(img, noise, seed)
    img = jpeg(img, quality)
    if scene:
        img = place_in_scene(img, seed=seed)
    return img


# --------------------------------------------------------------------------
# camera geometry
# --------------------------------------------------------------------------

def camera_homography(src_w: int, src_h: int, yaw: float, pitch: float,
                      roll: float, distance: float, focal: float,
                      plate_w: float = 0.52) -> np.ndarray:
    """True projective transform for a plane viewed from a given pose.

    Not a trapezoid nudge: the scene is treated as a plane in 3D, rotated by
    yaw (camera round the side), pitch (camera above, looking down) and roll
    (camera tilted), then projected through a pinhole. It matters because a
    pole-mounted camera applies pitch *and* yaw at once, and the result is a
    general quadrilateral with foreshortening in both axes - which a
    single-axis trapezoid cannot produce and the rectifier therefore never
    gets tested against.

    distance is in metres, focal in pixels, plate_w the real width in metres.
    """
    m_per_px = plate_w / max(1, src_w)
    # pixels -> metres, origin at the plane's centre
    S = np.array([[m_per_px, 0, -src_w * m_per_px / 2.0],
                  [0, m_per_px, -src_h * m_per_px / 2.0],
                  [0, 0, 1.0]])

    cy_, sy = np.cos(np.deg2rad(yaw)), np.sin(np.deg2rad(yaw))
    cp, sp = np.cos(np.deg2rad(pitch)), np.sin(np.deg2rad(pitch))
    cr, sr = np.cos(np.deg2rad(roll)), np.sin(np.deg2rad(roll))
    Ry = np.array([[cy_, 0, sy], [0, 1, 0], [-sy, 0, cy_]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]])
    R = Rz @ Rx @ Ry

    t = np.array([0.0, 0.0, float(distance)])
    K = np.array([[focal, 0, 0.0], [0, focal, 0.0], [0, 0, 1.0]])
    # a plane's homography uses only the first two rotation columns
    H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
    return H


def apply_camera(img: np.ndarray, yaw: float = 0.0, pitch: float = 0.0,
                 roll: float = 0.0, distance: float = 8.0, focal: float = 1400.0,
                 plate_w: float = 0.52, out_size=None,
                 bg: int = 110) -> np.ndarray:
    """Render a flat scene as seen from a camera pose, framed to fit."""
    h, w = img.shape[:2]
    H = camera_homography(w, h, yaw, pitch, roll, distance, focal, plate_w)

    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], np.float64).T
    proj = H @ corners
    proj = proj[:2] / proj[2]
    x0, y0 = proj[0].min(), proj[1].min()
    x1, y1 = proj[0].max(), proj[1].max()

    ow, oh = out_size or (int(np.ceil(x1 - x0)), int(np.ceil(y1 - y0)))
    ow, oh = max(32, min(ow, 4000)), max(32, min(oh, 4000))
    # centre the projected scene in the output frame
    sx = ow / max(1e-6, (x1 - x0))
    sy = oh / max(1e-6, (y1 - y0))
    s = min(sx, sy) * 0.92
    T = np.array([[s, 0, ow / 2 - s * (x0 + x1) / 2],
                  [0, s, oh / 2 - s * (y0 + y1) / 2],
                  [0, 0, 1.0]])

    border = (int(bg),) * 3 if img.ndim == 3 else int(bg)
    return cv2.warpPerspective(img, T @ H, (ow, oh), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_CONSTANT, borderValue=border)


def barrel(img: np.ndarray, k: float = -0.18) -> np.ndarray:
    """Wide-angle lens distortion, which entry and barrier cameras all have."""
    if abs(k) < 1e-4:
        return img
    h, w = img.shape[:2]
    f = max(w, h)
    K = np.array([[f, 0, w / 2.0], [0, f, h / 2.0], [0, 0, 1.0]])
    dist = np.array([k, 0.0, 0.0, 0.0, 0.0])
    return cv2.undistort(img, K, dist)


# Where plate cameras actually sit. Angles in degrees, distance in metres.
CAMERA_PRESETS: dict[str, dict] = {
    "gantry":        dict(yaw=6, pitch=34, roll=1, distance=6.5, focal=2200,
                          barrel=0.0,
                          note="overhead toll//motorway gantry, looking down"),
    "pole-cctv":     dict(yaw=32, pitch=24, roll=3, distance=11.0, focal=2600,
                          barrel=0.0,
                          note="street pole: above and off to one side"),
    "parking-entry": dict(yaw=41, pitch=14, roll=2, distance=3.2, focal=900,
                          barrel=-0.16,
                          note="car park entry: close, wide lens, sharp angle"),
    "barrier":       dict(yaw=52, pitch=27, roll=4, distance=2.6, focal=800,
                          barrel=-0.22,
                          note="barrier arm camera: very close and very oblique"),
    "roadside":      dict(yaw=34, pitch=9, roll=2, distance=14.0, focal=3000,
                          barrel=0.0,
                          note="roadside speed camera, near level, oblique"),
    "dashcam":       dict(yaw=7, pitch=4, roll=2, distance=9.0, focal=1500,
                          barrel=-0.10,
                          note="dashcam following the car ahead"),
}


# --------------------------------------------------------------------------
# named scenarios - the real-world situations worth testing against
# --------------------------------------------------------------------------

SCENARIOS: dict[str, dict] = {
    "truck-screen": dict(
        text="TXR4821", style="us", banner="TEXAS", body=(236, 236, 238),
        kind="truck", size=(1200, 1500), plate_frac=0.20,
        note="white pickup, plate photographed off a monitor (moire)",
        degrade=dict(moire=0.18, blur=0.7, noise=4, quality=82),
    ),
    "truck-screen-hard": dict(
        text="TXR4821", style="us", banner="TEXAS", body=(236, 236, 238),
        kind="truck", size=(1200, 1500), plate_frac=0.15,
        note="same, further away and a coarser screen pattern",
        degrade=dict(moire=0.30, blur=1.1, noise=6, quality=70),
    ),
    "eu-defocus": dict(
        text="KH5299", style="eu", banner=None, body=(52, 86, 66),
        kind="car", size=(1100, 800), plate_frac=0.26,
        note="dark green car, badly out of focus - the honesty case",
        degrade=dict(blur=4.5, noise=5, quality=75),
    ),
    "cctv-small": dict(
        text="VWB2740", style="us", banner=None, body=(238, 240, 242),
        kind="car", size=(1000, 700), plate_frac=0.10,
        note="surveillance still: small plate, compressed, slightly soft",
        degrade=dict(scale=0.5, blur=0.9, noise=7, quality=45),
    ),
    "cctv-tiny": dict(
        text="VWB2740", style="us", banner=None, body=(238, 240, 242),
        kind="car", size=(1000, 700), plate_frac=0.06,
        note="same camera, further back - expected to be unreadable",
        degrade=dict(scale=0.45, blur=1.2, noise=9, quality=40),
    ),
    "angled-street": dict(
        text="BNK7315", style="us", banner="TEXAS", body=(70, 74, 82),
        kind="car", size=(1200, 900), plate_frac=0.24,
        note="dark car photographed from the side of the road",
        degrade=dict(persp=0.30, rot=-11, motion=6, noise=5, quality=80),
    ),
}


def make_camera_view(camera: str, text: str = "TXR4821", style: str = "us",
                     banner: str | None = "TEXAS", body=(228, 230, 234),
                     kind: str = "car", scene_size=(1100, 800),
                     plate_frac: float = 0.22, degrade: dict | None = None,
                     seed: int | None = 7) -> tuple[np.ndarray, str]:
    """Render a vehicle as one of the standard camera positions would see it.

    The homography is applied to the whole vehicle, not just the plate, which
    treats the car's rear as planar - close enough for a tailgate, and it
    keeps the surrounding structure (lights, bumper, badge) distorted
    consistently with the plate, so the detector faces a coherent scene rather
    than a warped plate pasted onto a straight-on car.
    """
    if camera not in CAMERA_PRESETS:
        raise ValueError(f"unknown camera {camera!r}; "
                         f"known: {', '.join(sorted(CAMERA_PRESETS))}")
    cam = CAMERA_PRESETS[camera]
    truth = text.upper()

    plate = render_plate(truth, style, banner=banner)
    scene = vehicle_rear(plate, body=body, kind=kind, size=scene_size,
                         plate_frac=plate_frac, seed=seed)

    # the plate is plate_frac of the scene width; tell the camera model how
    # wide the whole scene is in metres so the geometry is to scale
    scene_w_m = 0.52 / max(plate_frac, 1e-3)
    view = apply_camera(scene, yaw=cam["yaw"], pitch=cam["pitch"],
                        roll=cam["roll"], distance=cam["distance"],
                        focal=cam["focal"], plate_w=scene_w_m)
    if cam.get("barrel"):
        view = barrel(view, cam["barrel"])

    d = degrade or {}
    view = motion_blur(view, d.get("motion", 0.0), d.get("motion_angle", 0.0))
    view = defocus(view, d.get("blur", 0.0))
    view = downscale(view, d.get("scale", 1.0))
    if d.get("moire", 0.0) > 0:
        view = screen_moire(view, strength=d["moire"])
    view = add_noise(view, d.get("noise", 0.0), seed)
    view = jpeg(view, d.get("quality", 0))
    return view, truth


def make_scenario(name: str, text: str | None = None,
                  seed: int | None = 7) -> tuple[np.ndarray, str]:
    """Render one named scenario. Returns (image, ground truth text)."""
    if name not in SCENARIOS:
        raise ValueError(f"unknown scenario {name!r}; "
                         f"known: {', '.join(sorted(SCENARIOS))}")
    spec = SCENARIOS[name]
    truth = (text or spec["text"]).upper()

    plate = render_plate(truth, spec["style"], banner=spec.get("banner"))
    img = vehicle_rear(plate, body=spec["body"], kind=spec["kind"],
                       size=spec["size"], plate_frac=spec["plate_frac"], seed=seed)

    d = spec["degrade"]
    img = perspective(img, d.get("persp", 0.0))
    img = rotate(img, d.get("rot", 0.0))
    img = motion_blur(img, d.get("motion", 0.0), d.get("motion_angle", 0.0))
    img = defocus(img, d.get("blur", 0.0))
    img = downscale(img, d.get("scale", 1.0))
    if d.get("moire", 0.0) > 0:
        img = screen_moire(img, strength=d["moire"])
    img = add_noise(img, d.get("noise", 0.0), seed)
    img = jpeg(img, d.get("quality", 0))
    return img, truth


def random_text(pattern: str = "LLLDDDD", rng: random.Random | None = None) -> str:
    rng = rng or random.Random()
    out = []
    for c in pattern:
        if c == "L":
            out.append(rng.choice(string.ascii_uppercase))
        elif c == "D":
            out.append(rng.choice(string.digits))
        else:
            out.append(c)
    return "".join(out)
