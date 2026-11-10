"""
Stop-motion style movie: "How my neural net counts fingers" (The Impenetrable Safe).

Paper-cutout look: kraft-paper background, cut-out cards with shadows, and every
piece wobbles slightly between frames like it was moved by hand.

Run:   ~/.venvs/impenetrable-safe/bin/python make_movie.py            (renders both videos + script.md)
       ~/.venvs/impenetrable-safe/bin/python make_movie.py --scene 3  (renders one scene, for quick checks)
       ~/.venvs/impenetrable-safe/bin/python make_movie.py --still 3 0.5   (saves one PNG: scene 3 at 50%)

Outputs:
  neural_net_movie.mp4         clean video (add your voiceover to this one)
  neural_net_movie_GUIDE.mp4   same video with your lines at the bottom, highlighted
                               as you go - watch this while you record
  script.md                    the script with timings
"""
import math
import random
import sys
from functools import lru_cache
from pathlib import Path

import imageio_ffmpeg
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

HERE = Path(__file__).parent
W, H, FPS = 1280, 720, 12
WORDS_PER_SEC = 2.4          # comfortable reading pace

# ---------------------------------------------------------------------------
# The script (one source for the video timing, the guide subtitles and script.md)
# ---------------------------------------------------------------------------
SCENES = [
    ("Title", "The Impenetrable Safe",
     "This is the Impenetrable Safe. It only opens when I show it the right three numbers "
     "with my fingers. In this video I'll explain how the neural network inside it figures "
     "out how many fingers I'm holding up."),
    ("Pixels", "A picture is just numbers",
     "A webcam looks straight down at the safe, where my hand goes between two red markers. "
     "The whole picture is shrunk to 112 by 64 pixels. To the computer, every pixel is just "
     "three numbers, for red, green and blue. So the network starts with 21,504 numbers."),
    ("Convolution", "Layer 1: convolution",
     "The first layer is a convolution. A filter is a small grid of weights. It slides "
     "across the image, and at each spot it multiplies the pixels under it by its weights "
     "and adds them up. When the patch matches the filter's pattern, like the edge of a "
     "finger, the answer is big. My first layer has 16 different filters, so it makes 16 "
     "new images called feature maps."),
    ("ReLU", "ReLU: keep the positives",
     "Next comes ReLU. Every negative number becomes zero, and positive numbers stay the "
     "same. This is the same bend we used to fit the V data. Without it, stacking layers "
     "would just add up to one big straight line."),
    ("Pooling", "Max pooling: shrink it",
     "Then max pooling. In every two by two block, only the biggest number survives. The "
     "feature maps shrink to half the size, but the strong features stay. It also means a "
     "finger that moves a few pixels still gives the same answer."),
    ("Three layers", "Repeat three times",
     "The network does this three times: convolution, ReLU, pooling. Each time the images "
     "get smaller, but there are more of them: 16, then 32, then 64 feature maps. The "
     "early filters find simple edges. The later ones find bigger shapes, like fingertips "
     "and the gaps between fingers."),
    ("Linear layer", "Flatten + linear layer",
     "At the end, the 64 maps of 8 by 14 are flattened into one long list of 7,168 numbers. "
     "A linear layer connects every one of those numbers to seven outputs, one for each "
     "possible answer: zero to five fingers, or blank, when there's no hand at all."),
    ("Softmax", "Softmax: the answer",
     "Softmax turns the seven scores into probabilities that add up to one hundred percent. "
     "The biggest one is the answer. Here the network says three fingers, and it's 92 "
     "percent sure."),
    ("Training", "How it learns",
     "But how does it learn what the filters should be? Training. I showed it hundreds of "
     "labeled photos. When it guesses wrong, cross-entropy loss measures how wrong it was. "
     "Backpropagation uses the chain rule to find how much each weight caused the error, "
     "and gradient descent nudges every weight a little bit downhill. It's the same idea "
     "as m and b in my spreadsheet, just with about 49,000 weights instead of two."),
    ("Augmentation", "More practice",
     "To give it more practice, every training photo is randomly rotated, shifted, "
     "flipped and made brighter or darker. The network sees a slightly different version "
     "every time, so it learns what fingers look like instead of memorizing the photos."),
    ("The lock", "Opening the safe",
     "On the safe, I press the button and the lights come on. A digit only counts if the "
     "network is sure for a full second, and I have to take my hand away before the next "
     "one. After three digits, if they match the secret code, the servo swings the latch "
     "and the safe opens."),
    ("End", "Thanks for watching!",
     "And that's how the Impenetrable Safe counts fingers. Thanks for watching!"),
]


def scene_duration(text):
    return round(max(5.0, len(text.split()) / WORDS_PER_SEC + 2.0), 1)


DURS = [scene_duration(s[2]) for s in SCENES]

# ---------------------------------------------------------------------------
# Paper-cutout drawing kit
# ---------------------------------------------------------------------------
KRAFT = (214, 190, 150)
INK = (45, 38, 32)
PAPER = (250, 246, 236)
SKIN = (240, 196, 150)
SKIN_DARK = (214, 160, 118)
RED = (214, 64, 52)
ORANGE = (240, 150, 50)
YELLOW = (246, 206, 80)
BLUE = (70, 130, 200)
GREEN = (70, 170, 90)
GREY = (120, 124, 130)
DARK = (60, 62, 68)

HOLD = [0]    # which "pose" we're on - pieces jiggle every 2 frames (shooting "on twos")


def find_font(name):
    for d in ("/System/Library/Fonts/Supplemental/", "/System/Library/Fonts/", "/Library/Fonts/"):
        p = Path(d) / name
        if p.exists():
            return str(p)
    return None


FONT_FILES = {"title": find_font("MarkerFelt.ttc"), "hand": find_font("ChalkboardSE.ttc"),
              "mono": find_font("Menlo.ttc")}


@lru_cache(maxsize=None)
def font(size, kind="hand"):
    path = FONT_FILES.get(kind)
    if path:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size)


def ease(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3 - 2 * t)


def seg(sec, start, dur):
    """0 before `start`, 1 after `start+dur`, eased in between."""
    return ease((sec - start) / dur) if dur > 0 else float(sec >= start)


def lerp(a, b, t):
    return a + (b - a) * t


@lru_cache(maxsize=1)
def background():
    rng = np.random.default_rng(3)
    noise = rng.normal(0, 1, (H // 4, W // 4))
    img = Image.fromarray(np.uint8(np.clip(128 + noise * 40, 0, 255))).resize((W, H), Image.BICUBIC)
    img = img.filter(ImageFilter.GaussianBlur(2))
    n = np.asarray(img, dtype=np.float32) / 255 - 0.5
    fine = rng.normal(0, 1, (H, W)) * 0.04
    base = np.array(KRAFT, dtype=np.float32)
    arr = base[None, None, :] * (1 + n[..., None] * 0.10 + fine[..., None])
    # a few paper fibres
    im = Image.fromarray(np.uint8(np.clip(arr, 0, 255)))
    d = ImageDraw.Draw(im)
    for _ in range(260):
        x, y = rng.uniform(0, W), rng.uniform(0, H)
        a, L = rng.uniform(0, math.pi), rng.uniform(6, 22)
        c = tuple(int(v * rng.uniform(0.85, 1.05)) for v in KRAFT)
        d.line([(x, y), (x + L * math.cos(a), y + L * math.sin(a))], fill=c, width=1)
    return im


def rough_poly(w, h, rnd, rough):
    pts = []
    n = 6
    for i in range(n):
        pts.append((w * i / n + rnd.uniform(-rough, rough), rnd.uniform(0, rough)))
    for i in range(n):
        pts.append((w - rnd.uniform(0, rough), h * i / n + rnd.uniform(-rough, rough)))
    for i in range(n):
        pts.append((w - w * i / n + rnd.uniform(-rough, rough), h - rnd.uniform(0, rough)))
    for i in range(n):
        pts.append((rnd.uniform(0, rough), h - h * i / n + rnd.uniform(-rough, rough)))
    return pts


def card(w, h, fill=PAPER, key="card", rough=2.5):
    """A cut piece of paper (RGBA) with slightly uneven edges and grain."""
    w, h = int(w), int(h)
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    rnd = random.Random(key)
    ImageDraw.Draw(img).polygon(rough_poly(w - 1, h - 1, rnd, rough), fill=fill + (255,))
    grain = np.random.default_rng(abs(hash(key)) % 2**32).normal(0, 6, (h, w, 1))
    arr = np.asarray(img).astype(np.float32)
    arr[..., :3] = np.clip(arr[..., :3] + grain, 0, 255)
    return Image.fromarray(arr.astype(np.uint8), "RGBA")


GLYPHS = {"×": "x", "→": "->", "≈": "~", "✗": "X", "✓": "OK", "—": "-", "±": "+/-"}


def safe_text(text):
    for a, b in GLYPHS.items():
        text = text.replace(a, b)
    return text


def text_img(text, size, color=INK, kind="hand"):
    text = safe_text(text)
    f = font(size, kind)
    box = ImageDraw.Draw(Image.new("RGBA", (1, 1))).textbbox((0, 0), text, font=f)
    img = Image.new("RGBA", (box[2] - box[0] + 4, box[3] - box[1] + 6), (0, 0, 0, 0))
    ImageDraw.Draw(img).text((2 - box[0], 3 - box[1]), text, font=f, fill=color + (255,))
    return img


def label(text, size=30, fill=PAPER, color=INK, kind="hand", pad=(18, 10), key=None):
    t = text_img(text, size, color, kind)
    c = card(t.width + 2 * pad[0], t.height + 2 * pad[1], fill, key or f"label-{text}")
    c.alpha_composite(t, (pad[0], pad[1]))
    return c


def place(frame, img, cx, cy, key, rot=0.0, jit=1.0, shadow=True, alpha=1.0):
    """Paste a cutout centered at (cx, cy) with a hand-moved wobble and a drop shadow."""
    if alpha <= 0.01:
        return
    r = random.Random(f"{key}|{HOLD[0]}")
    dx, dy = r.uniform(-1.6, 1.6) * jit, r.uniform(-1.6, 1.6) * jit
    ang = rot + r.uniform(-0.7, 0.7) * jit
    im = img
    if abs(ang) > 0.05:
        im = im.rotate(ang, resample=Image.BICUBIC, expand=True)
    if alpha < 1:
        a = im.getchannel("A").point(lambda v: int(v * alpha))
        im = im.copy()
        im.putalpha(a)
    x, y = int(cx - im.width / 2 + dx), int(cy - im.height / 2 + dy)
    if shadow:
        m = im.getchannel("A").point(lambda v: int(v * 0.42)).filter(ImageFilter.GaussianBlur(4))
        frame.paste(Image.new("RGB", im.size, (70, 50, 30)), (x + 5, y + 7), m)
    frame.paste(im.convert("RGB"), (x, y), im.getchannel("A"))


def arrow(frame, x1, y1, x2, y2, color=INK, width=6, t=1.0, key="arrow"):
    """A hand-drawn-ish arrow that can grow in (t = 0..1)."""
    if t <= 0:
        return
    r = random.Random(f"{key}|{HOLD[0]}")
    j = lambda: r.uniform(-1.2, 1.2)
    x2, y2 = lerp(x1, x2, t), lerp(y1, y2, t)
    d = ImageDraw.Draw(frame)
    d.line([(x1 + j(), y1 + j()), (x2 + j(), y2 + j())], fill=color, width=width)
    a = math.atan2(y2 - y1, x2 - x1)
    L = 18 + width
    for s in (-1, 1):
        d.line([(x2, y2), (x2 - L * math.cos(a + s * 0.5), y2 - L * math.sin(a + s * 0.5))],
               fill=color, width=width)


def scene_tag(frame, i, sec):
    tag = label(f"{i}  ·  {SCENES[i][1]}", 30, YELLOW, key=f"tag{i}")
    place(frame, tag, 30 + tag.width / 2, lerp(-60, 50, seg(sec, 0, 0.6)), f"tag{i}", rot=-1.5)


# ---------------------------------------------------------------------------
# Props
# ---------------------------------------------------------------------------
@lru_cache(maxsize=None)
def hand(n, size=360, flip=False):
    """Paper cutout of a hand (seen from above) holding up n fingers."""
    s = size
    img = Image.new("RGBA", (s, int(s * 1.25)), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    cx, palm_y = s * 0.5, s * 0.78
    pw, ph = s * 0.46, s * 0.40
    d.rectangle([cx - s * 0.16, palm_y, cx + s * 0.16, s * 1.25], fill=SKIN)           # wrist
    d.ellipse([cx - pw / 2, palm_y - ph / 2, cx + pw / 2, palm_y + ph / 2], fill=SKIN)  # palm
    fw = s * 0.085
    # four fingers (index..pinky) + thumb; extended ones are long, folded ones are knuckles
    order = [1, 2, 3, 4, 0]     # which fingers go up for n = 1..5: index, middle, ring, pinky, thumb
    up = set(order[:n])
    angles = {1: -12, 2: -3, 3: 6, 4: 15}
    bases = {1: -0.135, 2: -0.045, 3: 0.045, 4: 0.13}
    for f in (1, 2, 3, 4):
        bx, by = cx + bases[f] * s, palm_y - ph * 0.38
        L = (s * [0, 0.42, 0.47, 0.43, 0.34][f]) if f in up else s * 0.07
        a = math.radians(angles[f] - 90)
        tx, ty = bx + L * math.cos(a), by + L * math.sin(a)
        d.line([(bx, by), (tx, ty)], fill=SKIN, width=int(fw))
        d.ellipse([tx - fw / 2, ty - fw / 2, tx + fw / 2, ty + fw / 2], fill=SKIN)
        if f in up:   # fingernail
            nx, ny = bx + (L - fw * 0.55) * math.cos(a), by + (L - fw * 0.55) * math.sin(a)
            d.ellipse([nx - fw * 0.28, ny - fw * 0.36, nx + fw * 0.28, ny + fw * 0.36], fill=(248, 220, 196))
    # thumb
    bx, by = cx - pw * 0.42, palm_y + ph * 0.05
    if 0 in up:
        a, L = math.radians(-150), s * 0.30
    else:
        a, L = math.radians(-60), s * 0.12
    tx, ty = bx + L * math.cos(a), by + L * math.sin(a)
    d.line([(bx, by), (tx, ty)], fill=SKIN, width=int(fw * 1.15))
    d.ellipse([tx - fw * .58, ty - fw * .58, tx + fw * .58, ty + fw * .58], fill=SKIN)
    # creases
    d.arc([cx - pw * 0.3, palm_y - ph * 0.1, cx + pw * 0.3, palm_y + ph * 0.45], 200, 330, fill=SKIN_DARK, width=3)
    if flip:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # paper edge: thin outline
    a = img.getchannel("A")
    edge = a.filter(ImageFilter.MaxFilter(5))
    out = Image.new("RGBA", img.size, (0, 0, 0, 0))
    out.paste(Image.new("RGBA", img.size, (255, 252, 245, 255)), (0, 0), edge)
    out.alpha_composite(img)
    return out


@lru_cache(maxsize=None)
def camera_view(n, size=300):
    """What the webcam sees (the whole wide frame): dark safe top, red markers on the
    left and right, hand in between. `size` is the width; height = width * 64 / 112."""
    w, h = size, int(round(size * 64 / 112))
    img = Image.new("RGBA", (w, h), (55, 58, 64, 255))
    d = ImageDraw.Draw(img)
    m = int(w * 0.07)
    d.rectangle([0, 0, m, h], fill=RED)
    d.rectangle([w - m, 0, w, h], fill=RED)
    hd = hand(n, int(h * 0.72))
    img.alpha_composite(hd, (int(w / 2 - hd.width / 2), int(h * 1.02 - hd.height)))
    return img


@lru_cache(maxsize=None)
def pixelated(n, px=(112, 64), size=300):
    view = camera_view(n, size)
    small = view.convert("RGB").resize(px, Image.BOX)
    return small, small.resize(view.size, Image.NEAREST).convert("RGBA")


def number_grid(values, cell=56, colors=None, key="grid", fsize=26, fill=PAPER, highlight=None):
    rows, cols = len(values), len(values[0])
    img = card(cols * cell + 12, rows * cell + 12, fill, key)
    d = ImageDraw.Draw(img)
    for r in range(rows):
        for c in range(cols):
            x0, y0 = 6 + c * cell, 6 + r * cell
            col = colors[r][c] if colors else None
            if col:
                d.rectangle([x0 + 2, y0 + 2, x0 + cell - 2, y0 + cell - 2], fill=col)
            if highlight and (r, c) in highlight:
                d.rectangle([x0 + 1, y0 + 1, x0 + cell - 1, y0 + cell - 1], outline=highlight[(r, c)], width=5)
            d.rectangle([x0, y0, x0 + cell, y0 + cell], outline=(150, 140, 125), width=1)
            v = values[r][c]
            if v is None or v == "":
                continue
            txt = str(v)
            f = font(fsize)
            bb = d.textbbox((0, 0), txt, font=f)
            dark = col is not None and sum(col[:3]) < 330
            d.text((x0 + cell / 2 - (bb[2] - bb[0]) / 2 - bb[0], y0 + cell / 2 - (bb[3] - bb[1]) / 2 - bb[1]),
                   txt, font=f, fill=(250, 246, 236) if dark else INK)
    return img


def safe_img(leds=0, status=None, door=0.0, size=330):
    """Cardboard safe: body, 3 progress LEDs, status light, door that swings open (door 0..1)."""
    w, h = size, int(size * 0.95)
    img = Image.new("RGBA", (w + 120, h), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    d.rounded_rectangle([0, 0, w, h], 18, fill=DARK)
    d.rounded_rectangle([18, 70, w - 18, h - 18], 10, fill=(30, 30, 34))       # inside
    d.text((w * 0.32, h * 0.55), "$ $", font=font(46, "title"), fill=(90, 160, 90))
    # door (perspective squash as it opens)
    dw = (w - 36) * (1 - 0.85 * door)
    skew = 30 * door
    pts = [(18, 70), (18 + dw, 70 - skew), (18 + dw, h - 18 + skew), (18, h - 18)]
    if door < 0.98:
        d.polygon(pts, fill=(88, 92, 100))
        cx, cy = 18 + dw * 0.55, (70 + h - 18) / 2
        rr = 34 * (1 - 0.7 * door)
        d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=(170, 172, 178), outline=INK, width=3)
        d.line([(cx, cy), (cx + rr * 0.8, cy - rr * 0.3)], fill=INK, width=5)
    # LEDs on the top strip
    r = max(10, w * 0.048)
    for i in range(3):
        x = w * 0.15 + i * w * 0.165
        on = i < leds
        d.ellipse([x - r, 38 - r, x + r, 38 + r], fill=YELLOW if on else (80, 80, 80), outline=INK, width=2)
    sc = {"green": GREEN, "red": RED}.get(status, (80, 80, 80))
    d.ellipse([w * 0.84 - r, 38 - r, w * 0.84 + r, 38 + r], fill=sc, outline=INK, width=2)
    return img


# ---------------------------------------------------------------------------
# Scenes  (each draws onto `f` at `sec` seconds into the scene, `t` = 0..1)
# ---------------------------------------------------------------------------
def s_title(f, sec, t, D):
    title = label("THE IMPENETRABLE SAFE", 76, YELLOW, kind="title", pad=(34, 18), key="title")
    sub = label("how my neural net counts fingers", 38, PAPER, key="sub")
    place(f, title, W / 2, lerp(-120, 200, seg(sec, 0.2, 1.2)), "title", rot=-2)
    place(f, sub, W / 2, lerp(H + 80, 315, seg(sec, 1.0, 1.0)), "sub", rot=1.5)
    place(f, safe_img(leds=int(min(3, max(0, (sec - 3) * 1.2)))), lerp(-300, 420, seg(sec, 1.6, 1.2)), 520, "safe")
    n = [1, 3, 5][int((sec * 0.8) % 3)] if sec > 3 else 1
    place(f, hand(n, 260), lerp(W + 200, 880, seg(sec, 2.0, 1.2)), 520, f"thand", rot=8)


def s_pixels(f, sec, t, D):
    cam = camera_view(3)
    place(f, label("webcam view", 26, key="lbl-cam"), 230, 160, "lc")
    place(f, cam, 230, 380, "cam", alpha=seg(sec, 0.3, 0.6))
    arrow(f, 400, 380, 480, 380, t=seg(sec, 2.0, 0.8), key="a1")
    _, big = pixelated(3)
    place(f, label("112 × 64 pixels", 26, key="lbl-64"), 640, 160, "l64", alpha=seg(sec, 3.0, 0.5))
    place(f, big, 640, 380, "pix", alpha=seg(sec, 2.6, 1.0))
    arrow(f, 810, 380, 890, 380, t=seg(sec, 6.0, 0.8), key="a2")
    small, _ = pixelated(3)
    arr = np.asarray(small)
    r0, c0 = 14, 48   # a patch on the edge of a finger
    reveal = int(lerp(0, 16, seg(sec, 6.5, 5.0)))
    vals, cols = [], []
    for r in range(4):
        vr, cr = [], []
        for c in range(4):
            k = r * 4 + c
            p = arr[r0 + r, c0 + c]
            cr.append(tuple(int(v) for v in p))
            vr.append(f"{int(p[0])}\n{int(p[1])}\n{int(p[2])}" if k < reveal else "")
        vals.append(vr)
        cols.append(cr)
    g = Image.new("RGBA", (4 * 74 + 12, 4 * 74 + 12))
    grid = number_grid([[""] * 4] * 4, 74, cols, key="pgrid")
    d = ImageDraw.Draw(grid)
    for r in range(4):
        for c in range(4):
            if vals[r][c]:
                x0, y0 = 6 + c * 74, 6 + r * 74
                d.rectangle([x0 + 8, y0 + 6, x0 + 66, y0 + 68], fill=(255, 255, 255, 230))
                d.multiline_text((x0 + 37, y0 + 37), vals[r][c], font=font(15, "mono"), fill=INK,
                                 anchor="mm", align="center", spacing=1)
    place(f, label("zoomed in: R, G, B", 26, key="lbl-z"), 1050, 160, "lz", alpha=seg(sec, 6.3, 0.5))
    place(f, grid, 1050, 380, "grid", alpha=seg(sec, 6.3, 0.6))
    place(f, label("112 × 64 × 3 = 21,504 numbers in", 34, YELLOW, key="lbl-n"), W / 2,
          lerp(H + 60, 630, seg(sec, D - 6, 1.0)), "ln", rot=-1)


# convolution demo data: a bright finger stripe on a dark background
CONV_IN = [[1, 2, 8, 9, 8, 1, 2], [2, 1, 9, 9, 8, 2, 1], [1, 1, 8, 9, 9, 1, 1], [2, 1, 9, 8, 8, 1, 2],
           [1, 2, 8, 9, 8, 2, 1], [1, 1, 9, 9, 8, 1, 1], [2, 1, 8, 9, 9, 1, 2]]
KERNEL = [[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]]


def conv_out():
    out = []
    for r in range(5):
        row = []
        for c in range(5):
            row.append(sum(CONV_IN[r + i][c + j] * KERNEL[i][j] for i in range(3) for j in range(3)))
        out.append(row)
    return out


CONV_OUT = conv_out()


def shade(v, lo=0, hi=9):
    k = (v - lo) / (hi - lo)
    g = int(lerp(70, 245, max(0, min(1, k))))
    return (g, int(g * 0.93), int(g * 0.85))


def s_conv(f, sec, t, D):
    cell = 58
    in_cols = [[shade(v) for v in row] for row in CONV_IN]
    gx, gy = 70 + 6, 170 + 6               # top-left of the input grid cells (card at 70,170)
    sweep = seg(sec, 4.0, D - 9.0)
    pos = min(24, int(sweep * 25))
    started = sec > 4.0
    hl = {}
    if started:
        pr, pc = divmod(pos, 5)
        for i in range(3):
            for j in range(3):
                hl[(pr + i, pc + j)] = ORANGE
    inp = number_grid(CONV_IN, cell, in_cols, key="cin", highlight=hl)
    place(f, label("image patch", 26, key="l-in"), 70 + inp.width / 2, 135, "lin")
    place(f, inp, 70 + inp.width / 2, 170 + inp.height / 2, "cin", jit=0.6)
    # the filter card
    kern = number_grid(KERNEL, cell, [[(250, 200, 140)] * 3] * 3, key="kern", fill=(252, 226, 190))
    if not started:
        kx, ky = 640, 330
    else:
        pr, pc = divmod(pos, 5)
        kx, ky = gx + (pc + 1.5) * cell + 3, gy + (pr + 1.5) * cell + 3
    kx = lerp(640, kx, seg(sec, 3.2, 0.8)) if started else 640
    place(f, label("filter (weights)", 26, ORANGE, key="l-k"), 640, 220, "lk", alpha=1 - seg(sec, 3.2, 0.5))
    place(f, kern, kx, ky, "kern", rot=-2, alpha=0.93)
    # output
    revealed = (pos + 1) if started else 0
    vals = [[(CONV_OUT[r][c] if r * 5 + c < revealed else "") for c in range(5)] for r in range(5)]
    cols = [[(shade(CONV_OUT[r][c], -27, 27) if r * 5 + c < revealed else None) for c in range(5)] for r in range(5)]
    out = number_grid(vals, cell, cols, key="cout")
    place(f, label("feature map", 26, key="l-out"), 960, 165, "lout", alpha=seg(sec, 3.5, 0.5))
    place(f, out, 960, 200 + out.height / 2, "cout", alpha=seg(sec, 3.5, 0.5), jit=0.6)
    if started and pos < 25:
        pr, pc = divmod(pos, 5)
        sx, sy = gx + (pc + 1.5) * cell, gy + (pr + 1.5) * cell
        ox, oy = 960 - out.width / 2 + 6 + (pc + 0.5) * cell, 206 + (pr + 0.5) * cell
        arrow(f, sx + 70, sy, ox - 12, oy, color=ORANGE, width=4, key="ca")
    place(f, label("multiply, then add up", 30, PAPER, key="l-mul"), 640, 630,
          "lmul", alpha=seg(sec, 5, 0.6) * (1 - seg(sec, D - 4.5, 0.5)))
    place(f, label("× 16 filters  →  16 feature maps", 34, YELLOW, key="l-16"), 640,
          lerp(H + 60, 630, seg(sec, D - 4, 0.8)), "l16", rot=-1)


def s_relu(f, sec, t, D):
    cell = 66
    order = [(r, c) for r in range(5) for c in range(5) if CONV_OUT[r][c] < 0]
    k = int(seg(sec, 2.5, 5.0) * (len(order) + 0.999))
    zeroed = set(order[:k])
    vals = [[(0 if (r, c) in zeroed else CONV_OUT[r][c]) for c in range(5)] for r in range(5)]
    cols = [[(shade(vals[r][c], -27, 27)) for c in range(5)] for r in range(5)]
    hl = {p: RED for p in order[k:k + 1]} if k < len(order) else None
    g = number_grid(vals, cell, cols, key="relu", highlight=hl)
    place(f, label("feature map", 26, key="l-fm2"), 380, 150, "lfm2")
    place(f, g, 380, 175 + g.height / 2, "relug", jit=0.6)
    # ReLU graph card
    gc = card(380, 320, PAPER, "graph")
    d = ImageDraw.Draw(gc)
    ox, oy = 190, 220
    d.line([(30, oy), (350, oy)], fill=GREY, width=3)
    d.line([(ox, 300), (ox, 30)], fill=GREY, width=3)
    p = seg(sec, 1.0, 2.5)
    end = lerp(30, 350, p)
    d.line([(30, oy), (min(end, ox), oy)], fill=RED, width=7)
    if end > ox:
        d.line([(ox, oy), (end, oy - (end - ox) * 1.0)], fill=RED, width=7)
    d.text((40, 30), "ReLU(z) = max(0, z)", font=font(24), fill=INK)
    d.text((40, 232), "negative -> 0", font=font(20), fill=GREY)
    place(f, gc, 930, 340, "graph", rot=2)
    place(f, label("the same bend that fit the V!", 32, YELLOW, key="l-v"), 930,
          lerp(H + 60, 600, seg(sec, D - 7, 0.8)), "lv", rot=-1.5)


POOL_IN = [[0, 3, 21, 4], [6, 2, 18, 24], [0, 0, 5, 1], [9, 4, 0, 2]]


def s_pool(f, sec, t, D):
    cell = 92
    blocks = [(0, 0), (0, 2), (2, 0), (2, 2)]
    colors = [YELLOW, (140, 200, 240), (170, 220, 150), (240, 170, 200)]
    k = int(seg(sec, 2.0, 6.0) * 4.999)
    cols = [[None] * 4 for _ in range(4)]
    for b in range(min(k, 4)):
        r0, c0 = blocks[b]
        for i in range(2):
            for j in range(2):
                cols[r0 + i][c0 + j] = colors[b]
    g = number_grid(POOL_IN, cell, cols, key="pin", fsize=32)
    place(f, label("4 × 4", 28, key="l-44"), 330, 135, "l44")
    place(f, g, 330, 170 + g.height / 2, "pin", jit=0.6)
    outv = [["", ""], ["", ""]]
    outc = [[None, None], [None, None]]
    for b in range(min(k, 4)):
        r0, c0 = blocks[b]
        mx = max(POOL_IN[r0 + i][c0 + j] for i in range(2) for j in range(2))
        outv[r0 // 2][c0 // 2] = mx
        outc[r0 // 2][c0 // 2] = colors[b]
    o = number_grid(outv, cell, outc, key="pout", fsize=36)
    arrow(f, 560, 355, 760, 355, t=seg(sec, 1.5, 0.8), key="pa")
    place(f, label("keep the biggest", 26, key="l-kb"), 660, 300, "lkb", alpha=seg(sec, 1.5, 0.6))
    place(f, label("2 × 2", 28, key="l-22"), 900, 235, "l22")
    place(f, o, 900, 270 + o.height / 2, "pout", jit=0.6)
    place(f, label("112 × 64  →  56 × 32", 34, YELLOW, key="l-shrink"), 640,
          lerp(H + 60, 640, seg(sec, D - 6, 0.8)), "lsh", rot=1)


def stack(n_cards, side, color, key, show=1.0, aspect=1.75):
    """A pile of feature maps. side = height; maps are `aspect` times wider than tall."""
    off = 7
    w = int(side * aspect)
    k = max(1, int(round(n_cards * show)))
    img = Image.new("RGBA", (w + off * (k - 1) + 10, side + off * (k - 1) + 10), (0, 0, 0, 0))
    for i in range(k):
        c = card(w, side, color, f"{key}-{i}")
        d = ImageDraw.Draw(c)
        d.rectangle([4, 4, w - 5, side - 5], outline=tuple(int(v * 0.8) for v in color), width=2)
        img.alpha_composite(c, (off * (k - 1 - i), off * (k - 1 - i)))
    return img


def s_three(f, sec, t, D):
    stages = [("112×64×3", 0, 3, (200, 180, 150)), ("56×32×16", 110, 6, (150, 190, 230)),
              ("28×16×32", 80, 9, (170, 215, 160)), ("14×8×64", 54, 12, (240, 190, 120))]
    xs = [150, 450, 765, 1065]
    _, big = pixelated(3, size=230)
    halves = [big.width / 2] + [(int(sd * 1.75) + 7 * (n - 1) + 10) / 2 for _, sd, n, _ in stages[1:]]
    for i, (txt, side, n, col) in enumerate(stages):
        a = seg(sec, 0.6 + i * 2.2, 0.8)
        if a <= 0:
            continue
        if i == 0:
            img = big
        else:
            img = stack(n, side, col, f"st{i}", show=a)
        place(f, img, xs[i], 330, f"stk{i}", alpha=a, rot=[-2, 1, -1, 2][i])
        place(f, label(txt, 26, key=f"l-st{i}"), xs[i], 500, f"lst{i}", alpha=a)
        if i > 0:
            x_from, x_to = xs[i - 1] + halves[i - 1] + 8, xs[i] - halves[i] - 8
            arrow(f, x_from, 330, x_to, 330, t=a, key=f"sa{i}")
            place(f, label("conv → ReLU → pool", 18, PAPER, key=f"l-crp{i}"),
                  (x_from + x_to) / 2, 190, f"lcrp{i}", alpha=a)
    notes = [("edges", 470), ("curves", 760), ("fingertips & gaps", 1040)]
    for j, (txt, x) in enumerate(notes):
        place(f, label(txt, 28, YELLOW, key=f"l-note{j}"), x, 600, f"note{j}",
              alpha=seg(sec, D - 9 + j * 1.6, 0.6), rot=[-2, 1.5, -1][j])


def s_linear(f, sec, t, D):
    # flatten: 8x8 map cards turn into a long strip
    p = seg(sec, 1.0, 3.0)
    strip_y = 230
    ncells = 64
    cw = 16
    x0 = W / 2 - ncells * cw / 2
    for i in range(6):
        sx, sy = lerp(200 + i * 14, x0 + i * (ncells * cw / 6) + 40, p), lerp(300 + i * 14, strip_y, p)
        side = lerp(90, 12, p)
        if p < 0.98:
            place(f, card(side, side, (240, 190, 120), f"fl{i}"), sx, sy, f"fl{i}", alpha=1 - p * 0.5)
    if p > 0.6:
        strip = card(ncells * cw + 10, 30, PAPER, "strip")
        d = ImageDraw.Draw(strip)
        rnd = random.Random(5)
        for i in range(ncells):
            g = rnd.randint(110, 240)
            d.rectangle([5 + i * cw + 1, 6, 5 + (i + 1) * cw - 1, 24], fill=(g, int(g * 0.85), int(g * 0.6)))
        place(f, strip, W / 2, strip_y, "strip", alpha=seg(sec, 2.8, 0.8), jit=0.5)
        place(f, label("7,168 numbers  (only some drawn)", 24, key="l-4096"), W / 2, strip_y - 55, "l4096",
              alpha=seg(sec, 3.2, 0.6))
    # outputs + strings
    OUT_NAMES = ["0", "1", "2", "3", "4", "5", "blank"]
    outs_x = [W / 2 + (k - 3) * 150 for k in range(7)]
    q = seg(sec, 5.5, 4.0)
    d = ImageDraw.Draw(f)
    if q > 0:
        rnd = random.Random(9)
        for i in range(0, ncells, 3):
            sx = x0 + i * cw + cw / 2
            for k in range(7):
                if rnd.random() < 0.55 and rnd.random() < q:
                    d.line([(sx, strip_y + 18), (outs_x[k], 520)], fill=(110, 90, 70), width=1)
    for k in range(7):
        a = seg(sec, 5.0 + k * 0.25, 0.5)
        place(f, label(OUT_NAMES[k], 44 if k < 6 else 32, PAPER, kind="title", pad=(26, 8), key=f"o{k}"), outs_x[k], 545, f"o{k}", alpha=a)
    place(f, label("7 outputs: 0-5 fingers, or blank", 32, YELLOW, key="l-7"), W / 2,
          lerp(H + 60, 650, seg(sec, D - 5, 0.8)), "l6", rot=-1)


PROBS = [0.01, 0.02, 0.03, 0.92, 0.01, 0.01, 0.00]
BAR_NAMES = ["0", "1", "2", "3", "4", "5", "blank"]


def s_softmax(f, sec, t, D):
    base_y, maxh = 560, 330
    for k in range(7):
        x = 250 + k * 125
        g = seg(sec, 1.0 + k * 0.3, 2.0)
        h = max(8, PROBS[k] * maxh * g)
        col = YELLOW if k == 3 else (190, 200, 210)
        bar = card(80, h, col, f"bar{k}")
        place(f, bar, x, base_y - h / 2, f"bar{k}", jit=0.5)
        place(f, label(BAR_NAMES[k], 36 if k < 6 else 26, PAPER, kind="title", key=f"bk{k}"), x, base_y + 45, f"bk{k}")
        place(f, text_img(f"{int(round(PROBS[k] * 100 * g))}%", 28), x, base_y - h - 28, f"bp{k}", shadow=False)
    place(f, label("adds up to 100%", 26, key="l-100"), 1120, 250, "l100", alpha=seg(sec, 4.0, 0.6), rot=3)
    place(f, label("3 fingers  —  92% sure", 46, YELLOW, kind="title", key="l-ans"), W / 2,
          lerp(-80, 150, seg(sec, D - 7, 0.8)), "lans", rot=-1.5)


def s_training(f, sec, t, D):
    # photo -> network -> guess
    place(f, camera_view(3, 200), 150, 300, "tphoto")
    place(f, label("label: 3", 26, YELLOW, key="l-lab"), 150, 430, "llab")
    layers_x = [380, 500, 620, 740]
    for i, x in enumerate(layers_x):
        place(f, card(70, 220 - i * 30, [(150, 190, 230), (170, 215, 160), (240, 190, 120), PAPER][i], f"nl{i}"),
              x, 300, f"nl{i}")
        if i:
            arrow(f, layers_x[i - 1] + 38, 300, x - 38, 300, width=4, key=f"fa{i}", t=seg(sec, 0.8 + i * 0.3, 0.4))
    arrow(f, 230, 300, 340, 300, width=4, key="fa0", t=seg(sec, 0.6, 0.4))
    wrong = sec < D * 0.62
    guess = "guess: 5  ✗" if wrong else "guess: 3  ✓"
    place(f, label(guess, 30, (250, 200, 200) if wrong else (200, 240, 200), key=f"g{wrong}"), 950, 300, "guess",
          alpha=seg(sec, 2.0, 0.5))
    arrow(f, 778, 300, 820, 300, width=4, key="fa5", t=seg(sec, 1.8, 0.4))
    # loss + backprop
    if sec > 4:
        loss_val = lerp(2.3, 0.05, seg(sec, D * 0.55, D * 0.3))
        place(f, label(f"loss = {loss_val:.2f}", 30, (255, 220, 200), key=f"loss{loss_val:.2f}"), 950, 380, "lossl",
              alpha=seg(sec, 4, 0.5))
        bp = seg(sec, 7, 4)
        if bp > 0 and wrong:
            arrow(f, 860, 450, lerp(860, 360, bp), 450, color=RED, width=7, key="bp")
            place(f, label("backpropagation (chain rule)", 24, (255, 220, 200), key="l-bp"), 610, 495,
                  "lbp", alpha=seg(sec, 7.5, 0.5))
            for i, x in enumerate(layers_x):
                if lerp(860, 360, bp) < x:
                    nudge = math.sin(HOLD[0] * 1.3 + i) * 6
                    place(f, text_img("±", 34, RED), x + nudge, 175, f"nudge{i}", shadow=False)
    # loss curve card
    if sec > D * 0.45:
        gc = card(360, 220, PAPER, "losscurve")
        d = ImageDraw.Draw(gc)
        d.line([(30, 190), (340, 190)], fill=GREY, width=2)
        d.line([(30, 190), (30, 20)], fill=GREY, width=2)
        p = seg(sec, D * 0.45, D * 0.4)
        pts = [(30 + 300 * (i / 60), 190 - 160 * math.exp(-i / 12)) for i in range(int(61 * p))]
        if len(pts) > 1:
            d.line(pts, fill=RED, width=4)
        d.text((150, 196 - 26), "epochs", font=font(18), fill=GREY)
        d.text((40, 22), "loss", font=font(18), fill=GREY)
        place(f, gc, 1080, 560, "lcurve", rot=2, alpha=seg(sec, D * 0.45, 0.6))
    place(f, label("≈ 75,000 weights, all nudged downhill", 30, YELLOW, key="l-49"), 470,
          lerp(H + 60, 640, seg(sec, D - 8, 0.8)), "l49", rot=-1)


def s_augment(f, sec, t, D):
    base = camera_view(3, 220)
    place(f, base, W / 2, 360, "aug0")
    variants = [
        ("rotated", base.rotate(10, resample=Image.BICUBIC, expand=False, fillcolor=(55, 58, 64, 255)), 220, 200, -6),
        ("shifted", base.transform(base.size, Image.AFFINE, (1, 0, 25, 0, 1, -15), fillcolor=(55, 58, 64, 255)), 1060, 200, 5),
        ("flipped", base.transpose(Image.FLIP_LEFT_RIGHT), 220, 540, 4),
        ("brighter", Image.eval(base, lambda v: min(255, int(v * 1.35))), 1060, 540, -4),
        ("darker", Image.eval(base, lambda v: int(v * 0.6)), 640, 600, 2),
    ]
    for i, (name, img, x, y, rot) in enumerate(variants):
        a = seg(sec, 1.2 + i * 1.3, 0.9)
        if a <= 0:
            continue
        img = img.copy()
        img.putalpha(255)
        sm = img.resize((230, int(230 * img.height / img.width)))
        px, py = lerp(W / 2, x, a), lerp(360, y, a)
        place(f, sm, px, py, f"aug{i + 1}", rot=rot, alpha=a)
        place(f, label(name, 22, key=f"l-aug{i}"), px, py + sm.height / 2 + 28, f"laug{i}", alpha=a)


def s_lock(f, sec, t, D):
    sec = sec * 16.0 / D     # the sequence below is laid out for 16 s; stretch it to the narration
    # timeline: button 1.5s, digits at fixed slots, open at the end
    slots = [(3.0, 1), (6.5, 3), (10.0, 5)]
    leds = 0
    showing, hold = None, 0.0
    for start, dgt in slots:
        if start <= sec < start + 2.2:
            showing = dgt
            hold = min(1.0, (sec - start) / 1.2)
        if sec >= start + 1.2:
            leds += 1
    open_t = seg(sec, 13.0, 2.0)
    status = "green" if sec > 12.6 else None
    lights = sec > 1.5
    s = safe_img(leds=leds if lights else 0, status=status, door=open_t, size=360)
    place(f, s, 420, 390, "bigsafe")
    if 0.8 < sec < 2.0:
        place(f, label("button!", 30, YELLOW, key="l-btn"), 420, 140, "lbtn", rot=-3)
    if lights:
        place(f, label("lights on", 24, key="l-lon"), 230, 150, "llon", alpha=seg(sec, 1.5, 0.5) * (1 - seg(sec, 3, 0.5)))
    if showing:
        place(f, hand(showing, 280), 930, 380, f"lh{showing}", rot=4)
        bar = card(240, 26, PAPER, "holdbg")
        d = ImageDraw.Draw(bar)
        d.rectangle([4, 4, 4 + 232 * hold, 22], fill=YELLOW if hold < 1 else GREEN)
        place(f, bar, 930, 620, "holdbar", jit=0.4)
        place(f, label("hold for 1 second" if hold < 1 else "got it!", 24, key=f"l-hold{hold >= 1}"), 930, 665,
              "lhold")
    if sec > 13.2:
        place(f, label("UNLOCKED!", 60, (200, 240, 200), kind="title", key="l-unl"), 930, 380, "unl", rot=-4,
              alpha=seg(sec, 13.2, 0.4))


def s_end(f, sec, t, D):
    place(f, label("Thanks for watching!", 70, YELLOW, kind="title", pad=(34, 18), key="thx"), W / 2,
          lerp(-100, 250, seg(sec, 0.3, 1.0)), "thx", rot=-2)
    place(f, label("voiceover: Mia Solomon   ·   animation: Python + Claude", 26, key="cred"), W / 2, 420,
          "cred", alpha=seg(sec, 1.5, 0.8))
    place(f, hand(5, 220), 330, 560, "bye1", rot=-10 + 8 * math.sin(sec * 3))
    place(f, safe_img(leds=3, status="green", door=1.0, size=220), 950, 560, "byesafe")


DRAW = [s_title, s_pixels, s_conv, s_relu, s_pool, s_three, s_linear, s_softmax,
        s_training, s_augment, s_lock, s_end]


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def guide_overlay(frame, i, sec, D):
    """Narration at the bottom; words turn dark as you should be reading them."""
    words = SCENES[i][2].split()
    speak = max(1.0, D - 2.0)
    done = int(len(words) * max(0.0, min(1.0, (sec - 0.6) / speak)))
    out = Image.new("RGB", (W, H), (20, 20, 24))
    sw, sh = int(W * 0.74), int(H * 0.74)
    out.paste(frame.resize((sw, sh), Image.BILINEAR), ((W - sw) // 2, 0))
    d = ImageDraw.Draw(out)
    f = font(25)
    x, y = 40, sh + 12
    for k, w in enumerate(words):
        wl = d.textlength(w + " ", font=f)
        if x + wl > W - 40:
            x, y = 40, y + 33
        d.text((x, y), w, font=f, fill=(255, 225, 120) if k < done else (140, 140, 140))
        x += wl
    d.text((W - 150, 12), f"scene {i}", font=font(20, "mono"), fill=(255, 255, 255))
    d.text((W - 150, 40), f"{sec:4.1f}/{D:.0f}s", font=font(20, "mono"), fill=(255, 255, 255))
    return out


def render_frame(i, sec):
    D = DURS[i]
    HOLD[0] = int((sum(DURS[:i]) + sec) * FPS) // 2
    f = background().copy()
    DRAW[i](f, sec, sec / D, D)
    if i not in (0, len(SCENES) - 1):
        scene_tag(f, i, sec)
    return f


def write_script():
    lines = ["# Stop-motion movie script: How my neural net counts fingers", "",
             "Read each scene's lines while its pictures are on screen. Easiest way: play "
             "`neural_net_movie_GUIDE.mp4` and read the highlighted words as they light up.", ""]
    t = 0.0
    for i, (name, title, text) in enumerate(SCENES):
        m0, s0 = divmod(int(t), 60)
        t += DURS[i]
        m1, s1 = divmod(int(t), 60)
        lines += [f"## {i}. {name}: {title}  ({m0}:{s0:02d}–{m1}:{s1:02d}, {DURS[i]:.0f}s)", "", text, ""]
    lines.append(f"**Total: {int(t) // 60}:{int(t) % 60:02d}**")
    (HERE / "script.md").write_text("\n".join(lines))


def main():
    args = sys.argv[1:]
    if args and args[0] == "--still":
        i, frac = int(args[1]), float(args[2])
        render_frame(i, DURS[i] * frac).save(HERE / f"still_{i}_{frac}.png")
        guide_overlay(render_frame(i, DURS[i] * frac), i, DURS[i] * frac, DURS[i]).save(HERE / f"still_{i}_{frac}_guide.png")
        return
    only = int(args[1]) if args and args[0] == "--scene" else None
    write_script()
    suffix = f"_scene{only}" if only is not None else ""
    writers = []
    for name in (f"neural_net_movie{suffix}.mp4", f"neural_net_movie_GUIDE{suffix}.mp4"):
        w = imageio_ffmpeg.write_frames(str(HERE / name), (W, H), fps=FPS, codec="libx264",
                                        pix_fmt_out="yuv420p", quality=8, macro_block_size=8)
        w.send(None)
        writers.append(w)
    total = 0
    for i in range(len(SCENES)):
        if only is not None and i != only:
            continue
        n = int(round(DURS[i] * FPS))
        for k in range(n):
            sec = k / FPS
            fr = render_frame(i, sec)
            writers[0].send(fr.tobytes())
            writers[1].send(guide_overlay(fr, i, sec, DURS[i]).tobytes())
            total += 1
        print(f"scene {i:2d} {SCENES[i][0]:<13} {DURS[i]:5.1f}s  done", flush=True)
    for w in writers:
        w.close()
    print(f"{total} frames, {total / FPS:.0f} seconds")


if __name__ == "__main__":
    main()
