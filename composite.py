"""Cut characters out of flat-background pictures and composite them into per-shot plates (PIL + numpy only)."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

TOL = 64.0          # colour distance (0-441) from the border colour that still counts as background


def cutout(img):
    """img: RGB PIL image whose background is one flat colour. Returns an RGBA image cropped to the figure.
    The background is the region connected to the border whose colour is near the border's median colour."""
    rgb = np.asarray(img.convert("RGB"), dtype=np.float32)
    h, w = rgb.shape[:2]
    band = np.concatenate([rgb[:6].reshape(-1, 3), rgb[-6:].reshape(-1, 3),
                           rgb[:, :6].reshape(-1, 3), rgb[:, -6:].reshape(-1, 3)])
    bg = np.median(band, axis=0)
    cand = (np.linalg.norm(rgb - bg, axis=2) < TOL).astype(np.uint8) * 255
    flood = Image.fromarray(cand, "L").copy()
    seeds = [(x, 0) for x in range(0, w, max(1, w // 24))] + [(x, h - 1) for x in range(0, w, max(1, w // 24))] + \
            [(0, y) for y in range(0, h, max(1, h // 24))] + [(w - 1, y) for y in range(0, h, max(1, h // 24))]
    for sx, sy in seeds:
        if flood.getpixel((sx, sy)) == 255:
            ImageDraw.floodfill(flood, (sx, sy), 128)
    bgmask = (np.asarray(flood) == 128)
    alpha = Image.fromarray(np.where(bgmask, 0, 255).astype(np.uint8), "L")
    alpha = alpha.filter(ImageFilter.MinFilter(3)).filter(ImageFilter.GaussianBlur(1.0))
    a = np.asarray(alpha, dtype=np.float32) / 255.0
    # despill: pull the background hue out of the edge band
    edge = (a > 0.02) & (a < 0.98)
    if edge.any():
        ch = int(np.argmax(bg))
        others = [c for c in range(3) if c != ch]
        lim = np.maximum(rgb[..., others[0]], rgb[..., others[1]])
        rgb[..., ch] = np.where(edge, np.minimum(rgb[..., ch], lim), rgb[..., ch])
    out = np.dstack([rgb.clip(0, 255), a * 255.0]).astype(np.uint8)
    im = Image.fromarray(out, "RGBA")
    box = im.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox()
    return im.crop(box) if box else im


def framing_of(action):
    f = (action or "").lower()
    return "close" if "close" in f else "wide" if "wide" in f else "medium"


def place_cast(plate, figs, framing, idx=0):
    """Composite cutouts (list of RGBA) into plate (RGB PIL, already gen size). Returns RGB PIL."""
    W, H = plate.size
    canvas = plate.convert("RGBA")
    n = len(figs)
    if not n:
        return plate.convert("RGB")
    centres = {1: [0.45 + 0.10 * (idx % 2)], 2: [0.32, 0.68], 3: [0.22, 0.5, 0.78]}[min(n, 3)]
    target_h = {"close": 1.45, "medium": 0.86, "wide": 0.42}[framing] * H
    if n > 1 and framing == "close":
        target_h = 1.15 * H
    mean = np.asarray(plate.convert("RGB").resize((8, 8)), dtype=np.float32).reshape(-1, 3).mean(axis=0)
    for fig, c in zip(figs, centres):
        hh = int(target_h)
        ww = max(1, int(fig.width * hh / fig.height))
        sized = fig.resize((ww, hh), Image.LANCZOS)
        arr = np.asarray(sized, dtype=np.float32)
        arr[..., :3] = arr[..., :3] * 0.85 + (mean * (arr[..., :3] / 255.0 * 0.5 + 0.5)) * 0.15   # take on a little plate colour
        sized = Image.fromarray(arr.clip(0, 255).astype(np.uint8), "RGBA")
        x = int(c * W - ww / 2)
        y = int((0.06 * H) if framing == "close" else (0.90 * H - hh))
        if framing != "close":
            sh = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
            d = ImageDraw.Draw(sh)
            d.ellipse([x + ww * 0.12, y + hh - hh * 0.04, x + ww * 0.88, y + hh + hh * 0.05], fill=(0, 0, 0, 120))
            canvas = Image.alpha_composite(canvas, sh.filter(ImageFilter.GaussianBlur(max(2, ww // 30))))
        layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        layer.paste(sized, (x, y), sized)
        canvas = Image.alpha_composite(canvas, layer)
    return canvas.convert("RGB")
