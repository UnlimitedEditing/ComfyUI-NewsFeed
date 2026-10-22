"""Shot lists, Kokoro TTS and the ffmpeg compile (no ComfyUI imports besides folder_paths, tested offline)."""
import json
import os
import re
import shutil
import subprocess
import tempfile
import urllib.request
import wave
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from . import core

FPS = 24
PAD = 0.35            # silence after every shot's narration
W, H = 1280, 720

KOKORO_BASE = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/"
KOKORO_FILES = ("kokoro-v1.0.onnx", "voices-v1.0.bin")


# ── shot lists ────────────────────────────────────────────────────────────────

def shot_lists(result):
    """result: dict from core.finish_script. Returns (prompts, seeds, lines, plan)."""
    if not result.get("ok") or not result.get("shots"):
        raise RuntimeError(f"script step failed: {result.get('error') or result.get('warnings')}")
    shots = result["shots"]
    prompts = [s["image_prompt"] for s in shots if s["media_id"] is None]
    seeds = [s["seed"] for s in shots if s["media_id"] is None]
    spare = False
    if not prompts:  # an empty list would skip every downstream node
        prompts = [result["story"]["visual_prefix"] + result["story"]["title"]]
        seeds = [result["story"]["seed"]]
        spare = True
    lines = [s["line"] or "." for s in shots]
    plan = dict(result, spare_image=spare)
    return prompts, seeds, lines, plan


# ── Kokoro ────────────────────────────────────────────────────────────────────

_KOKORO = {}


def _kokoro_dir():
    import folder_paths
    return os.path.join(folder_paths.models_dir, "kokoro")


def kokoro_synth(text, voice, speed, lang="en-us"):
    """Returns (float32 mono samples, sample_rate). Model files come from concept_mapping; the
    GitHub-release fallback only runs if they were not pre-staged."""
    d = _kokoro_dir()
    os.makedirs(d, exist_ok=True)
    for name in KOKORO_FILES:
        p = os.path.join(d, name)
        if not os.path.isfile(p):
            urllib.request.urlretrieve(KOKORO_BASE + name, p)
    if "k" not in _KOKORO:
        from kokoro_onnx import Kokoro
        _KOKORO["k"] = Kokoro(os.path.join(d, KOKORO_FILES[0]), os.path.join(d, KOKORO_FILES[1]))
    k = _KOKORO["k"]
    if voice not in k.get_voices():
        voice = "af_heart" if "af_heart" in k.get_voices() else k.get_voices()[0]
    samples, sr = k.create(text, voice=voice, speed=float(speed), lang=lang)
    return np.asarray(samples, dtype=np.float32).reshape(-1), int(sr)


# ── captions (PIL overlays, no libass/fontconfig dependency) ──────────────────

def _font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _wrap(draw, text, font, max_w):
    lines, cur = [], ""
    for w in text.split():
        t = (cur + " " + w).strip()
        if draw.textlength(t, font=font) <= max_w or not cur:
            cur = t
        else:
            lines.append(cur)
            cur = w
    if cur:
        lines.append(cur)
    return lines


def caption_png(text, path, size=46, max_w=1100, bold=False):
    font = _font(size)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines = _wrap(probe, text, font, max_w)
    lh = int(size * 1.25)
    stroke = max(2, size // 12) if not bold else max(3, size // 9)
    img = Image.new("RGBA", (max_w + 2 * stroke + 8, lh * len(lines) + 2 * stroke + 8), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    for i, ln in enumerate(lines):
        tw = d.textlength(ln, font=font)
        d.text(((img.width - tw) / 2, 4 + i * lh), ln, font=font, fill=(255, 255, 255, 255),
               stroke_width=stroke, stroke_fill=(0, 0, 0, 255))
    img.save(path)
    return img.size


def lower_third_png(text, path, size=28, max_w=560):
    font = _font(size)
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines = _wrap(probe, text, font, max_w)[:3]
    lh = int(size * 1.3)
    pad = 14
    img = Image.new("RGBA", (max_w + 2 * pad, lh * len(lines) + 2 * pad), (10, 10, 10, 190))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, 6, img.height], fill=(240, 190, 40, 255))
    for i, ln in enumerate(lines):
        d.text((pad + 6, pad + i * lh), ln, font=font, fill=(255, 255, 255, 255))
    img.save(path)
    return img.size


def chunk_line(line, mode):
    words = line.split()
    if mode in (2, 3):
        n = 3
    else:
        n = 8
    chunks, cur = [], []
    for w in words:
        cur.append(w)
        if len(cur) >= n or re.search(r"[.!?;:]$", w) and len(cur) >= 3:
            chunks.append(" ".join(cur))
            cur = []
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [line]


def chunk_windows(line, mode, dur):
    chunks = chunk_line(line, mode)
    total = sum(len(c) for c in chunks) or 1
    out, t = [], 0.0
    for c in chunks:
        span = dur * len(c) / total
        out.append((c, t, t + span))
        t += span
    return out


# ── image helpers ─────────────────────────────────────────────────────────────

def cover_crop(img, w, h):
    iw, ih = img.size
    s = max(w / iw, h / ih)
    img = img.resize((max(w, round(iw * s)), max(h, round(ih * s))), Image.LANCZOS)
    x, y = (img.width - w) // 2, (img.height - h) // 2
    return img.crop((x, y, x + w, y + h))


def download_image(url, path):
    req = urllib.request.Request(url, headers={**core.UA, "Referer": url.split("/ace/")[0] + "/"})
    data = urllib.request.urlopen(req, timeout=25).read()
    with open(path, "wb") as f:
        f.write(data)
    Image.open(path).verify()
    return path


def title_card(title, path):
    img = Image.new("RGB", (2560, 1440), (18, 22, 30))
    d = ImageDraw.Draw(img)
    font = _font(120)
    y = 500
    for ln in _wrap(d, title, font, 2000):
        tw = d.textlength(ln, font=font)
        d.text(((2560 - tw) / 2, y), ln, font=font, fill=(235, 235, 235))
        y += 150
    img.save(path)


def zoompan(camera, n):
    c = "(iw-iw/zoom)/2"
    cy = "(ih-ih/zoom)/2"
    z, x, y = "1.12", c, cy
    if camera == "slow push-in":
        z = f"min(1.0+0.14*on/{n},1.14)"
    elif camera == "slow pull-out":
        z = f"max(1.14-0.14*on/{n},1.0)"
    elif camera == "pan left":
        x = f"(iw-iw/zoom)*(1-on/{n})"
    elif camera == "pan right":
        x = f"(iw-iw/zoom)*on/{n}"
    elif camera == "tilt up":
        y = f"(ih-ih/zoom)*(1-on/{n})"
    else:
        z = f"1.03+0.03*on/{n}"
    return f"zoompan=z='{z}':x='{x}':y='{y}':d={n}:s={W}x{H}:fps={FPS}"


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"ffmpeg failed: {' '.join(cmd[:6])} ... {r.stderr[-600:]}")


# ── compile ───────────────────────────────────────────────────────────────────

def compile_video(plan, gen_images, audios, out_path, work):
    """plan: dict from shot_lists; gen_images: list of HxWx3 uint8 arrays (shots without article
    media, in order); audios: list of (samples, sr) per shot."""
    shots = plan["shots"]
    mode_name = plan["story"]["caption_mode"]
    mode = int(plan["story"].get("caption", 1))
    report, seg_inputs = [], []
    gi = 0
    sr_out = audios[0][1] if audios else 24000

    # decide each shot's picture
    pics = []
    for i, s in enumerate(shots):
        path = os.path.join(work, f"pic{i}.png")
        source, lt = "generated", None
        if s["media_id"] and s.get("media"):
            try:
                raw = download_image(s["media"]["url"], os.path.join(work, f"raw{i}"))
                cover_crop(Image.open(raw).convert("RGB"), 2560, 1440).save(path)
                source = f"article {s['media_id']}"
                lt = (s["media"].get("caption") or s["media"].get("alt") or "")[:140] or None
            except Exception as e:
                report.append(f"shot {i+1}: article image failed ({e}); using fallback")
                s = dict(s, media_id=None)
        if source == "generated":
            if gi < len(gen_images):
                cover_crop(Image.fromarray(gen_images[gi]).convert("RGB"), 2560, 1440).save(path)
                gi += 1
            else:
                title_card(plan["story"]["title"], path)
                source = "title card"
        pics.append((path, source, lt))

    def render(i):
        s = shots[i]
        samples, sr = audios[i] if i < len(audios) else (np.zeros(int(sr_out * 2), np.float32), sr_out)
        dur = len(samples) / sr + PAD
        n = max(2, round(dur * FPS))
        path, source, lt = pics[i]
        seg = os.path.join(work, f"seg{i}.mp4")
        cmd = ["ffmpeg", "-y", "-v", "error", "-i", path]
        filt = [f"[0:v]{zoompan(s['camera'], n)},format=yuv420p[v0]"]
        last, k = "v0", 1
        if lt:
            p = os.path.join(work, f"lt{i}.png")
            w_, h_ = lower_third_png(lt, p)
            cmd += ["-i", p]
            filt.append(f"[{last}][{k}:v]overlay=x=40:y=40:format=auto[v{k}]")
            last, k = f"v{k}", k + 1
        if mode:
            big = mode == 3
            for j, (txt, a, b) in enumerate(chunk_windows(s["line"], mode, dur - PAD)):
                p = os.path.join(work, f"cap{i}_{j}.png")
                w_, h_ = caption_png(txt, p, size=72 if big else 46, bold=big)
                cmd += ["-i", p]
                y = "(main_h-overlay_h)/2" if big else "main_h-overlay_h-70"
                filt.append(f"[{last}][{k}:v]overlay=x=(main_w-overlay_w)/2:y={y}:"
                            f"enable='between(t,{a:.2f},{b:.2f})':format=auto[v{k}]")
                last, k = f"v{k}", k + 1
        cmd += ["-filter_complex", ";".join(filt), "-map", f"[{last}]", "-t", f"{dur:.3f}", "-r", str(FPS),
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "22", "-pix_fmt", "yuv420p", "-an", seg]
        run(cmd)
        return seg, dur, source

    with ThreadPoolExecutor(max_workers=3) as ex:
        results = list(ex.map(render, range(len(shots))))

    lst = os.path.join(work, "list.txt")
    with open(lst, "w") as f:
        for seg, _, _ in results:
            f.write(f"file '{seg.replace(chr(92), '/')}'\n")
    silent = os.path.join(work, "silent.mp4")
    run(["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", lst, "-c", "copy", silent])

    chunks = []
    for i, (seg, dur, src) in enumerate(results):
        samples = audios[i][0] if i < len(audios) else np.zeros(0, np.float32)
        pad = np.zeros(max(0, int(round(dur * sr_out)) - len(samples)), np.float32)
        chunks.append(np.concatenate([samples, pad]))
    pcm = (np.clip(np.concatenate(chunks), -1, 1) * 32767).astype(np.int16)
    wav = os.path.join(work, "narration.wav")
    with wave.open(wav, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr_out)
        wf.writeframes(pcm.tobytes())
    run(["ffmpeg", "-y", "-v", "error", "-i", silent, "-i", wav, "-map", "0:v:0", "-map", "1:a:0",
         "-c:v", "copy", "-c:a", "aac", "-b:a", "160k", "-shortest", out_path])

    total = sum(d for _, d, _ in results)
    info = {"seconds": round(total, 2), "shots": [{"i": i + 1, "seconds": round(d, 2), "picture": src,
            "line": shots[i]["line"][:80]} for i, (_, d, src) in enumerate(results)], "notes": report,
            "captions": mode_name}
    return info
