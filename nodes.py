import json
import math

import numpy as np
import torch

from . import core, video, qwentts, script as scriptlib


def _data_image(text):
    payload = core.encode_string_bytes(text)
    side = max(1, math.ceil(math.sqrt(len(payload) // 3)))
    padded = payload + b"\x00" * (side * side * 3 - len(payload))
    arr = np.frombuffer(padded, dtype=np.uint8).reshape(side, side, 3).astype(np.float32) / 255.0
    return torch.from_numpy(arr).unsqueeze(0)


class NewsPickStory:
    """Fetch live RSS, pick a story by seed, download its full text + images, and build the LLM brief."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "category":     ("STRING", {"default": "any"}),
            "pick_seed":    ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            "tone":         ("INT", {"default": 0, "min": 0, "max": 5}),
            "visual":       ("INT", {"default": 0, "min": 0, "max": 4}),
            "length":       ("INT", {"default": 48, "min": 15, "max": 120}),
            "shot_seconds": ("INT", {"default": 6, "min": 3, "max": 15}),
            "caption":      ("INT", {"default": 1, "min": 0, "max": 3}),
            "format":       ("INT", {"default": 0, "min": 0, "max": 2}),
            "query":        ("STRING", {"default": ""}),
        },
            # optional so older workflow JSONs (which do not send it) keep working
            "optional": {"max_shots": ("INT", {"default": 8, "min": 2, "max": 8}),
                        "energy": ("INT", {"default": 1, "min": 0, "max": 2})}}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("system_prompt", "user_prompt", "story_json")
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    def run(self, category, pick_seed, tone, visual, length, shot_seconds, caption, format, query, max_shots=8, energy=1):
        print(f"[NewsFeed] inputs category={category!r} seed={pick_seed} tone={tone} visual={visual} "
              f"length={length} shot_seconds={shot_seconds} caption={caption} format={format} query={query!r}")
        item, paras, media = core.pick_story(category, pick_seed, query=query)
        system, user, story = core.build_brief(item, paras, media, pick_seed, tone, visual,
                                               length, shot_seconds, caption, format, max_shots, energy)
        print(f"[NewsFeed] picked [{item['category']}] {item['title']} "
              f"({len(paras)} paras, {len(media)} images)")
        return (system, user, json.dumps(story))


class NewsFinishScript:
    """Validate the LLM's JSON, resolve media ids to image URLs, and emit one result as a data image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "response_text": ("STRING", {"forceInput": True}),
            "story_json":    ("STRING", {"forceInput": True}),
        }}

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("result_json", "result_data")
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    def run(self, response_text, story_json):
        result = core.finish_script(response_text, json.loads(story_json))
        text = json.dumps(result, ensure_ascii=False)
        print(f"[NewsFeed] script ok={result.get('ok')} shots={len(result.get('shots', []))} "
              f"warnings={result.get('warnings')}")
        return (text, _data_image(text))


class NewsShotLists:
    """Turns the finished script into per-shot lists. Downstream nodes run once per list item."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"result_json": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING", "INT", "INT", "STRING")
    RETURN_NAMES = ("image_prompts", "seeds", "lines", "plan_json", "width", "height", "instructs")
    OUTPUT_IS_LIST = (True, True, True, False, False, False, True)
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    def run(self, result_json):
        prompts, seeds, lines, plan, w, h, instructs = video.shot_lists(json.loads(result_json))
        print(f"[NewsFeed] {len(lines)} lines, {len(prompts)} images to generate at {w}x{h} (spare={plan['spare_image']})")
        return (prompts, seeds, lines, json.dumps(plan), w, h, instructs)


class NewsParseScript:
    """Parses a written story script (TITLE/VOICE/LOOK, CHARACTER and LOCATION lines, SHOT blocks) into per-picture
    and per-shot lists. No LLM involved."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "script":   ("STRING", {"multiline": True, "default": ""}),
            "speaker":  ("STRING", {"default": "Ryan"}),
            "width":    ("INT", {"default": 0, "min": 0, "max": 1280}),
            "height":   ("INT", {"default": 0, "min": 0, "max": 1280}),
            "seed":     ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
            "caption":  ("INT", {"default": 1, "min": 0, "max": 3}),
            "energy":   ("INT", {"default": 1, "min": 0, "max": 2}),
        }}

    RETURN_TYPES = ("STRING", "INT", "STRING", "INT", "STRING", "STRING", "INT", "INT", "STRING", "STRING", "STRING",
                    "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("ref_prompts", "ref_seeds", "shot_prompts", "shot_seeds", "lines", "plan_json", "width", "height",
                    "instructs", "speakers", "layout_prompts", "asset_prompts", "asset_seeds", "asset_widths",
                    "asset_heights")
    OUTPUT_IS_LIST = (True, True, True, True, True, False, False, False, True, True, True, True, True, True, True)
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    def run(self, script, speaker, width, height, seed, caption, energy):
        snap = lambda v, d: d if v <= 0 else max(384, min(1280, int(round(v / 32.0)) * 32))
        w, h = snap(width, 1024), snap(height, 576)
        plan, rp, rs, sp_, ss, lines, instructs, speakers = scriptlib.parse(script, speaker, seed, caption, energy, (w, h))
        print(f"[NewsFeed] script: {plan['n_chars']} characters, {plan['n_locs']} locations, {len(lines)} shots at {w}x{h}, "
              f"speakers={sorted(set(speakers))}, warnings={plan['warnings']}")
        return (rp, rs, sp_, ss, lines, json.dumps(plan), w, h, instructs, speakers, plan["layouts"],
                plan["assets"]["prompts"], plan["assets"]["seeds"], plan["assets"]["w"], plan["assets"]["h"])


class NewsShotRefs:
    """Per shot: the location picture, and a 'cast board' with that shot's characters side by side (the location
    picture again when nobody is in the shot). Input is the drawn reference pictures: characters first, then locations."""
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plan_json": ("STRING", {"forceInput": True}), "refs": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("location_refs", "cast_refs")
    OUTPUT_IS_LIST = (True, True)
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    @staticmethod
    def _fit(t, w, h):
        """Cover-crop a picture to w x h: the shot edit takes its size from its location picture."""
        from PIL import Image
        im = Image.fromarray((t.cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
        if im.size != (w, h):
            im = video.cover_crop(im, w, h)
        return torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.0)

    def run(self, plan_json, refs):
        from PIL import Image
        plan = json.loads(plan_json[0])
        imgs = [r[0] if r.dim() == 4 else r for r in refs]
        for i, t in enumerate(imgs):
            print(f"[NewsFeed] ref {i} ({'character' if i < plan['n_chars'] else 'location'}) size {t.shape[1]}x{t.shape[0]}")
        nc = plan["n_chars"]
        chars, locs = imgs[:nc], imgs[nc:]
        ch = 512
        gw, gh = plan["story"]["gen_size"]
        loc_out, cast_out = [], []
        for s in plan["shots"]:
            loc = self._fit(locs[s["place"]], gw, gh)
            loc_out.append(loc.unsqueeze(0))
            if not s["cast"]:
                cast_out.append(loc.unsqueeze(0))
                continue
            tiles = []
            for c in s["cast"]:
                im = Image.fromarray((chars[c].cpu().numpy() * 255).clip(0, 255).astype(np.uint8))
                tw = max(32, int(round(im.width * ch / im.height / 32.0)) * 32)
                tiles.append(im.resize((tw, ch), Image.LANCZOS))
            board = Image.new("RGB", (sum(t.width for t in tiles), ch), (255, 255, 255))
            x = 0
            for t in tiles:
                board.paste(t, (x, 0))
                x += t.width
            cast_out.append(torch.from_numpy(np.asarray(board, dtype=np.float32) / 255.0).unsqueeze(0))
        print(f"[NewsFeed] shot refs: {len(loc_out)} shots, cast boards {[(c.shape[2], c.shape[1]) for c in cast_out]}")
        return (loc_out, cast_out)


class NewsCompositeShots:
    """Cutout mode: cuts each drawn character out of its flat background and pastes the cast into every shot's own plate.
    Input is the drawn assets: characters first, then one plate per shot. Output is one finished picture per shot."""
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plan_json": ("STRING", {"forceInput": True}), "assets": ("IMAGE",)}}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    def run(self, plan_json, assets):
        from PIL import Image
        from . import composite
        plan = json.loads(plan_json[0])
        pics = [Image.fromarray((a[0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)) for a in assets]
        nc = plan["n_chars"]
        figs = [composite.cutout(p) for p in pics[:nc]]
        for i, f in enumerate(figs):
            print(f"[NewsFeed] cutout {plan['char_ids'][i]}: {f.width}x{f.height}")
        gw, gh = plan["story"]["gen_size"]
        out = []
        for i, s in enumerate(plan["shots"]):
            plate = pics[nc + i]
            if plate.size != (gw, gh):
                plate = video.cover_crop(plate, gw, gh)
            img = composite.place_cast(plate, [figs[c] for c in s["cast"]], s.get("framing", "medium"), i)
            out.append(torch.from_numpy(np.asarray(img, dtype=np.float32) / 255.0).unsqueeze(0))
        return (out,)


class NewsVoiceScript:
    """Story script -> Drama 3 / VoiceCast text (one take per SAY line, a voice per speaker)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"script": ("STRING", {"multiline": True, "default": ""})}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("cast_text", "plan_json")
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    def run(self, script):
        cast_text, plan = scriptlib.voice_script(script)
        print(f"[NewsFeed] voice script: {len(plan['lines'])} lines, voices={plan['voices']}, warnings={plan['warnings']}")
        print(f"[NewsFeed] cast text: {cast_text[:300]}")
        return (cast_text, json.dumps(plan))


class NewsDramaTimes:
    """Start and end second of every Drama 3 take in the decoded audio (Decode joins takes with 0.3 s of silence and
    must run with shorten_long_pauses off). Output is a JSON string and the same JSON as a data image."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"codes": ("DRAMA3_CODES",), "plan_json": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("STRING", "IMAGE")
    RETURN_NAMES = ("timing_json", "timing_data")
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    def run(self, codes, plan_json):
        frame, gap = 2048 / 44100.0, 0.3
        plan = json.loads(plan_json)
        bounds, acc = [], 0
        for t in codes["takes"]:
            acc += t
            bounds.append(acc)

        def sec(f, end=False):
            k = sum(1 for b in bounds[:-1] if (f > b if end else f >= b))
            return round(f * frame + gap * k, 3)
        out = []
        for j, ln in enumerate(codes["lines"]):
            want = plan["lines"][j] if j < len(plan["lines"]) else {}
            out.append({"i": j, "speaker": want.get("speaker", ln.get("speaker")), "text": want.get("text", ln.get("text")),
                        "start": sec(ln["start"]), "end": sec(ln["end"], True)})
        total = round(acc * frame + gap * max(0, len(bounds) - 1), 3)
        if len(out) != len(plan["lines"]):
            print(f"[NewsFeed] WARNING: Drama 3 produced {len(out)} lines for {len(plan['lines'])} SAY lines")
        text = json.dumps({"seconds": total, "lines": out}, ensure_ascii=False)
        print(f"[NewsFeed] timing: {text[:600]}")
        return (text, _data_image(text))


class NewsDramaGaps:
    """Puts 0.3 s of digital silence between EVERY pair of lines in a Drama 3 narration (Decode already does it between
    takes; lines of one take run together). Needs Decode with shorten_long_pauses off. story-assemble cuts at these gaps."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",), "codes": ("DRAMA3_CODES",)}}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    def run(self, audio, codes):
        sr = int(audio["sample_rate"])
        frame, gap = 2048 / 44100.0, 0.3
        wave = audio["waveform"]
        bounds, acc = [], 0
        for t in codes["takes"]:
            acc += t
            bounds.append(acc)
        starts = [0] + bounds[:-1]
        cuts = []
        for ln in codes["lines"][1:]:
            f = ln["start"]
            if f in starts:
                continue
            k = sum(1 for b in bounds[:-1] if f >= b)
            cuts.append(int(round((f * frame + gap * k) * sr)))
        zeros = torch.zeros(wave.shape[:-1] + (int(round(gap * sr)),), dtype=wave.dtype)
        parts, last = [], 0
        for c in sorted(cuts):
            parts += [wave[..., last:c], zeros]
            last = c
        parts.append(wave[..., last:])
        print(f"[NewsFeed] inserted {len(cuts)} line gaps into {wave.shape[-1] / sr:.1f}s of narration")
        return ({"waveform": torch.cat(parts, dim=-1), "sample_rate": sr},)


class NewsAssemble:
    """Cuts a story's stills to its narration: audio file + stills (by filename) + the story script -> captioned mp4.
    The narration is split at the silences between lines (story-voice joins takes with 0.3 s of silence)."""

    @classmethod
    def INPUT_TYPES(cls):
        opt = {f"image{i}": ("STRING", {"default": ""}) for i in range(1, 9)}
        opt.update({"audio_init_filename": ("STRING", {"default": ""}), "audio_init_url": ("STRING", {"default": ""})})
        return {"required": {
            "script": ("STRING", {"multiline": True, "default": ""}),
            "audio_file": ("STRING", {"default": ""}),
            "caption": ("INT", {"default": 1, "min": 0, "max": 3}),
            "seed": ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
        }, "optional": opt}

    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "report")
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    async def run(self, script, audio_file, caption, seed, audio_init_filename="", audio_init_url="", **images):
        import asyncio
        return await asyncio.to_thread(self._run_sync, script, audio_file, caption, seed, audio_init_filename,
                                       audio_init_url, images)

    @staticmethod
    def _load_audio(name):
        import os
        import re
        import urllib.request
        import folder_paths
        import av
        if name.startswith(("http://", "https://")):
            cache = os.path.join(folder_paths.get_temp_directory(), "newsassemble")
            os.makedirs(cache, exist_ok=True)
            path = os.path.join(cache, re.sub(r"[^A-Za-z0-9_.-]", "_", name.split("?")[0])[-80:] or "audio")
            urllib.request.urlretrieve(name, path)
        else:
            path = folder_paths.get_annotated_filepath(name)
        with av.open(path) as container:
            stream = container.streams.audio[0]
            rate = stream.rate
            resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
            parts = [o.to_ndarray() for frame in container.decode(stream) for o in resampler.resample(frame)]
        return np.concatenate(parts, axis=-1).reshape(-1).astype(np.float32), int(rate)

    def _run_sync(self, script, audio_file, caption, seed, init_name, init_url, images):
        import os
        import shutil
        import tempfile
        import folder_paths
        from PIL import Image
        from comfy_api.latest import InputImpl
        from . import assemble
        name = next((x.strip() for x in (audio_file, init_name, init_url) if x and x.strip()), "")
        if not name:
            raise ValueError("NewsAssemble: no audio. Give /audio1:R<hash of an audio render> or reply to an audio file")
        samples, sr = self._load_audio(name)
        pics = []
        for i in range(1, 9):
            v = (images.get(f"image{i}") or "").strip()
            if v:
                pics.append(np.asarray(Image.open(folder_paths.get_annotated_filepath(v)).convert("RGB")))
        if not pics:
            raise ValueError("NewsAssemble: no stills. Give /image1:R<hash> ... /image8:R<hash> in shot order")
        h, w = pics[0].shape[:2]
        out_size = (int(round(w * 1.25 / 2)) * 2, int(round(h * 1.25 / 2)) * 2) if w <= 1024 else (w // 2 * 2, h // 2 * 2)
        plan, lines = scriptlib.assemble_plan(script, len(pics), caption, out_size, seed)
        pics = pics[:len(lines)]
        segs, how = assemble.split_audio(samples, sr, lines)
        work = tempfile.mkdtemp(prefix="newsasm_")
        out = os.path.join(tempfile.mkdtemp(prefix="newsasm_out_"), "story.mp4")
        info = video.compile_video(plan, pics, [(s, sr) for s in segs], out, work)
        shutil.rmtree(work, ignore_errors=True)
        info.update({"audio_seconds": round(len(samples) / sr, 2), "split": how, "warnings": plan["warnings"]})
        print(f"[NewsFeed] assembled {info['seconds']}s from {len(pics)} stills, audio {len(samples) / sr:.1f}s, split={how}, "
              f"warnings={plan['warnings']}")
        return (InputImpl.VideoFromFile(out), json.dumps(info, ensure_ascii=False))


class NewsKokoro:
    """Kokoro-82M (ONNX) text to speech for one line. Run per list item."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "text":  ("STRING", {"forceInput": True}),
            "voice": ("STRING", {"default": "af_heart"}),
            "speed": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 1.6, "step": 0.05}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    def run(self, text, voice, speed):
        samples, sr = video.kokoro_synth(text, voice, speed)
        wave = torch.from_numpy(samples).reshape(1, 1, -1)
        print(f"[NewsFeed] kokoro {len(samples) / sr:.2f}s for {len(text)} chars")
        return ({"waveform": wave, "sample_rate": sr},)


class NewsCompile:
    """Assemble pictures (article photos + generated stills), per-line narration and captions into an mp4."""
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan_json": ("STRING", {"forceInput": True}),
            "images":    ("IMAGE",),
            "audios":    ("AUDIO",),
        }}

    RETURN_TYPES = ("VIDEO", "STRING", "IMAGE")
    RETURN_NAMES = ("video", "report", "report_image")
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    async def run(self, plan_json, images, audios):
        import asyncio
        # ffmpeg is blocking: keep the event loop alive (a blocked loop can make a finished job report timed_out)
        return await asyncio.to_thread(self._run_sync, plan_json[0], images, audios)

    def _run_sync(self, plan_json, images, audios):
        import os
        import shutil
        import tempfile
        from comfy_api.latest import InputImpl
        plan = json.loads(plan_json)
        gen = [(im[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8) for im in images]
        if plan.get("spare_image"):
            gen = []
        auds = []
        for a in audios:
            w = a["waveform"]
            auds.append((w.reshape(-1).cpu().numpy().astype(np.float32), int(a["sample_rate"])))
        work = tempfile.mkdtemp(prefix="newsvideo_")
        out = os.path.join(tempfile.mkdtemp(prefix="newsvideo_out_"), "news.mp4")
        info = video.compile_video(plan, gen, auds, out, work)
        shutil.rmtree(work, ignore_errors=True)
        info["title"] = plan["story"]["title"]
        info["link"] = plan["story"]["link"]
        text = json.dumps(info, ensure_ascii=False, indent=1)
        print(f"[NewsFeed] compiled {info['seconds']}s: " + ", ".join(s["picture"] for s in info["shots"]))
        return (InputImpl.VideoFromFile(out), text, _data_image(text))


class NewsMark:
    """Passes AUDIO through unchanged and logs seconds since the ComfyUI process started (the 180 s clock),
    so a probe workflow can time its stages. Logs as `[NewsFeed] MARK <label> t=<s>`."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"audio": ("AUDIO",), "label": ("STRING", {"default": "mark"})},
                "optional": {"wait_for": ("AUDIO",)}}   # ignored; only forces this node to run after another

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    def run(self, audio, label, wait_for=None):
        import time
        try:
            import psutil
            since = time.time() - psutil.Process().create_time()
        except Exception:
            since = -1.0
        w = audio["waveform"]
        print(f"[NewsFeed] MARK {label} t={since:.1f}s audio={w.shape[-1] / audio['sample_rate']:.2f}s")
        return (audio,)


class NewsGateAudio:
    """Lazy two-way switch for AUDIO: with enable > 0 it returns `on`, else `off`, and only the chosen branch is
    ever evaluated (ComfyUI lazy inputs), so an optional edit pass costs nothing when it is switched off."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "enable": ("INT", {"default": 0, "min": 0, "max": 9}),
            "off": ("AUDIO", {"lazy": True}),
            "on": ("AUDIO", {"lazy": True}),
        }}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    def check_lazy_status(self, enable, off=None, on=None):
        if enable > 0:
            return ["on"] if on is None else []
        return ["off"] if off is None else []

    def run(self, enable, off=None, on=None):
        return ((on if enable > 0 else off),)


NODE_CLASS_MAPPINGS = {"NewsPickStory": NewsPickStory, "NewsFinishScript": NewsFinishScript,
                       "NewsShotLists": NewsShotLists, "NewsParseScript": NewsParseScript, "NewsShotRefs": NewsShotRefs, "NewsDramaGaps": NewsDramaGaps, "NewsAssemble": NewsAssemble, "NewsVoiceScript": NewsVoiceScript, "NewsDramaTimes": NewsDramaTimes, "NewsCompositeShots": NewsCompositeShots, "NewsKokoro": NewsKokoro, "NewsCompile": NewsCompile,
                       "NewsMark": NewsMark, "NewsGateAudio": NewsGateAudio,
                       "NewsQwenTTS": qwentts.NewsQwenTTS, "NewsRepeatLine": qwentts.NewsRepeatLine,
                       "NewsSplitLines": qwentts.NewsSplitLines}
NODE_DISPLAY_NAME_MAPPINGS = {"NewsPickStory": "News Pick Story", "NewsFinishScript": "News Finish Script",
                              "NewsShotLists": "News Shot Lists", "NewsParseScript": "News Parse Script", "NewsShotRefs": "News Shot Refs", "NewsDramaGaps": "News Drama Gaps", "NewsAssemble": "News Assemble", "NewsVoiceScript": "News Voice Script", "NewsDramaTimes": "News Drama Times", "NewsCompositeShots": "News Composite Shots", "NewsKokoro": "News Kokoro TTS",
                              "NewsCompile": "News Compile", "NewsMark": "News Mark", "NewsGateAudio": "News Gate Audio",
                              "NewsQwenTTS": "News Qwen3-TTS (batch)", "NewsRepeatLine": "News Repeat Line",
                              "NewsSplitLines": "News Split Lines"}
