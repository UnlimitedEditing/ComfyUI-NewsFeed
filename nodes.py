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

    RETURN_TYPES = ("STRING", "INT", "STRING", "INT", "STRING", "STRING", "INT", "INT", "STRING", "STRING")
    RETURN_NAMES = ("ref_prompts", "ref_seeds", "shot_prompts", "shot_seeds", "lines", "plan_json", "width", "height",
                    "instructs", "speakers")
    OUTPUT_IS_LIST = (True, True, True, True, True, False, False, False, True, True)
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    def run(self, script, speaker, width, height, seed, caption, energy):
        snap = lambda v, d: d if v <= 0 else max(384, min(1280, int(round(v / 32.0)) * 32))
        w, h = snap(width, 1024), snap(height, 576)
        plan, rp, rs, sp_, ss, lines, instructs, speakers = scriptlib.parse(script, speaker, seed, caption, energy, (w, h))
        print(f"[NewsFeed] script: {plan['n_chars']} characters, {plan['n_locs']} locations, {len(lines)} shots at {w}x{h}, "
              f"speakers={sorted(set(speakers))}, warnings={plan['warnings']}")
        return (rp, rs, sp_, ss, lines, json.dumps(plan), w, h, instructs, speakers)


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
            bw = max(sum(t.width for t in tiles), 2 * ch)   # a lone character sits centred on a wide board
            board = Image.new("RGB", (bw, ch), (255, 255, 255))
            x = (bw - sum(t.width for t in tiles)) // 2
            for t in tiles:
                board.paste(t, (x, 0))
                x += t.width
            cast_out.append(torch.from_numpy(np.asarray(board, dtype=np.float32) / 255.0).unsqueeze(0))
        print(f"[NewsFeed] shot refs: {len(loc_out)} shots, cast boards {[(c.shape[2], c.shape[1]) for c in cast_out]}")
        return (loc_out, cast_out)


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
                       "NewsShotLists": NewsShotLists, "NewsParseScript": NewsParseScript, "NewsShotRefs": NewsShotRefs, "NewsKokoro": NewsKokoro, "NewsCompile": NewsCompile,
                       "NewsMark": NewsMark, "NewsGateAudio": NewsGateAudio,
                       "NewsQwenTTS": qwentts.NewsQwenTTS, "NewsRepeatLine": qwentts.NewsRepeatLine,
                       "NewsSplitLines": qwentts.NewsSplitLines}
NODE_DISPLAY_NAME_MAPPINGS = {"NewsPickStory": "News Pick Story", "NewsFinishScript": "News Finish Script",
                              "NewsShotLists": "News Shot Lists", "NewsParseScript": "News Parse Script", "NewsShotRefs": "News Shot Refs", "NewsKokoro": "News Kokoro TTS",
                              "NewsCompile": "News Compile", "NewsMark": "News Mark", "NewsGateAudio": "News Gate Audio",
                              "NewsQwenTTS": "News Qwen3-TTS (batch)", "NewsRepeatLine": "News Repeat Line",
                              "NewsSplitLines": "News Split Lines"}
