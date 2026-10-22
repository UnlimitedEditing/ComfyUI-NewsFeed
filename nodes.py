import json
import math

import numpy as np
import torch

from . import core, video


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
            "length":       ("INT", {"default": 45, "min": 15, "max": 120}),
            "shot_seconds": ("INT", {"default": 6, "min": 3, "max": 15}),
            "caption":      ("INT", {"default": 1, "min": 0, "max": 3}),
        }}

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("system_prompt", "user_prompt", "story_json")
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    def run(self, category, pick_seed, tone, visual, length, shot_seconds, caption):
        print(f"[NewsFeed] inputs category={category!r} seed={pick_seed} tone={tone} visual={visual} "
              f"length={length} shot_seconds={shot_seconds} caption={caption}")
        item, paras, media = core.pick_story(category, pick_seed)
        system, user, story = core.build_brief(item, paras, media, pick_seed, tone, visual,
                                               length, shot_seconds, caption)
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

    RETURN_TYPES = ("STRING", "INT", "STRING", "STRING")
    RETURN_NAMES = ("image_prompts", "seeds", "lines", "plan_json")
    OUTPUT_IS_LIST = (True, True, True, False)
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    def run(self, result_json):
        prompts, seeds, lines, plan = video.shot_lists(json.loads(result_json))
        print(f"[NewsFeed] {len(lines)} lines, {len(prompts)} images to generate (spare={plan['spare_image']})")
        return (prompts, seeds, lines, json.dumps(plan))


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


NODE_CLASS_MAPPINGS = {"NewsPickStory": NewsPickStory, "NewsFinishScript": NewsFinishScript,
                       "NewsShotLists": NewsShotLists, "NewsKokoro": NewsKokoro, "NewsCompile": NewsCompile}
NODE_DISPLAY_NAME_MAPPINGS = {"NewsPickStory": "News Pick Story", "NewsFinishScript": "News Finish Script",
                              "NewsShotLists": "News Shot Lists", "NewsKokoro": "News Kokoro TTS",
                              "NewsCompile": "News Compile"}
