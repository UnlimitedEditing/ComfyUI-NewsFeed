import json
import math

import numpy as np
import torch

from . import core


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


NODE_CLASS_MAPPINGS = {"NewsPickStory": NewsPickStory, "NewsFinishScript": NewsFinishScript}
NODE_DISPLAY_NAME_MAPPINGS = {"NewsPickStory": "News Pick Story", "NewsFinishScript": "News Finish Script"}
