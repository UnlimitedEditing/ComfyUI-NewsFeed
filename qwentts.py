"""Qwen3-TTS (CustomVoice) batch narration through the OFFICIAL `qwen-tts` package (Apache-2.0).

All lines are generated in ONE batched call, each with its own natural-language `instruct` (e.g. "excited, fast"),
which is what makes this much cheaper per line than one-clip-at-a-time engines. Model files come from concept_mapping
(models/qwen-tts/Qwen3-TTS-12Hz-<size>-CustomVoice/); the Hugging Face id is only a fallback.
"""
import os
import time

import numpy as np
import torch

_MODELS = {}


def load_model(size):
    if size not in _MODELS:
        import folder_paths
        from qwen_tts import Qwen3TTSModel
        name = f"Qwen3-TTS-12Hz-{size}-CustomVoice"
        local = os.path.join(folder_paths.models_dir, "qwen-tts", name)
        src = local if os.path.isdir(local) else f"Qwen/{name}"
        t = time.time()
        _MODELS[size] = Qwen3TTSModel.from_pretrained(
            src, device_map="cuda:0", dtype=torch.bfloat16, attn_implementation="sdpa")
        print(f"[NewsFeed] qwen-tts loaded {size} from {src} in {time.time() - t:.1f}s")
    return _MODELS[size]


def _gen_batch(model, texts, instructs, speakers, language):
    n = len(texts)
    return model.generate_custom_voice(
        text=list(texts), language=[language] * n, speaker=list(speakers), instruct=list(instructs))


def generate(texts, instructs, speaker, size, language, seed, batch=8, speakers=None):
    """Returns (list of float32 mono arrays, sample_rate). Lines go through in batches of `batch`; if a batch fails
    (e.g. out of memory) it is retried in halves, down to single lines."""
    model = load_model(size)
    n = len(texts)
    instructs = (list(instructs) + [""] * n)[:n]
    speakers = [(speakers[i] if speakers and i < len(speakers) and speakers[i] else speaker) for i in range(n)]
    if seed:
        torch.manual_seed(int(seed))
    t = time.time()
    out, sr = [None] * n, 24000

    def run(idx):
        nonlocal sr
        try:
            wavs, sr = _gen_batch(model, [texts[i] for i in idx], [instructs[i] for i in idx], [speakers[i] for i in idx], language)
            for i, w in zip(idx, wavs):
                out[i] = np.asarray(w.detach().cpu().numpy() if hasattr(w, "detach") else w,
                                    dtype=np.float32).reshape(-1)
        except Exception as e:
            if len(idx) == 1:
                raise
            print(f"[NewsFeed] qwen-tts batch of {len(idx)} failed ({str(e)[:80]}); retrying in halves")
            torch.cuda.empty_cache()
            mid = len(idx) // 2
            run(idx[:mid])
            run(idx[mid:])

    for s in range(0, n, batch):
        run(list(range(s, min(n, s + batch))))
    took = time.time() - t
    audio_s = sum(len(w) for w in out) / sr
    print(f"[NewsFeed] qwen-tts generated {n} lines, {audio_s:.1f}s of speech in {took:.1f}s "
          f"(compute per second of speech = {took / max(audio_s, 0.01):.2f})")
    return out, int(sr)


class NewsQwenTTS:
    """One batched Qwen3-TTS call for every line; outputs one AUDIO per line (list)."""
    INPUT_IS_LIST = True

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "texts":    ("STRING", {"forceInput": True}),
            "instructs": ("STRING", {"forceInput": True}),
            "speaker":  ("STRING", {"default": "Ryan"}),
            "model_size": (["1.7B", "0.6B"], {"default": "1.7B"}),
            "language": ("STRING", {"default": "English"}),
            "seed":     ("INT", {"default": 0, "min": 0, "max": 2**31 - 1}),
        }, "optional": {"speakers": ("STRING", {"forceInput": True})}}

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    def run(self, texts, instructs, speaker, model_size, language, seed, speakers=None):
        wavs, sr = generate(texts, instructs, speaker[0], model_size[0], language[0], seed[0], speakers=speakers)
        return ([{"waveform": torch.from_numpy(w).reshape(1, 1, -1), "sample_rate": sr} for w in wavs],)


class NewsRepeatLine:
    """Probe helper: one line repeated `count` times, as a list."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"line": ("STRING", {"default": ""}), "count": ("INT", {"default": 4, "min": 1, "max": 16})}}

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    def run(self, line, count):
        return ([line] * count,)


class NewsSplitLines:
    """Splits multi-line text into a list (one item per non-empty line)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"text": ("STRING", {"multiline": True, "default": ""})}}

    RETURN_TYPES = ("STRING",)
    OUTPUT_IS_LIST = (True,)
    FUNCTION = "run"
    CATEGORY = "NewsFeed"

    def run(self, text):
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        return (lines or [""],)
