"""Parser for the story script format (plain labelled lines, stdlib only so it is testable offline).

TITLE: ...              story title (optional)
VOICE: ...              overall narrator direction (optional)
LOOK: ...               picture style, added to every picture (optional)
CHARACTER odd: ...      one line per character: id, then a full precise description (face, hair, clothes, build)
LOCATION lighthouse: .. one line per place: id, then a full precise description (architecture, light, palette)
SHOT 1                  then, in any order, per shot:
SAY: ...                the spoken line (required)
PLACE: lighthouse       location id (required)
CAST: odd, mira         character ids appearing in the picture (optional, at most 3)
ACTION: ...             what is happening and how it is framed (required)
DELIVERY: ...           short performance note for that line (optional)
SPEAKER: ...            NARRATOR (default) or one of the Qwen3-TTS voices (optional)
CAMERA: ...             push-in, pull-out, pan left, pan right, tilt up, drift (optional)

No square brackets, angle brackets or pipes are used: Graydient's prompt parser mangles them. A value may continue
over several lines. Keys are upper case and end with a colon; "SHOT 3" may or may not have a colon.
The renderer draws every CHARACTER and LOCATION once (text to image), then draws each shot as an edit that uses the
location picture and the cast picture as references, so characters and places stay the same from shot to shot.
"""
import json
import random
import re

from . import core

SPEAKERS = ["Ryan", "Aiden", "Serena", "Vivian", "Uncle_Fu", "Dylan", "Eric"]
MAX_SHOTS = 12
MAX_CAST = 3
CAMERAS = {"push-in": "slow push-in", "pull-out": "slow pull-out", "pan left": "pan left",
           "pan right": "pan right", "tilt up": "tilt up", "drift": "static with subtle drift"}

KEY = re.compile(r"(?:^|(?<=\s))(?:\*\*)?((?:CHARACTER|LOCATION)\s+[A-Za-z0-9_-]+\s*:|SHOT\s*\d+\s*:?|"
                 r"OBJECT\s+[A-Za-z0-9_-]+(?:\s+\d+){4}\s*:|SCENE:|"
                 r"VOICE\s+[A-Za-z0-9_-]+\s*:|TITLE:|VOICE:|LOOK:|SAY:|DELIVERY:|PLACE:|CAST:|ACTION:|SPEAKER:|CAMERA:)(?:\*\*)?")
NO_TEXT = " No text, no lettering, no logos."
CHAR_SUFFIX = ", full body character reference, standing, facing the viewer, plain light neutral background, even lighting"
LOC_SUFFIX = ", empty location with no people, establishing view"


def tokenize(text):
    """Returns list of (KEY, id, value) in order; text before the first key is dropped."""
    text = (text or "").replace("\\<", "<").replace("\r", "")
    marks = list(KEY.finditer(text))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        raw = m.group(1)
        val = re.sub(r"\s+", " ", text[m.end():end]).strip()
        if raw.startswith("SHOT"):
            out.append(("SHOT", "", ""))
        elif raw.startswith("OBJECT"):
            parts = raw.rstrip(":").split()
            out.append(("OBJECT", " ".join(parts[1:]).lower(), val))
        elif re.match(r"VOICE\s+\S", raw):
            out.append(("VOICEOF", raw.rstrip(":").split(None, 1)[1].strip().lower(), val))
        elif raw.startswith(("CHARACTER", "LOCATION")):
            kind, ident = raw.rstrip(":").split(None, 1)
            out.append((kind, ident.strip().lower(), val))
        else:
            out.append((raw.rstrip(":"), "", val))
    return out


def _speaker(raw, default):
    s = (raw or "").strip().strip("\"'").replace(" ", "_")
    if not s or s.upper() == "NARRATOR":
        return default, None
    for sp in SPEAKERS:
        if sp.lower() == s.lower():
            return sp, None
    return default, f"unknown speaker {raw!r}, using {default}"


def explicit_layout(scene, objects, chars, size):
    """Layout JSON from the script's own boxes. objects: (ident, description) where ident is 'name x0 y0 x1 y1' with the box
    in percent of the frame (x left to right, y top to bottom). A name that matches a CHARACTER id pulls in that character's
    full description verbatim, with the line after the colon as the pose and expression."""
    import math
    w, h = size
    items = []
    for ident, desc in objects[:4]:
        parts = ident.split()
        name, nums = parts[0], [max(0, min(100, int(p))) for p in parts[1:5]]
        x0, y0, x1, y1 = nums
        x0, x1 = sorted((x0, x1))
        y0, y1 = sorted((y0, y1))
        if x1 - x0 < 3 or y1 - y0 < 3:
            continue
        y1 = min(y1, 90)        # bottom of the frame stays free for captions
        base = core._plain(chars[name], 400) if name in chars else ""
        cap = (base + (", " if base and desc else "") + core._plain(desc, 300)).strip() or name
        bb = [math.floor(y0 * 10 * h / 1000) - h // 2, math.floor(x0 * 10 * w / 1000) - w // 2,
              math.ceil(y1 * 10 * h / 1000) - h // 2, math.ceil(x1 * 10 * w / 1000) - w // 2]
        items.append({"label": "object", "caption": cap + ", no text", "bbox": bb})
    return json.dumps({"summary": {"caption": scene}, "items": items, "bbox_coord_space": "abs_centralized",
                       "bbox_format": "yxyx"}, ensure_ascii=False, separators=(",", ":"))


def moss_layout(scene, people, framing, size, idx=0):
    """MOSS-Image layout JSON (same schema as core.layout_prompt): the scene caption, then one object box per character
    carrying that character's full description verbatim, placed by the shot's framing and cast size.
    bboxes are [y0, x0, y1, x1] on a 0-1000 grid; the bottom ~12% stays clear for captions."""
    import json
    import math
    w, h = size
    f = (framing or "").lower()
    if "close" in f:
        top, bot, half = 60, 880, 190 if len(people) > 1 else 230
    elif "wide" in f:
        top, bot, half = 430, 880, 75
    else:
        top, bot, half = 160, 880, 140 if len(people) > 1 else 170
    centres = {1: [450 + 100 * (idx % 2)], 2: [320, 680], 3: [220, 500, 780]}.get(len(people), [])

    def bbox(y0, x0, y1, x1):
        return [math.floor(y0 * h / 1000) - h // 2, math.floor(x0 * w / 1000) - w // 2,
                math.ceil(y1 * h / 1000) - h // 2, math.ceil(x1 * w / 1000) - w // 2]
    items = [{"label": "object", "caption": desc, "bbox": bbox(top, max(20, c - half), bot, min(980, c + half))}
             for desc, c in zip(people, centres)]
    return json.dumps({"summary": {"caption": scene}, "items": items, "bbox_coord_space": "abs_centralized",
                       "bbox_format": "yxyx"}, ensure_ascii=False, separators=(",", ":"))


def parse(text, default_speaker="Ryan", seed=0, caption=1, energy=1, gen_size=(1024, 576)):
    """Returns (plan, ref_prompts, ref_seeds, shot_prompts, shot_seeds, lines, instructs, speakers).
    ref_prompts are the characters first, then the locations (plan['n_chars'] says where the split is)."""
    default_speaker, _ = _speaker(default_speaker, "Ryan")
    toks = tokenize(text)
    head = {"TITLE": "", "VOICE": "", "LOOK": ""}
    chars, locs, shots, cur, warnings = {}, {}, [], None, []
    for k, ident, v in toks:
        if k == "SHOT":
            cur = {}
            shots.append(cur)
        elif k in head:
            head[k] = head[k] or v
        elif k == "CHARACTER":
            chars.setdefault(ident, v)
        elif k == "LOCATION":
            locs.setdefault(ident, v)
        elif k == "OBJECT" and cur is not None:
            cur.setdefault("objects", []).append((ident, v))
        elif cur is not None:
            cur[k] = v if k not in cur else cur[k] + " " + v
    if not locs and any(x.get("SCENE") for x in shots):
        locs["scene"] = ""
    for x in shots:
        if not x.get("ACTION") and x.get("SCENE"):
            x["ACTION"] = x["SCENE"]
    if not locs:
        raise ValueError("no LOCATION lines: define each place as LOCATION id: description. Received %d chars: %r" % (len(text or ""), (text or "")[:200]))
    if len(chars) + len(locs) > 6:
        warnings.append(f"{len(chars)} characters + {len(locs)} locations: each costs a picture, trim to save time")
    good = []
    for n, s in enumerate(shots, 1):
        place = (s.get("PLACE") or "").strip().lower()
        if not s.get("SAY") or not s.get("ACTION"):
            warnings.append(f"shot {n}: needs SAY and ACTION, skipped")
            continue
        if place not in locs:
            if place:
                warnings.append(f"shot {n}: PLACE {place!r} is not a defined location, using {next(iter(locs))!r}")
            place = next(iter(locs))
        cast = [c.strip().lower() for c in re.split(r"[,;]| and ", s.get("CAST") or "") if c.strip()]
        keep = [c for c in cast if c in chars]
        if len(keep) != len(cast):
            warnings.append(f"shot {n}: unknown cast ids dropped: {sorted(set(cast) - set(keep))}")
        s["place"], s["cast"] = place, keep[:MAX_CAST]
        good.append(s)
    if not good:
        raise ValueError("no usable shots. " + "; ".join(warnings))
    if len(good) > MAX_SHOTS:
        warnings.append(f"{len(good)} shots, only the first {MAX_SHOTS} are made")
        good = good[:MAX_SHOTS]

    pre = core.load_json("presets.json")
    rng = random.Random(seed)
    look = core._plain(head["LOOK"], 200)
    look_s = (look.rstrip(".") + ". ") if look else ""
    energy_note = pre["energy"].get(str(int(energy)), "")
    ids_c, ids_l = list(chars), list(locs)
    ref_prompts = [look_s + core._plain(chars[c], 500) + CHAR_SUFFIX + "." + NO_TEXT for c in ids_c]
    ref_prompts += [look_s + core._plain(locs[p], 500) + LOC_SUFFIX + "." + NO_TEXT for p in ids_l]
    ref_seeds = [int(seed) + 100 + i for i in range(len(ref_prompts))]

    shot_prompts, shot_seeds, lines, instructs, speakers, plan_shots = [], [], [], [], [], []
    layouts = []
    for i, s in enumerate(good):
        sp, warn = _speaker(s.get("SPEAKER"), default_speaker)
        if warn:
            warnings.append(f"shot {i+1}: {warn}")
        is_narr = (s.get("SPEAKER") or "NARRATOR").strip().upper() in ("NARRATOR", "")
        instruct = core.compose_instruct(head["VOICE"] if is_narr else "", "", energy_note, s.get("DELIVERY"))
        if not is_narr and not instruct:
            instruct = "Speak as a character in a story."
        if s["cast"]:
            who = "; ".join(f"{c} ({core._plain(chars[c], 160)})" for c in s["cast"])
            n = len(s["cast"])
            cast_s = (f"Image 2 shows the one character in this picture: {who}. " if n == 1 else
                      f"Image 2 shows the characters, side by side from left to right: {who}. ")
            cast_s += (f"There {'is exactly one person' if n == 1 else f'are exactly {n} people'} in the picture, each shown "
                       "once, never duplicated. ")
            with_s = ", with those characters from image 2"
        else:
            cast_s, with_s = "", ", with nobody in it"
        p = ("Image 1 shows the location. " + cast_s + "Draw a new picture set in the location from image 1" + with_s
             + ". What happens: " + core._plain(s["ACTION"], 400).rstrip(".") + ". "
             + "Keep each character's exact face, hair, clothes and proportions, and keep the place, light and colours "
             "of the location. " + look_s + NO_TEXT.strip())
        shot_prompts.append(p)
        scene = (look_s + (core._plain(locs[s["place"]], 400).rstrip(".") + ". " if locs[s["place"]] else "")
                 + core._plain(s.get("SCENE") or s["ACTION"], 400).rstrip(".") + "." + NO_TEXT)
        if s.get("objects"):
            layouts.append(explicit_layout(scene, s["objects"], chars, gen_size))
        else:
            layouts.append(moss_layout(scene, [core._plain(chars[c], 400) + ", no text" for c in s["cast"]],
                                       s["ACTION"], gen_size, i))
        shot_seeds.append(int(seed) + i)
        line = core._plain(s["SAY"], 600) or "."
        lines.append(line)
        instructs.append(instruct)
        speakers.append(sp)
        cam = (s.get("CAMERA") or "").strip().lower()
        plan_shots.append({"line": line, "media_id": None, "media": None, "instruct": instruct, "image_prompt": p,
                           "framing": ("close" if "close" in s["ACTION"].lower() else "wide" if "wide" in s["ACTION"].lower() else "medium"),
                           "camera": CAMERAS.get(cam) or rng.choice(pre["camera"]), "seed": int(seed) + i,
                           "seconds": 6, "speaker": sp, "place": ids_l.index(s["place"]),
                           "cast": [ids_c.index(c) for c in s["cast"]]})
    est = sum(len(x.split()) for x in lines) / 2.6
    if est > 75:
        warnings.append(f"about {est:.0f} s of speech: long for the 180 s budget, trim lines")
    w, h = gen_size
    out_w, out_h = int(round(w * 1.25 / 2)) * 2, int(round(h * 1.25 / 2)) * 2
    title = head["TITLE"]
    plan = {"ok": True, "warnings": warnings, "headline": title, "voice": head["VOICE"], "shots": plan_shots,
            "n_chars": len(ids_c), "n_locs": len(ids_l), "char_ids": ids_c, "loc_ids": ids_l,
            "story": {"title": title or "Story", "link": "", "caption": int(caption),
                      "caption_mode": pre["caption"][str(int(caption))], "out_size": [out_w, out_h],
                      "gen_size": [w, h], "seed": int(seed)},
            "spare_image": False, "pad": 0.12}
    plan["layouts"] = layouts
    # cutout mode: characters on a flat green background (portrait), then one empty plate per shot
    flat = ", full body character reference, standing, facing the viewer, on a plain solid bright green background, flat even lighting, no shadow, no props"
    a_prompts = [look_s + core._plain(chars[c], 500) + flat + "." + NO_TEXT for c in ids_c]
    a_w = [min(w, h) // 32 * 32] * len(ids_c)
    a_h = [max(w, h) // 32 * 32] * len(ids_c)
    for i, s in enumerate(good):
        fr = "close" if "close" in (s["ACTION"] or "").lower() else "wide" if "wide" in (s["ACTION"] or "").lower() else "medium"
        view = {"close": "tight detail view of the background, shallow depth of field", "wide": "wide establishing view",
                "medium": "medium view"}[fr]
        a_prompts.append(look_s + core._plain(locs[s["place"]], 450).rstrip(".") + ", " + view
                         + ", empty scene with no people and no figures." + NO_TEXT)
        a_w.append(w)
        a_h.append(h)
    plan["assets"] = {"prompts": a_prompts, "seeds": [int(seed) + 100 + i for i in range(len(a_prompts))],
                      "w": a_w, "h": a_h}
    return plan, ref_prompts, ref_seeds, shot_prompts, shot_seeds, lines, instructs, speakers


QWEN_VOICES = {"ryan": "a clear, confident young man", "aiden": "a friendly, relaxed young man", "serena": "a warm young woman",
               "vivian": "a bright, lively young woman", "dylan": "a deep, gravelly older man", "eric": "a calm middle-aged man",
               "uncle_fu": "an old, wise, raspy man"}


def voice_script(text, default_desc="a warm, clear storyteller"):
    """Story script -> text for VoiceCast (Drama 3 + voice bank): {1:a warm clear storyteller}line {2:desc}line ...
    A shot's speaker is NARRATOR (default) or a key; the voice description comes from `VOICE key: ...`, else the
    character's own description, else a default. DELIVERY is not used (Drama 3 takes sound cues, not direction).
    Returns (cast_text, plan) with plan["lines"] = [{i, speaker (1-8), key, text}]."""
    chars, voices, shots, cur = {}, {}, [], None
    for k, ident, v in tokenize(text):
        if k == "SHOT":
            cur = {}
            shots.append(cur)
        elif k == "CHARACTER":
            chars.setdefault(ident, v)
        elif k == "VOICEOF":
            voices.setdefault(ident, v)
        elif cur is not None and k in ("SAY", "SPEAKER"):
            cur.setdefault(k, v)
    order, lines, parts, warnings = [], [], [], []
    for i, s in enumerate(shots):
        say = core._plain(s.get("SAY", ""), 600).replace("{", "").replace("}", "")
        if not say:
            continue
        key = (s.get("SPEAKER") or "narrator").strip().lower().replace(" ", "_") or "narrator"
        if key not in order:
            if len(order) >= 8:
                warnings.append(f"shot {i+1}: more than 8 voices, {key!r} spoken by {order[-1]!r}")
                key = order[-1]
            else:
                order.append(key)
        n = order.index(key) + 1
        first = not any(p[0] == key for p in parts)
        desc = ""
        if first:
            desc = (voices.get(key) or (core._plain(chars[key], 90) if key in chars else "") or QWEN_VOICES.get(key)
                    or default_desc)
            desc = re.sub(r"[{}:]", " ", desc).strip()
        parts.append((key, ("{%d:%s}" % (n, desc) if first else "{%d}" % n) + say))
        lines.append({"i": len(lines), "speaker": n, "key": key, "text": say})
    if not lines:
        raise ValueError("no SAY lines found in the script")
    return " ".join(p[1] for p in parts), {"lines": lines, "voices": order, "warnings": warnings}
