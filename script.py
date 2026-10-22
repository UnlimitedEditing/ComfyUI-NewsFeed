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
                 r"TITLE:|VOICE:|LOOK:|SAY:|DELIVERY:|PLACE:|CAST:|ACTION:|SPEAKER:|CAMERA:)(?:\*\*)?")
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
        elif cur is not None:
            cur[k] = v if k not in cur else cur[k] + " " + v
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
        scene = (look_s + core._plain(locs[s["place"]], 400).rstrip(".") + ". " + core._plain(s["ACTION"], 300).rstrip(".")
                 + "." + NO_TEXT)
        layouts.append(moss_layout(scene, [core._plain(chars[c], 400) + ", no text" for c in s["cast"]],
                                   s["ACTION"], gen_size, i))
        shot_seeds.append(int(seed) + i)
        line = core._plain(s["SAY"], 600) or "."
        lines.append(line)
        instructs.append(instruct)
        speakers.append(sp)
        cam = (s.get("CAMERA") or "").strip().lower()
        plan_shots.append({"line": line, "media_id": None, "media": None, "instruct": instruct, "image_prompt": p,
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
    return plan, ref_prompts, ref_seeds, shot_prompts, shot_seeds, lines, instructs, speakers
