"""Stdlib-only RSS + article extraction and script-brief building (no torch, testable offline)."""
import html, json, random, re, struct, urllib.request
import xml.etree.ElementTree as ET
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin

HERE = Path(__file__).parent
UA = {"User-Agent": "Mozilla/5.0 (compatible; NewsFeedNode/1.0)"}
SKIP_LINK = re.compile(r"/(sounds|iplayer|programmes|av)/|/videos?/")
MIN_BODY = 600
MAX_SOURCE_CHARS = 6500


def load_json(name):
    return json.loads((HERE / name).read_text(encoding="utf8"))


def clean(s):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", s or ""))).strip()


def get(url, timeout=20):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def read_feed(url, category):
    root = ET.fromstring(get(url))
    out = []
    for it in root.iter("item"):
        link = it.findtext("link") or ""
        if not link or SKIP_LINK.search(link):
            continue
        out.append({"category": category, "title": clean(it.findtext("title")),
                    "summary": clean(it.findtext("description")), "link": link})
    return out


class ArticleParser(HTMLParser):
    SKIP = {"script", "style", "nav", "footer", "aside", "form", "noscript"}

    def __init__(self, base):
        super().__init__(convert_charrefs=True)
        self.base, self.paras, self.media = base, [], []
        self.skip, self.in_p, self.buf = 0, False, []
        self.fig, self.cap = None, None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        if tag in self.SKIP:
            self.skip += 1
        if tag == "figure":
            self.fig = {"imgs": [], "caption": ""}
        elif tag == "figcaption":
            self.cap = []
        elif tag == "p" and not self.skip:
            self.in_p, self.buf = True, []
        elif tag == "img" and not self.skip:
            src = a.get("src") or ""
            if not src or src.startswith("data:"):
                return
            w = int(a["width"]) if (a.get("width") or "").isdigit() else 0
            if 0 < w < 300:
                return
            item = {"url": urljoin(self.base, src), "alt": (a.get("alt") or "").strip(),
                    "after_para": len(self.paras), "caption": ""}
            (self.fig["imgs"] if self.fig is not None else self.media).append(item)

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        if tag == "p" and self.in_p:
            t = re.sub(r"\s+", " ", "".join(self.buf)).strip()
            if t:
                self.paras.append(t)
            self.in_p = False
        elif tag == "figcaption" and self.cap is not None and self.fig is not None:
            self.fig["caption"] = re.sub(r"\s+", " ", "".join(self.cap)).strip()
            self.cap = None
        elif tag == "figure" and self.fig is not None:
            for it in self.fig["imgs"]:
                it["caption"] = self.fig["caption"]
                self.media.append(it)
            self.fig = None

    def handle_data(self, d):
        if self.skip:
            return
        if self.in_p:
            self.buf.append(d)
        if self.cap is not None:
            self.cap.append(d)


def fetch_article(link):
    page = get(link, 25).decode("utf8", "replace")
    i, j = page.find("<article"), page.find("</article>")
    pp = ArticleParser(link)
    pp.feed(page[i:j] if 0 <= i < j else page)
    if not pp.paras:
        pp = ArticleParser(link)
        pp.feed(page)
    m = (re.search(r'property="og:image"\s+content="([^"]+)"', page)
         or re.search(r'content="([^"]+)"\s+property="og:image"', page))
    media, seen = [], set()
    for it in pp.media:
        key = re.sub(r"/(\d+)/", "/N/", it["url"])
        if key not in seen:
            seen.add(key)
            media.append(it)
    cut = 0.9 * len(pp.paras)  # trailing uncaptioned images are "related story" cards
    media = [it for it in media if it["caption"] or it["after_para"] < cut] or media
    if not media and m:
        media.append({"url": html.unescape(m.group(1)), "alt": "", "after_para": 0, "caption": ""})
    for n, it in enumerate(media):
        it["caption"] = re.sub(r"^(Image (caption|source),\s*)+", "", it["caption"]).strip()
        it["id"] = f"M{n+1}"
    return pp.paras, media


def pick_story(category, seed, tries=6):
    """Seeded pick among live feed items; skips stubs. Returns (item, paras, media)."""
    cfg = load_json("feeds.json")["feeds"]
    cat = (category or "").strip().lower()
    feeds = [f for f in cfg if cat in ("", "any", f["category"])]
    if not feeds:
        raise ValueError(f"unknown category '{category}'. have: {sorted({f['category'] for f in cfg})}")
    items, errs = [], []
    for f in feeds:
        try:
            items += read_feed(f["url"], f["category"])
        except Exception as e:
            errs.append(f"{f['category']}: {e}")
    if not items:
        raise RuntimeError("no feed items fetched: " + "; ".join(errs))
    items.sort(key=lambda x: x["link"])           # stable order so seed -> same pick while feed unchanged
    random.Random(seed).shuffle(items)
    for it in items[:tries]:
        try:
            paras, media = fetch_article(it["link"])
        except Exception:
            continue
        if len("\n".join(paras)) >= MIN_BODY:
            return it, paras, media
    raise RuntimeError(f"no usable article in {tries} tries ({'; '.join(errs)})")


SYSTEM_PROMPT = ("You write short-video scripts from news articles. "
                 "Respond with valid JSON only, no prose, no markdown fences.")


def build_brief(item, paras, media, seed, tone, visual, length, shot_seconds, caption):
    pre = load_json("presets.json")
    tn, vs = pre["tone"][str(tone)], pre["visual"][str(visual)]
    nshots = max(2, round(length / shot_seconds))
    words = int(length * pre["words_per_second"])
    rng = random.Random(seed)
    cams = [rng.choice(pre["camera"]) for _ in range(nshots)]

    src, used = [], 0
    for i, t in enumerate(paras):
        if used + len(t) > MAX_SOURCE_CHARS:
            break
        src.append(f"[P{i}] {t}")
        used += len(t)
    kept = len(src)
    mblock = "\n".join(
        f'  {m["id"]}: alt="{m["alt"]}" caption="{m["caption"]}" (next to paragraph {m["after_para"]})'
        for m in media if m["after_para"] <= kept) or "  (none)"
    user = (
        f"Write a {length}-second narration (~{words} words) for a short video.\n"
        f"TONE: {tn['script']}\n"
        "Use only facts from the source. Do not invent quotes, names or numbers.\n"
        f"AVAILABLE ARTICLE IMAGES:\n{mblock}\n"
        f'Output JSON: {{"narration": str, "shots": [exactly {nshots} x '
        '{"line": str, "media_id": str|null, "image_prompt": str|null}]}.\n'
        f"The shots' lines, in order, are the narration split into spoken parts, each about {max(8, words // nshots)} words. "
        "If a line is about what an article image shows (its subject, person or place), set media_id to that "
        "image and image_prompt to null. Never use an image for a line it does not depict; use each at most once. "
        "Otherwise media_id is null and image_prompt describes one still, no text in the image.\n\n"
        f"SOURCE HEADLINE: {item['title']}\nSOURCE TEXT:\n" + "\n".join(src) + "\n")
    story = {
        "title": item["title"], "link": item["link"], "category": item["category"],
        "media": media, "tone": tn["name"], "visual": vs["name"], "visual_prefix": vs["prefix"],
        "sfx": tn["sfx"], "caption_mode": pre["caption"][str(caption)], "caption": caption, "seed": seed,
        "length": length, "shot_seconds": shot_seconds, "cameras": cams, "nshots": nshots,
    }
    return SYSTEM_PROMPT, user, story


def parse_llm_json(text):
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    try:
        return json.loads(t)
    except Exception:
        a, b = t.find("{"), t.rfind("}")
        if a >= 0 and b > a:
            return json.loads(t[a:b + 1])
        raise


def salvage_llm_json(text):
    """Small models often emit almost-JSON. Try cheap repairs, then pull shots out with regexes."""
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    fixed = re.sub(r",\s*([}\]])", r"\1", t)                      # trailing commas
    fixed = re.sub(r"}\s*{", "},{", fixed)                        # missing comma between objects
    fixed = re.sub(r'"\s*\n\s*"', '",\n"', fixed)                 # missing comma between string fields
    fixed = re.sub(r'("|null|\d)\s+("(?:line|media_id|image_prompt)")', r"\1, \2", fixed)
    try:
        a, b = fixed.find("{"), fixed.rfind("}")
        return json.loads(fixed[a:b + 1])
    except Exception:
        pass
    S = r'"((?:[^"\\]|\\.)*)"'
    shots = []
    for m in re.finditer(r'"line"\s*:\s*' + S + r'(.*?)(?="line"|$)', t, re.S):
        rest = m.group(2)
        mid = re.search(r'"media_id"\s*:\s*(?:"(M\d+)"|null)', rest)
        ip = re.search(r'"image_prompt"\s*:\s*' + S, rest)
        shots.append({"line": m.group(1).replace('\\"', '"'),
                      "media_id": mid.group(1) if mid else None,
                      "image_prompt": ip.group(1).replace('\\"', '"') if ip else None})
    nar = re.search(r'"narration"\s*:\s*' + S, t)
    return {"narration": nar.group(1) if nar else "", "shots": shots}



def finish_script(llm_text, story):
    """Validate/repair the LLM output and merge with the story plan into one job result."""
    warnings = []
    try:
        d = parse_llm_json(llm_text)
    except Exception as e:
        d = salvage_llm_json(llm_text)
        if not d or not d.get("shots"):
            return {"ok": False, "error": f"LLM output was not JSON: {e}", "raw": llm_text[:2000], "story": story}
        warnings.append(f"JSON was malformed ({e}); salvaged {len(d['shots'])} shots")
    shots_in = d.get("shots") or []
    n = story["nshots"]
    if len(shots_in) != n:
        warnings.append(f"expected {n} shots, got {len(shots_in)}")
    byid = {m["id"]: m for m in story["media"]}
    used, shots = set(), []
    for i, s in enumerate(shots_in[:n]):
        mid = s.get("media_id")
        if mid in byid and mid not in used:
            used.add(mid)
        else:
            if mid:
                warnings.append(f"shot {i+1}: media_id {mid!r} unknown or reused, using generated still")
            mid = None
        prompt = (s.get("image_prompt") or "").strip()
        if mid is None:
            prompt = story["visual_prefix"] + (prompt or story["title"]) + ", no text, no letters, no captions, no logos"
        shots.append({"line": (s.get("line") or "").strip(), "media_id": mid,
                      "media": byid.get(mid), "image_prompt": prompt if mid is None else None,
                      "camera": story["cameras"][i] if i < len(story["cameras"]) else "static with subtle drift",
                      "seed": story["seed"] + i, "seconds": story["shot_seconds"]})
    narration = (d.get("narration") or " ".join(s["line"] for s in shots)).strip()
    return {"ok": bool(shots), "warnings": warnings,
            "narration": narration, "shots": shots,
            "story": {k: v for k, v in story.items() if k not in ("media", "cameras")}}


def encode_string_bytes(text):
    """Same layout as ComfyUI-TripoSG._encode_string_as_image: 4-byte BE length + UTF-8, 3 bytes/pixel."""
    b = text.encode("utf-8")
    payload = struct.pack(">I", len(b)) + b
    while len(payload) % 3:
        payload += b"\x00"
    return payload
