"""Cut one stitched narration into per-shot segments (numpy only, testable offline)."""
import numpy as np

QUIET = 1e-6        # window peak below this = digital silence (story-voice puts 0.3 s of exact zeros between every line)
MIN_GAP = 0.2       # seconds of silence that can separate two lines
LEAD = 0.05         # the next picture appears this long before its line starts


def split_audio(samples, sr, texts):
    """samples: mono float array; texts: the spoken line of each shot. Returns (segments, how) where segments is a list
    of len(texts) sample arrays that concatenate to the (trimmed) narration. Lines are separated at the longest silences;
    when there are too few silences the narration is cut in proportion to the words in each line."""
    n = len(texts)
    x = np.asarray(samples, dtype=np.float32).reshape(-1)
    if n <= 1 or len(x) < sr:
        return [x] * 1 if n <= 1 else _proportional(x, texts), "single"
    win = max(1, int(0.01 * sr))
    nw = len(x) // win
    peak = np.abs(x[:nw * win].reshape(nw, win)).max(axis=1)
    quiet = peak < QUIET
    runs, i = [], 0
    while i < nw:
        if quiet[i]:
            j = i
            while j < nw and quiet[j]:
                j += 1
            runs.append((i, j))
            i = j
        else:
            i += 1
    need = int(MIN_GAP / 0.01)
    start, end = 0, len(x)
    if runs and runs[0][0] == 0:
        start = min(runs[0][1] * win, len(x))
        runs = runs[1:]
    if runs and runs[-1][1] >= nw:
        end = runs[-1][0] * win
        runs = runs[:-1]
    cands = [r for r in runs if r[1] - r[0] >= need]
    if len(cands) < n - 1:
        return _proportional(x[start:end], texts), "proportional"
    best = sorted(sorted(cands, key=lambda r: -(r[1] - r[0]))[:n - 1])
    cuts = [start] + [max(start, r[1] * win - int(LEAD * sr)) for r in best] + [end]
    return [x[cuts[k]:cuts[k + 1]] for k in range(n)], "silence"


def _proportional(x, texts):
    w = np.array([max(1, len(t.split())) for t in texts], dtype=np.float64)
    edges = np.concatenate([[0], np.cumsum(w) / w.sum()]) * len(x)
    return [x[int(edges[k]):int(edges[k + 1])] for k in range(len(texts))]
