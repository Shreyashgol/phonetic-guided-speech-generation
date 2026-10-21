"""Parse MFA long-format TextGrids into phone-level rows (seconds and mel frames)."""
import re
import os

from src.features.phones import split_stress

_ITEM = re.compile(r'item \[\d+\]:')
_NAME = re.compile(r'name = "([^"]*)"')
_IV = re.compile(r'xmin = ([-\d.eE+]+)\s*\n\s*xmax = ([-\d.eE+]+)\s*\n\s*text = "((?:[^"]|"")*)"')


def read_textgrid(path):
    """Return {tier_name: [(xmin, xmax, text), ...]} for a long-format TextGrid."""
    with open(path, encoding='utf-8') as f:
        txt = f.read()
    tiers = {}
    for part in _ITEM.split(txt)[1:]:
        m = _NAME.search(part)
        if not m:
            continue
        tiers[m.group(1)] = [(float(a), float(b), t.replace('""', '"')) for a, b, t in _IV.findall(part)]
    return tiers


def textgrid_to_rows(path, utt_id, sr=22050, hop=256):
    """One row per phone interval. Frames use cumulative rounding so boundaries stay consistent."""
    tiers = read_textgrid(path)
    phones, words = tiers['phones'], tiers['words']
    word_ivs = [(a, b, t) for a, b, t in words]
    # rank of each non-empty word interval
    rank, r = {}, 0
    for i, (_, _, t) in enumerate(word_ivs):
        if t.strip() != '':
            rank[i] = r
            r += 1
    rows = []
    for k, (a, b, lab) in enumerate(phones):
        mid = (a + b) / 2
        w_i = next((i for i, (wa, wb, _) in enumerate(word_ivs) if wa <= mid < wb), None)
        base, stress = split_stress(lab)
        is_word = w_i is not None and w_i in rank
        sf, ef = int(round(a * sr / hop)), int(round(b * sr / hop))
        rows.append(dict(
            utt_id=utt_id, phone_idx=k, label=lab, base=base, stress=stress,
            word_idx=rank[w_i] if is_word else -1,
            word=word_ivs[w_i][2] if is_word else '',
            start_s=a, end_s=b, dur_s=b - a, start_f=sf, end_f=ef, dur_f=ef - sf))
    return rows
