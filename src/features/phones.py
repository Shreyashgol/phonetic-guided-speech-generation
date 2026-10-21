"""Shared phoneme inventory (ARPAbet, as used by MFA english_us_arpa). Used by Phase 1 and Phase 2."""
import re

VOWELS = ['AA', 'AE', 'AH', 'AO', 'AW', 'AY', 'EH', 'ER', 'EY', 'IH', 'IY', 'OW', 'OY', 'UH', 'UW']
DIPHTHONGS = ['AW', 'AY', 'EY', 'OW', 'OY']
CONSONANT_CLASS = {
    'stop': ['B', 'D', 'G', 'K', 'P', 'T'],
    'affricate': ['CH', 'JH'],
    'fricative': ['DH', 'F', 'HH', 'S', 'SH', 'TH', 'V', 'Z', 'ZH'],
    'nasal': ['M', 'N', 'NG'],
    'liquid': ['L', 'R'],
    'glide': ['W', 'Y'],
}
SPECIAL = ['sil', 'spn']          # silence, unknown-word noise
PAD = '<pad>'

BASE_PHONES = VOWELS + [p for v in CONSONANT_CLASS.values() for p in v]
PHONE_LIST = [PAD] + sorted(BASE_PHONES) + SPECIAL
PHONE2ID = {p: i for i, p in enumerate(PHONE_LIST)}

_CLASS = {p: 'vowel' for p in VOWELS}
for _c, _ps in CONSONANT_CLASS.items():
    for _p in _ps:
        _CLASS[_p] = _c
_CLASS['sil'] = 'sil'
_CLASS['spn'] = 'spn'
CLASS_LIST = ['vowel', 'stop', 'affricate', 'fricative', 'nasal', 'liquid', 'glide', 'sil', 'spn']
CLASS2ID = {c: i for i, c in enumerate(CLASS_LIST)}


def split_stress(label):
    """'AH0' -> ('AH', 0); 'T' -> ('T', -1); '' -> ('sil', -1); 'spn' -> ('spn', -1)."""
    if label == '' or label is None:
        return 'sil', -1
    if label.lower() in ('sil', 'sp'):
        return 'sil', -1
    if label.lower() == 'spn':
        return 'spn', -1
    m = re.match(r'^([A-Z]+?)([012])?$', label)
    if not m:
        return label, -1
    return m.group(1), (int(m.group(2)) if m.group(2) is not None else -1)


def phone_class(base):
    return _CLASS.get(base, 'unknown')
