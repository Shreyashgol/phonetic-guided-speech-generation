# Phonetic Guided Speech Generation

Advanced ML / Deep Learning project: *Phonetic Guided Audio Generation via Alignment and Latent Diffusion*.
Compares explicit phoneme-duration alignment with a learned acoustic generator on LJSpeech.

## Status
- **Phase 1 (ML baseline):** in progress: alignment, duration EDA, features, duration predictor, metrics
- **Phase 2 (DL):** not started

## Dataset
[LJSpeech 1.1](https://keithito.com/LJ-Speech-Dataset/): not included in this repo.
Download `LJSpeech-1.1.tar.bz2` and extract it to `data/raw/LJSpeech-1.1/` (or point `configs/data.yaml` at it).

### Findings so far (notebook 01)
| Item | Value |
|---|---|
| Clips | 13,100 (all 22,050 Hz) |
| Total duration | 23.92 h |
| Clip length | min 1.11 s, mean 6.57 s, max 10.10 s |
| Vocabulary | 14,775 words (6,089 appear once) |
| Rows where raw != normalized text | 1,505 |
| Digits left in normalized text | 0 |

## Repo layout
`configs/` shared settings, `src/` reusable code, `notebooks/` EDA and experiments,
`data/` (not committed), `results/`, `reports/`, `tests/`.

## Audio config (shared by both phases)
22,050 Hz, n_fft 1024, hop 256, 80 mels, 0-8000 Hz. Do not change after alignments are cached.

## Notebooks
| # | Notebook | Status |
|---|---|---|
| 00 | environment and setup | done |
| 01 | dataset overview | done |
| 02 | text and phoneme prep | ready |
| 03 | MFA alignment and QA | ready |
| 04 | duration EDA | ready |
| 05 | feature engineering and frozen splits | ready |
| 06 | baseline and classical models | ready |
| 07 | neural (BiLSTM) duration model | ready |
| 08 | evaluation, error analysis, ablation | ready |
| 09 | report figures, tables, IEEE skeleton | ready |

## Team
See `CONTRIBUTIONS.md` (to be filled by each member).
