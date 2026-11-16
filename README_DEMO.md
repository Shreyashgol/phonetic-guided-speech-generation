# Demo app: running and hosting

## In Colab (what we use for the viva)
Run notebook 17 top to bottom. The last cell prints a public link (valid while the cell runs).

## Locally
1. `pip install -r requirements_demo.txt`
2. Folder layout next to `app.py`:
   - `src/` (the repo's `src` folder)
   - `data/processed/` with `phase2_contract.json`, `sil_stats.json`, `mel_stats.json`, `phase2_manifest.csv`
   - `checkpoints/acoustic/best.pt`
3. `python app.py` and open the printed address. (HiFi-GAN downloads on first run into `pretrained/hifigan`.)

## Permanent link (free Hugging Face Space)
1. Create a Space (SDK: Gradio, hardware: CPU basic) on huggingface.co.
2. Upload: `app.py`, `requirements.txt` (copy of `requirements_demo.txt`), `src/`, the four files from `data/processed/`, and `checkpoints/acoustic/best.pt` (about 35 MB).
3. The Space builds and serves the same app. CPU speech takes a few seconds per sentence.

## Ideas for later
- Drive the durations with the Phase 1 BiLSTM (expose the Phase 1 feature builder as a function and add it to the `SOURCES` list in `engine.py`).
- Add pitch and energy predictors (full FastSpeech 2) and sliders for them.
- Add a second speaker or a speaking-rate control learned from data.
