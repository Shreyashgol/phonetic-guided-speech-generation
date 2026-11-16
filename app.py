"""Standalone launcher (for a Hugging Face Space or a laptop). Paths can be overridden with environment variables."""
import os
from src.app.engine import TTSEngine
from src.app.gradio_app import build_demo

ROOT = os.path.dirname(os.path.abspath(__file__))
engine = TTSEngine(os.environ.get('PGSG_PROCESSED', f'{ROOT}/data/processed'),
                   os.environ.get('PGSG_CKPT', f'{ROOT}/checkpoints/acoustic/best.pt'),
                   os.environ.get('PGSG_HIFIGAN', f'{ROOT}/pretrained/hifigan'))
demo = build_demo(engine)
if __name__ == '__main__':
    demo.launch()
