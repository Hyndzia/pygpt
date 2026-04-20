from pathlib import Path
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from token_loader import load_token
import os
import nltk


HF_TOKEN = load_token("hf_token")

os.environ["HF_TOKEN"] = HF_TOKEN

cache_dir = Path("cache_dir")
cache_dir.mkdir(exist_ok=True, parents=True)

print("\n")

nltk.download('punkt', download_dir="nltk_data")

models = [
    "alexcleu/wav2vec2-large-xlsr-polish",
    "jonatasgrosman/wav2vec2-large-xlsr-53-english"
]

for model in models:
    print(f"Downloading {model} ...")
    Wav2Vec2Processor.from_pretrained(model, cache_dir=cache_dir)
    Wav2Vec2ForCTC.from_pretrained(model, cache_dir=cache_dir)

print("Downloading models DONE")
