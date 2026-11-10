"""Głosy VALL-E X dostępne w projekcie (pliki `presets/*.npz`).

Moduł nie ładuje modelu — tylko odczytuje nazwy presetów, dzięki czemu
lista głosów w GUI jest zawsze zgodna z tym, co leży na dysku.
"""

import glob
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRESETS_DIR = os.path.join(BASE_DIR, "presets")
ASSETS_DIR = os.path.join(BASE_DIR, "assets")

#: Rozszerzenia, jakie może mieć ikona głosu (kolejność ma znaczenie).
IMAGE_EXTS = (".png", ".webp", ".jpg", ".jpeg", ".gif")

#: Wartość oznaczająca syntezator systemowy Google (gTTS) zamiast VALL-E X.
GTTS = "gtts"

# Ładniejsze nazwy dla presetów, które ich nie mają.
_LABELS = {
    "acou_1": "ACOU 1",
    "acou_2": "ACOU 2",
    "acou_3": "ACOU 3",
    "acou_4": "ACOU 4",
    "babara": "Barbara",
    "bronya": "Bronya",
    "dingzhen": "Ding Zhen",
    "fuxuan": "Fu Xuan",
    "librispeech_1": "LibriSpeech 1",
    "librispeech_2": "LibriSpeech 2",
    "librispeech_3": "LibriSpeech 3",
    "librispeech_4": "LibriSpeech 4",
    "paimon": "Paimon",
    "rosalia": "Rosalia",
    "seel": "Seele",
    "vctk_1": "VCTK 1",
    "vctk_2": "VCTK 2",
    "vctk_3": "VCTK 3",
    "vctk_4": "VCTK 4",
    "yaesakura": "Yae Sakura",
}

# Kolejność, w jakiej głosy mają się pojawiać (postacie najpierw).
_PRIORITY = [
    "paimon", "bronya", "fuxuan", "dingzhen", "babara", "yaesakura",
    "rosalia", "seel", "alan", "esta", "cafe",
    "vctk_1", "vctk_2", "vctk_3", "vctk_4",
    "librispeech_1", "librispeech_2", "librispeech_3", "librispeech_4",
    "acou_1", "acou_2", "acou_3", "acou_4",
    "en2zh_tts_1", "en2zh_tts_2", "en2zh_tts_3", "en2zh_tts_4",
    "zh2en_tts_1", "zh2en_tts_2", "zh2en_tts_3", "zh2en_tts_4",
]


def voice_ids():
    """Wszystkie identyfikatory głosów (nazwy plików bez `.npz`)."""
    paths = glob.glob(os.path.join(PRESETS_DIR, "*.npz"))
    return sorted(os.path.splitext(os.path.basename(p))[0] for p in paths)


def label(voice_id: str) -> str:
    """Czytelna nazwa głosu."""
    if voice_id in _LABELS:
        return _LABELS[voice_id]
    if voice_id == GTTS:
        return "gTTS (system)"
    return voice_id.replace("_", " ").title()


def available(voice_id: str) -> bool:
    return os.path.exists(os.path.join(PRESETS_DIR, voice_id + ".npz"))


def preset_path(voice_id: str):
    return os.path.join(PRESETS_DIR, voice_id + ".npz")


def image_path(voice_id: str):
    """Ścieżka do ikony głosu w `assets/` albo `None`.

    Wystarczy wrzucić do `assets/` plik o nazwie presetu (np. `paimon.png`),
    żeby pojawił się przy wiadomościach — najlepiej PNG z przezroczystym tłem.
    """
    base = os.path.join(ASSETS_DIR, voice_id)
    for ext in IMAGE_EXTS:
        candidate = base + ext
        if os.path.exists(candidate):
            return candidate
    return None


def choices():
    """Lista `(etykieta, identyfikator)` gotowa do comboboxa.

    Na początku gTTS, potem znane postacie w ustalonej kolejności,
    a na końcu wszystko, co leży w `presets/` i nie ma własnej etykiety.
    """
    ids = voice_ids()
    ordered = [v for v in _PRIORITY if v in ids]
    ordered += [v for v in ids if v not in ordered]

    result = [(label(GTTS), GTTS)]
    result += [(label(v), v) for v in ordered]
    return result
