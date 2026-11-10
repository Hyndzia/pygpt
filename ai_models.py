
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TOKEN_FILE = os.path.join(BASE_DIR, "user_tokens", "nagaai_token")
API_URL = "https://api.naga.ac/v1/models"

#: embedded list
FALLBACK = [
    ("nemotron-3.5-lightning:free", "Nemotron 3.5 Lightning"),
    ("nemotron-3-ultra-550b-a55b:free", "Nemotron 3 Ultra"),
    ("nemotron-3.5-super-vl:free", "Nemotron 3.5 Super VL"),
    ("llama-3.3-70b-instruct:free", "Llama 3.3 70B Instruct"),
    ("xing4.0-29b-a4b:free", "Xing 4.0 29B"),
    ("flint-v1.0:free", "Flint"),
    ("dots-3-note-preview:free", "Dots3 Note"),
    ("lfm-2.5-2.6b:free", "LFM 2.5 2.6B"),
    ("sonar:free", "Sonar"),
]

DEFAULT_MODEL = "nemotron-3.5-lightning:free"

# non-chat models
_NOT_CHAT = (
    "tts", "whisper", "embed", "rerank", "guard", "eleven",
    "audio", "speech", "voice", "image", "dall", "flux", "video",
)


def _is_chat(model_id: str) -> bool:
    mid = model_id.lower()
    if not mid.endswith(":free"):
        return False
    return not any(word in mid for word in _NOT_CHAT)


def _clean_label(raw: str, model_id: str) -> str:
    text = (raw or model_id).strip()
    for junk in ("(free)", "(Free)", "Free", "(FREE)"):
        text = text.replace(junk, "")
    text = " ".join(text.split())
    return text or model_id.split(":")[0]


def _http_models():
    try:
        import requests
    except ImportError:
        return None

    try:
        with open(TOKEN_FILE, "r", encoding="utf-8") as handle:
            token = handle.read().strip()
    except OSError:
        return None
    if not token:
        return None

    try:
        response = requests.get(
            API_URL,
            headers={
                "Authorization": "Bearer " + token,
                "Accept": "application/json",
                "User-Agent": "PyGPT",
            },
            timeout=8,
        )
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def fetch_free_models():
    payload = _http_models()
    if not payload:
        return []

    found = []
    for item in payload.get("data", []):
        model_id = str(item.get("id", ""))
        if not _is_chat(model_id):
            continue
        found.append((model_id, _clean_label(item.get("display_name", ""), model_id)))

    unique = {}
    for model_id, name in found:
        unique[model_id] = name
    return sorted(unique.items(), key=lambda pair: pair[1].lower())


def choices(models=None):
    source = models if models else FALLBACK
    used = {}
    result = []
    for model_id, name in source:
        label = name
        if label in used:
            label = "{} ({})".format(name, model_id.split(":")[0])
        used[label] = model_id
        result.append((label, model_id))
    return result


def label_for(model_id: str) -> str:
    for mid, name in FALLBACK:
        if mid == model_id:
            return name
    return model_id.split(":")[0].replace("-", " ").title()


__all__ = [
    "API_URL",
    "DEFAULT_MODEL",
    "FALLBACK",
    "choices",
    "fetch_free_models",
    "label_for",
]
