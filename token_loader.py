    """
    Token loader utility for pygpt.

    Loads token from user_tokens/<name>
    Example: load_token("hf_token")
    """
from pathlib import Path

TOKENS_DIR = Path("user_tokens")

def load_token(name: str, default: str = None) -> str:
    file_path = TOKENS_DIR / name

    if not file_path.exists():
        if default is not None:
            return default
        raise FileNotFoundError(f"Token file not found: {file_path}")

    return file_path.read_text(encoding="utf-8").strip()

