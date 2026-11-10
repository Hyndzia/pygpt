#!/bin/bash

echo
echo "== Token setup =="

TOKENS_DIR="user_tokens"
mkdir -p "$TOKENS_DIR"

TOKENS=("hf_token" "nagaai_token")

for TOKEN in "${TOKENS[@]}"; do
    FILE="$TOKENS_DIR/$TOKEN"

    if [ ! -f "$FILE" ]; then
        echo "Creating $FILE"
        touch "$FILE"
    fi
	
    CONTENT=$(cat "$FILE")

    if [ -z "$CONTENT" ]; then
        echo
        echo "Enter $TOKEN:"
        read -r INPUT

        echo "$INPUT" > "$FILE"
    else
        echo "$TOKEN already set"
    fi
done

echo "Tokens ready."
echo

echo "== PyGPT Installer =="

python3.11 --version
if [ $? -ne 0 ]; then
    echo "Python 3.11 missing"
    read -p "Press enter to exit..."
    exit 1
fi

if [ -d "venv" ] && [ -f "venv/bin/activate" ]; then
    echo "venv already exists!"
else
    echo "Creating venv..."
    python3.11 -m venv venv
fi

echo "Activating venv..."
source venv/bin/activate

python -m pip install --upgrade pip setuptools wheel

CHECKPOINT_DIR="checkpoints"

if [ ! -d "$CHECKPOINT_DIR" ]; then
    mkdir -p "$CHECKPOINT_DIR"
fi

if [ ! -f "checkpoints/vallex-checkpoint.pt" ]; then
    echo "Downloading checkpoint..."

    wget -O "checkpoints/vallex-checkpoint.pt" \
    "https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt"
else
    echo "checkpoint exists"
fi

python -c "import torch" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing torch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
fi

if [ -f "requirements.txt" ]; then
    pip install -r requirements.txt
fi

echo Preloading Speech-to-Text models into cache_dir...
python preload_models.py

echo "DONE"
read -p "Press enter to exit..."