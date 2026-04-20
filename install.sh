#!/bin/bash


# dir constants

TOKENS_DIR="user_tokens"
VENV_DIR="venv"
CHECKPOINT_DIR="checkpoints"
CACHE_DIR="cache_dir"
NLTK_DIR="nltk_data"

DIRS=("$TOKENS_DIR" "$VENV_DIR" "$CHECKPOINT_DIR" "$CACHE_DIR" "$NLTK_DIR")


#  spinner loader
run() {
	local verbose=0

#  parse flag
	if [[ "$1" == "-v" ]]; then
		verbose=1
		shift
	fi
	local msg="$1"
	shift
	printf "$msg "

#  flag handler
	if [[ $verbose -eq 1 ]]; then
        	"$@" &
    	else
		"$@" >/dev/null 2>&1 &
    	fi
	local pid=$!
	trap 'cleanup' INT
	local frames=("⟋" "─" "⟍" "─")
	local delay=0.15
    
#  loading animation
	local show_spinner=1
	[[ -z "$msg" ]] && show_spinner=0
	if [[ $show_spinner -eq 1 ]]; then
		while kill -0 "$pid" 2>/dev/null; do
            		for f in "${frames[@]}"; do
                		printf "\b%s" "$f"
                		sleep $delay
            		done
        	done
    	else
		wait "$pid"
    	fi
    	local exit_code=$?
    	trap - INT
    	printf "\b"
    	if [[ $exit_code -eq 0 ]]; then
    		printf "✔ \n"
    	else
    		printf "✖ \n"
    	fi
    	return $exit_code
}

# SIGTERM handler
cleanup() {
	printf "\n\n===== INSTALLATION ABORTED =====\n\n"

    # killing foreground process	
	kill -TERM "$pid" 2>/dev/null
	wait "$pid" 2>/dev/null

    # disable CTRL+C after aborting
	trap - INT

	tput cnorm

    # iterate through dirs for removal
	for dir in "${DIRS[@]}"; do
		printf "\n"
		read -r -p "Remove $dir? (y/N): " answer
		if [[ "$answer" == "y" || "$answer" == "Y" ]]; then
			printf "Removing %s...\n" "$dir"
			rm -rf "$dir"
			printf "Cleanup complete.\n"
		else
			printf "Keeping %s.\n" "$dir"
	    	fi
	done

	printf "\n\nDONE\n"

	exit 130
}


token_setup() {
		if [ ! -d "$TOKENS_DIR" ]; then
		printf "\n===== Token setup =====\n"
		mkdir -p "$TOKENS_DIR"
		TOKENS=("hf_token" "nagaai_token")
		for TOKEN in "${TOKENS[@]}"; do
			FILE="$TOKENS_DIR/$TOKEN"
		    	if [ ! -f "$FILE" ]; then
				printf "\nCreating $FILE\n"
				touch "$FILE"
			fi

		    	CONTENT=$(cat "$FILE")

		    	if [ -z "$CONTENT" ]; then
				printf "\nEnter $TOKEN: \n"
				read -r -p INPUT
				printf "$INPUT\n" > "$FILE"
		    	else
				printf "$TOKEN already set \n"
		    	fi
		done
		printf "\n===== Tokens ready =====\n\n"
		sleep 2.1
		#clear
		fi
}

main() {

#  TOKEN SETUP

	token_setup

#  INSTALLER MAIN
	#  restoring cursor on exit
	trap 'tput cnorm' EXIT

	tput civis
	printf "\n===== PyGPT Installer =====\n\n"

	run "Checking Python 3.11..." python3.11 --version

	if [ ! -d "$VENV_DIR" ]; then
		run "Creating virtual environment..." python3.11 -m venv venv
		
	else
		printf "Virtual enviroment already created...✔ \n"
			
	fi
	run "Upgrading pip, setuptools, wheel..." venv/bin/pip install --upgrade pip setuptools wheel

	if [ ! -f "checkpoints/vallex-checkpoint.pt" ]; then
	    run -v "" bash -c '
		mkdir -p checkpoints
		printf "\bDownloading Plachtaa/VALL-E-X checkpoint...\n"
		wget --show-progress --progress=bar:force:noscroll -q -O "checkpoints/vallex-checkpoint.pt" \
		"https://huggingface.co/Plachta/VALL-E-X/resolve/main/vallex-checkpoint.pt" \
	    '
	else
		printf "VALL-E-X checkpoint already exists...✔ \n"
	fi

	venv/bin/python -c "import torch" >/dev/null 2>&1
	if [ $? -ne 0 ]; then
		#printf "Downloading Torch CUDA...."
		run "Downloading PyTorch CUDA..." venv/bin/pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
	else
		run "Checking for PyTorch CUDA..." venv/bin/python -c "import torch" 
		printf "PyTorch CUDA already installed...✔ \n"
	fi

	run "Installing Python packages via requirements.txt..." venv/bin/pip install -r requirements.txt


	if [ ! -d "$CACHE_DIR" ]; then
		#printf "Downloading Speech-to-Text models...\n"
		run "Downloading Speech-to-Text models...\n" venv/bin/python preload_models.py
	else
		printf "Speech-to-Text models already downloaded...✔ "
	fi

	printf "\n\n===== INSTALLATION SUCCESSFUL ===== \n\n"

	tput cnorm
	read -p "Press enter to exit..."
}

#  script execution
#clear
main
