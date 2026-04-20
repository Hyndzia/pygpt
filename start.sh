#!/bin/bash

#  pygpt starting script

#  check if properly installed
isInstalled() {

	if [ ! -d "venv" ]; then
	    printf "Python 3.11 venv not found. Run install.sh first. \n"
	    exit 1
	fi

	if [ ! -d "checkpoints" ]; then
	    printf "Checkpoints directory not found. Run install.sh first. \n"
	    exit 1
	fi

	if [ ! -d "user_tokens" ]; then
	    printf "User tokens directory not found. Run install.sh first. \n"
	    exit 1
	fi
}

isInstalled
source venv/bin/activate

printf "Launching PyGPT!\n"
python pygpt.py

#  exit venv after launch
deactivate
