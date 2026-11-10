[ Important info! ]

Hardware Requirements:
- NVIDIA GPU, at least GTX 1060 6GB or better.
- 10 GB Free disk space. 


Total installation size:

	[ 9.69 GB ]


Software Requirements:

- Latest Nvidia CUDA [ https://developer.nvidia.com/cuda-downloads?target_os=Windows&target_arch=x86_64&target_version=10&target_type=exe_local ].
- Python -3.11 [ https://www.python.org/downloads/release/python-3110/ ].



HOWTO START:

1. Open installer script that matches your system (Linux/Windows).
	[ install.sh | install.bat ]	



2. Paste your API user tokens from:

 - https://huggingface.co/
 - https://naga.ac/


		
3. After the installation starting script that matches your system (Linux/Windows).
	[ start.sh | start.bat ]	


ADDITIONAL:

[ For voice recording set the microphone you're using as the default device ]


VOICE ICONS:

Every VALL-E X voice has its own icon in the assets/ folder, named exactly
like the preset that speaks (for example assets/paimon.png, assets/bronya.png).

- To change a voice icon, replace that file with your own picture.
  A PNG with a transparent background works best (a cut-out character/photo).
  Plain JPG/PNG without transparency is cropped into a circle automatically.
- New presets added to presets/ get a placeholder icon automatically:
      venv\Scripts\python.exe tools\make_avatars.py
  Existing icons are not overwritten (add --force to regenerate them).

The icon is shown in the launcher next to the voice list, in the chat header
and next to every message of that voice.

