## Requirements and Installation

1. Download Mujoco Pro 2.00 and set up license. See requirements on [dm_control](https://github.com/deepmind/dm_control/tree/c4e0f4450aaedcedf7a2995ed058155dd7deb362#requirements-and-installation) buto dont pip install it.
2. Be in a python3.5 environment. If using conda `conda create -n mbrl python=3.5`
3. Clone this repo and install its requirements. `pip install -r requirments.txt`
4. Install dm_control. `cd src/dm-control && python setup.py install`
5. Validate install by going to root directory and running `python -c "from dm_control import suite"`
