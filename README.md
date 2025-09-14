# SafeRL-Trf
 
### Installation ###
1. Create a new python virtual env with python 3.8 
2. Install pytorch 2.1.2 with cuda-12.1:
    - `pip install torch==2.1.2+cu121 torchvision==0.16.2+cu121 torchaudio==2.1.2+cu121 --extra-index-url https://download.pytorch.org/whl/cu121`
3. Install Isaac Gym
   - Download and install Isaac Gym Preview 4 from https://developer.nvidia.com/isaac-gym
   - `cd isaacgym/python && pip install -e .`
   - Try running an example `cd examples && python 1080_balls_of_solitude.py`
   - For troubleshooting check docs `isaacgym/docs/index.html`)
4. Install rsl_rl (PPO implementation)
   -  `cd rsl_rl && pip install -e .`
   -  * some problems exists, if this does not work, then download rsl_rl from github and git checkout v1.0.2
5. Install legged_gym
    - Clone this repository
   - `cd legged_gym && pip install -e .`
6. Install Tensorboard
   - pip install tensorboard
7. Install Positional Encodings
   - pip install positional_encodings

### Before Running ###
1. Set the code environment:
   export PYTHONPATH=/home/Rain/Rain_tmp/BodyTransformer/a1_walk/rsl_rl:$PYTHONPATH
   export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH (If using Conda)

2. If you are using numpy>=1.20, just replace np.float in the isaacgym/python/isaacgym/torch_utils.py into float.

