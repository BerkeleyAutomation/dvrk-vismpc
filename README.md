# Da Vinci Code for Fabrics and Visual MPC

This is not for the original smoothing paper. For that, [see this code][1].

Use the following core API:

- `dvrkArm.py`: an API wrapping around the ROS messages to get or set positions.
- `dvrkClothSim.py` for moving the arms.


## Installation and Setup

We're using davinci-arm machine (formerly named davinci0).

We use a mix of the system Python 2.7 on there, and a Python 3.5 virtual env.
Unlike in the original smoothing paper (from fall 2019) the camera code has
been updated to use Python 3. However, the neural network code (as before)
still requires Python 3.

To install, first clone this repository. Then make a new Python 3.5 virtualenv
from a reference virtualenv on davinci0:

```
source ~/venv/bin/activate
```

I did `pip freeze` and put this in a file, and then made a new Python3 virtualenv:

```
virtualenv venvs/py3-vismpc-dvrk  --python=python3
```

and then did `pip install -r [filename].txt` to get all the packages updated.
After that, running `import zivid` should work, and then we later install
Tensorflow and other neural network code in that virtualenv.

## Experimental Usage

Performing our experiments involves several steps, due to loading a separate
neural network and having it run continuously. Roughly, the steps are:

1. Activate the robot via `roscore`, then (in a separate tab) run `./teleop` in
`~/catkin_ws` and click HOME. This gets the dvrk setup.

2. In another tab, *activate the Python 3 virtualenv above*, and run `python
load_net.py`. See `call_network/README.md` for detailed instructions.  This
script runs continuously in the background and checks for any new images in the
target directory.

3. In another tab, *activate the Python 3 virtualenv above*, and run `python
ZividCapture.py`. This script runs continuously and will activate with a
keyboard command.

4. Check the configuration file in `config.py` which will contain a bunch of
settings we can adjust. *In particular, adjust which calibration data we should
be using*. They are in the form of text files.

5. Finally, *in that same Python3 virtualenv*, run `python run.py` for
experiments, using the system Python. This requires some keyboard strokes. *The
code runs one episode* and then terminates.  Repeat for more episodes.

## Tips

- To test the camera code, just run `python ZividCapture.py` and adjust the
  main method. Must be in the Python3 virtualenv. The run script will do some
  similar image processing, but it should not be an issue if it's in Python2, I
  don't think image processing code changed that much.

- For quick testing of the da vinci with resorting to the machinery of the
  experimental pipeline, use `python dvrkClothSim.py` (for generic motion) or
  one of the test scripts in `tests/` (for calibration accuracy).


## Quick Start

Here's a minimal working example. Put this set of code in `tests/test_01_positions.py`:

```python
import sys
sys.path.append('.')
import dvrkArm

if __name__ == "__main__":
    p = dvrkArm.dvrkArm('/PSM1')
    pose_frame = p.get_current_pose_frame()
    pose_rad = p.get_current_pose()
    pose_deg = p.get_current_pose(unit='deg')

    print('pose_frame:\n{}'.format(pose_frame))
    print('pose_rad: {}'.format(pose_rad))
    print('pose_deg: {}'.format(pose_deg))

    targ_pos = [0.04845971,  0.05464517, -0.12231114]
    targ_rot = [4.65831872, -0.69974499,  0.87412989]
    p.set_pose(pos=targ_pos, rot=targ_rot, unit='rad')
```

and run `python tests/test_01_positions.py`. This will print out pose
information about the dvrk arm, which can then be used to hard code some
actions. For example, above we show how to set the pose of the arm given
pre-computed `targ_pos` and `targ_rot`.



[1]:https://github.com/BerkeleyAutomation/dvrk_python
