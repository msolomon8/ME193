# MuJoCo: beam walker and ping-pong

| Folder | What it is |
|---|---|
| [`mujoco-hpc/`](mujoco-hpc/) | Chris Rogers' toolkit for running MuJoCo and mjlab on the Tufts cluster from a laptop (setup steps in its [README](mujoco-hpc/README.md)). |
| [`mujoco-hpc/09_onshape_robot/`](mujoco-hpc/09_onshape_robot/) | **Our beam walker**: a LEGO double motor with a beam and foot on each side instead of wheels, imported from Onshape, with an mjlab policy that walks it in a straight line. |
| [`mujoco-hpc/09_onshape_robot/results/`](mujoco-hpc/09_onshape_robot/results/) | Videos, the trained policy and its walking table. |
| [`ping-pong/`](ping-pong/) | One- and two-paddle ping-pong simulations. |
| [`CHAT_LOG.md`](CHAT_LOG.md) | How this was set up and debugged, step by step. |

## Results

| | What happens | Video |
|---|---|---|
| Hand plan (`drive.py`: both motors at 8 rad/s, then spin, then curve) | The 63 mm beams lever the body over: upside down 54% of the time, gets nowhere | [`hand_plan_tumbles.mp4`](mujoco-hpc/09_onshape_robot/results/hand_plan_tumbles.mp4) |
| Trained policy (mjlab, 4096 robots, 1000 iterations, about 8 min on a GPU) | Walks 0.37 m in 10 s, stays within 2 cm of the line, heading rocks about ±15°, never tips | [`policy_walk.mp4`](mujoco-hpc/09_onshape_robot/results/policy_walk.mp4) |

Second-by-second position of the trained policy: [`policy_walk_table.txt`](mujoco-hpc/09_onshape_robot/results/policy_walk_table.txt).

## The reward and the policy

The task is in [`drive_task.py`](mujoco-hpc/09_onshape_robot/mjlab_task/drive_task.py). Each 10 s practice run starts the robot at the
origin, pointing up to 0.4 rad off the +x axis.

- **Sees** (13 numbers): body velocity (3), turning rate (3), heading as sin/cos plus sideways distance from the line (3),
  the two motor speeds (2) and its last command (2).
- **Does**: a neural network (two layers of 64) outputs one speed per motor, up to ±10 rad/s, 50 times a second.
- **Reward per step**:

| Term | Weight | Meaning |
|---|---|---|
| `forward_speed` | +1.0 | 1 at the target speed (0.04 m/s) along +x, 0 standing still, negative going backwards |
| `stay_on_line` | +1.0 | minus the square of the sideways distance from y = 0 |
| `face_forward` | +0.5 | minus the square of the heading error |
| `action_rate` | -0.05 | penalty for changing the motor commands suddenly |
| `alive` | +0.5 | 1 for every step the robot is still upright |

A run ends early if the robot tilts more than 60°, or wanders 1.5 m off the line.

The `alive` term was added for this robot. Without it, almost every term is a penalty, so flipping over (which ends the
run) scored better than walking, and the first test run tipped in nearly every episode.

## Changes from Chris's version of `drive_task.py`

- Finds the root body by position, not by the name `chassis` (ours is `double_motor`).
- Moves the body origin onto the chassis' centre of mass. Onshape put it about 10 cm to the side, so turning looked like
  drifting off the line.
- Works out the start height from the legs instead of a fixed 6 cm.
- `TARGET_SPEED` 0.04 m/s (was 0.5; this walker does about 0.04 m/s).
- Motor: `WHEEL_KD` 0.5, `WHEEL_TORQUE_LIMIT` 0.15 N·m (a stiff speed-controlled LEGO motor), joint friction 0.01 N·m
  (onshape-to-robot writes 0.1, which stalls the legs below about 2 rad/s).

Note: setting the chassis friction low (as Chris's code does) has no effect, because MuJoCo uses the larger friction of the
two touching surfaces and the floor's is 1.

## Getting started (partner)

On your laptop, in `MuJoCo/mujoco-hpc/`:

1. Put your Tufts username in `config.sh`, then `python3 setup.py` (password and Duo once).
2. To re-import the CAD: `python3 -m venv .venv`, `source .venv/bin/activate`, `pip install onshape-to-robot`, then
   `cp 09_onshape_robot/.env.example 09_onshape_robot/.env` and put **your own** Onshape API keys in it (never commit
   `.env`). Then `cd 09_onshape_robot && onshape-to-robot onshape`. The imported robot is already committed in
   `09_onshape_robot/onshape/`, so you can skip this until the CAD changes.
3. Train: see [`mjlab_task/README.md`](mujoco-hpc/09_onshape_robot/mjlab_task/README.md), with
   `--env.scene.num-envs 4096 --agent.max-iterations 1000` and `--time 00:20:00`.
4. Video of a policy: the `make_video.py` command in the same README. The committed policy is
   `09_onshape_robot/results/model_999.pt`.

## Sim vs. real: the big gap so far

The real robot (LEGO Education Computer Science & AI kit double motor, CAD modelled on the physical build) **walks
straight without flipping** when the motor is turned by hand. The simulated robot flips over on any full beam turn. So
something in the simulation is off. Things to check, most likely first:

- **Speed and torque.** Turning by hand is slow and gentle. In the simulation the motor is a stiff 0.15 N·m speed
  controller, which can lever the body over. Try a slow hand-turn speed (about 1 to 2 rad/s, a smooth turn) in
  `drive.py` with a lower torque limit, and measure the real motor's top speed and stall torque.
- **Beam angle.** The angle between the left and right beams in the CAD must match the physical robot exactly.
- **Mass.** Weigh the real robot (the simulation says 137 g, and the battery or hub may be missing in the CAD) and check
  where its weight sits.
- **Friction.** Real LEGO plastic on a table slips more than the simulation's default friction of 1.

## Still to do

- Close the gap above, then retrain and run the same test (same speed, same time) on the real robot and the simulation:
  compare distance, sideways drift and tipping.
