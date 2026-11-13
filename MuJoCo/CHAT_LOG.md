# Chat log: setting up the MuJoCo beam walker (13 Nov 2026)

A record of Mia's session with Claude Code that set up the MuJoCo work in this repo. Mia's messages are quoted, and
Claude's replies are condensed to what was done and found. API keys and usernames are left out on purpose.

---

### 1. "Is MuJoCo being downloaded to this folder now?"

No. Nothing was downloading, and MuJoCo was not installed in system Python or either repo venv. Another Claude session
had said it was "transferring", but nothing was running on the Mac.

### 2. Copy the MuJoCo files into ME193

> Copy all my MuJoCo files into my ME193 coursework repo as a new MuJoCo/ folder, on a new branch mujoco made from main,
> then commit and push. Don't touch the uncommitted Safe changes; use a git worktree.

- `MuJoCo/mujoco-hpc/` = the committed version of Chris's `Public stuff/mujoco-hpc` (`git archive`, so no logs or runs).
- `MuJoCo/ping-pong/` = the ping-pong sims. Their cluster run lines were changed to
  `python3 runIt.py --cpus 4 ../ping-pong/two_paddles.py` (run from `MuJoCo/mujoco-hpc/`).
- Checked for secrets (only placeholders), committed and pushed on branch `mujoco`.

### 3. Where are we on the Notion setup page?

Mia shared the class Notion page (steps: `config.sh`, `setup.py`, bouncing-ball demo, Onshape CAD, API key, `.env`,
`config.json`, `onshape-to-robot`, `runIt.py`). Status: setup and demo were done; the laptop venv, `.env` and `config.json`
were not.

> No, you are supposed to make a new folder in the current folder I am working in. I don't want things in Chris's folder.

A working copy of `MuJoCo/` was put in `~/Documents/GitHub/ME193/MuJoCo/`. From then on, all work happened there and
Chris's folder was left alone.

### 4. Keys, URL and first import

- Mia's Onshape API key (read-only) went into `09_onshape_robot/.env`, which is git-ignored.
- `config.json` got the Onshape assembly URL.
- Made the laptop venv `MuJoCo/mujoco-hpc/.venv` and installed `onshape-to-robot` 1.8.3.
- First import found both wheel joints but **3 root nodes**: two small parts were not attached to the robot, and one
  part had no material ("no dynamics").

### 5. Fixing the mates

> It is now 3 planar ones. Which should I make fastened?
>
> The issue is that if I make it fastened it is no longer joined to the faces I want it to be, hence the 3 planar mates.

`onshape-to-robot` ignores planar mates. A Fastened mate snaps two mate connectors together, which is why the part
jumped. Fix: make a mate connector **owned by the chassis** at the point where the part already sits, fasten the part's
own connector to it (so nothing moves), then delete the planar mates. Mia also renamed the parts (double motor, wheel,
leg, foot) and gave the feet a material. The import then gave **1 root node** and no warnings.

### 6. First run on the cluster

`python3 runIt.py --gpu --with 09_onshape_robot/onshape 09_onshape_robot/drive.py` ran. With both motors at 8 rad/s the
robot went backwards and drifted. A close-up local render showed why: the double motor lies flat with its shafts near one
end about 1.5 cm off the floor, and each beam is 63 mm long. **A full turn of a beam levers the body over.** The hand plan
spent 54% of its time upside down (video: `results/hand_plan_tumbles.mp4`).

### 7. The assignment

> Your goal here is to simulate a double motor with two beams instead of wheels in MuJoCo and use MJLab to develop a policy
> to drive straight. You can use the LEGO set or the UnoQ. The things I will be looking for in your writeup are: GitHub
> link, reflection questions, good layout, etc. A CAD drawing, MuJoCo video, and a description of the reward system and
> the overall policy. Rubric: there is a Notion page; there is more of a Notion page; you have a CAD drawing, video, and
> description (plus all the reflection questions); you got the simulation and reality to be somewhat close.

### 8. Making mjlab's drive-straight task work for the beam walker

Chris's `drive_task.py` was written for a wheeled robot with a body named `chassis`. Changes (details in `README.md`):
find the root body by position, recentre it on its centre of mass, compute the start height, `TARGET_SPEED` 0.04 m/s,
LEGO-like motor (stiff speed control, 0.15 N·m limit, joint friction 0.01 instead of 0.1).

A 20-iteration test on the cluster showed nearly every practice run ending in `tipped_over`. Because the other reward
terms are mostly penalties, flipping (which ends the run) scored better than walking. Local tests confirmed it: every
steady motor speed from 2 to 10 rad/s flips the robot, and the chassis friction setting does nothing (MuJoCo uses the larger
friction of the two surfaces, and the floor's is 1). Fix: an **`alive` reward** of +0.5 per step while upright.

### 9. Training result

4096 robots, 1000 iterations, about 8 minutes on a GPU node. From about iteration 100 every practice run lasts the full
10 s with no tipping, and the forward-speed reward reaches about 93% of its maximum.

Video of the trained policy (`results/policy_walk.mp4`): **0.37 m in 10 s, within 2 cm of the line, heading rocking about
±15°, never tips.** The policy file is `results/model_999.pt`.

### 10. Matching the real LEGO robot (next steps)

> How do we get the actual LEGO to then match this simulation?

1. Build it exactly like the CAD, including the angle between the left and right beams.
2. Run the same test on both (a fixed `drive.py` plan for a few seconds) and compare distance, sideways drift and whether
   it flips. Film the real robot from above next to the MuJoCo video.
3. Measure the real motor's top speed and stall torque and put them in `MAX_WHEEL_SPEED` and `WHEEL_TORQUE_LIMIT`.
4. Weigh the real robot (the simulated one is 137 g).
5. Running the policy itself on the robot needs heading and speed (IMU or camera). A simpler option is to record the
   motor speeds the policy chose in simulation and play them back on the real motors.

> It is the LEGO Education CS/AI kit double motor. The silly walk is run when the motor is spun by hand, so it technically
> just walks straight with funky legs instead of wheels. The CAD was modeled after the physical robot.

So the real robot walks straight without flipping, while the simulated one flips. That is the main sim-vs-real gap to
close next (speed and torque first, then beam angle, mass and friction; see `README.md`).

### 11. Committed and pushed

Everything above went to the `mujoco` branch on GitHub so Mia's partner can use it: the edited task, the imported robot,
the results (videos, policy, table), this log and a README. Keys, usernames, the venv and the cluster logs were left out.
