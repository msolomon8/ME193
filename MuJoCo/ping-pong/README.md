# Ping-pong returns

**The task:** can you hit the ball with a paddle just so, so that the ball's apex is 1.2 m from the ground? The paddle rests at 0.5 m off the ground.

[`ping_pong_paddle.py`](ping_pong_paddle.py) is a MuJoCo simulation of a red ball and a paddle. The paddle learns from a reward how to keep the ball peaking at 1.2 m while keeping it over the middle. [`demo.mp4`](demo.mp4) shows the paddle before, during and after learning, with the reward on screen.

## Run it

On a laptop (no GPU needed, about 80 s):

```
pip install mujoco imageio imageio-ffmpeg
python ping_pong_paddle.py
```

On the Tufts cluster, from `MuJoCo/mujoco-hpc/`:

```
python3 runIt.py ../ping-pong/ping_pong_paddle.py
```

## How the paddle works

Each rally starts with the ball at a random spot, drifting sideways, from a random height. The ball is soft (restitution about 0.5), so a still paddle would let it die; the paddle has to swing.

Before every hit, the paddle predicts where, when and how fast the ball will arrive at 0.5 m. Then it:

1. **slides** under that spot, so it moves with the ball;
2. **tilts** its face by `c_pos × offset + c_vel × sideways speed` to steer the ball back to the middle;
3. **pulls back** and **swings up** through 0.5 m at `u0 + k × (arrival speed − 3.64 m/s)`, where 3.64 m/s is how fast a ball that peaked at exactly 1.2 m arrives.

The four numbers `u0`, `k`, `c_pos` and `c_vel` are what it learns.

## The reward

For each bounce: `−(apex − 1.2 m)² − 0.5 × (distance from the middle)²`. A missed ball counts as an apex of 0 m. Higher is better; 0 is perfect.

The learning method is the cross-entropy method. It tries 32 sets of the four numbers, keeps the best 6, fits a new guess to them, and repeats 9 times.

## Results

Same rally (dropped off-center, drifting sideways) at three stages:

| Stage | Apex heights (m) | Off-center (cm) |
|---|---|---|
| Before learning | 0.73 0.58 0.54 0.53 0.52 0.52 | 1 7 11 13 14 14 |
| Learning (iteration 2) | 0.94 0.82 0.83 0.61 0.72 0.58 | 2 0 0 1 1 1 |
| After learning | 1.20 1.17 1.18 1.20 1.20 1.16 | 3 1 2 1 1 0 |

Learned policy:

- swing speed = 1.03 − 0.37 × (arrival speed − 3.64) m/s
- tilt = 0.43 × offset + 0.09 × sideways speed (rad)

# Two paddles, one ball

[`two_paddles.py`](two_paddles.py) has two paddles, both resting at 0.5 m and 0.6 m apart, rally one ball. Every return should peak at 1.2 m, halfway between them. [`two_paddles.mp4`](two_paddles.mp4) shows the rally before, during and after learning.

```
python two_paddles.py                                    # laptop, about 90 s
python3 runIt.py --cpus 4 ../ping-pong/two_paddles.py     # cluster, from MuJoCo/mujoco-hpc/
```

Both paddles share one policy, mirrored left and right, with five learned numbers. Swing speed works as above (`u0`, `k`). The tilt now also has to send the ball across: `lean = t0 − (c_pos × offset + c_vel × sideways speed)`, toward the other paddle. `t0` is the base tilt that sends the ball over, and `c_vel` cancels the speed the ball came in with.

The reward is the same as for one paddle: apex error plus distance from the middle. A miss, or the same paddle hitting twice in a row, ends the rally.

A rally is touchy, because a slightly wrong tilt ends it. So a single learner sometimes settles on a so-so policy. The script runs 4 learners with different random seeds side by side, one per CPU core, and keeps the best. In the run below, 3 of the 4 found a good policy.

| Stage | Apex heights (m) | Off-center (cm) |
|---|---|---|
| Before learning | 0.74, then the rally ends | 30 |
| Learning (iteration 4) | 0.99 1.12 1.06 1.07 1.07 1.08 1.10 1.05 | 18 6 2 2 3 3 6 11 |
| After learning | 1.20 1.17 1.24 1.21 1.20 1.22 1.19 1.19 | 3 3 1 1 1 0 1 1 |

Learned policy:

- swing speed = 1.11 − 0.48 × (arrival speed − 3.64) m/s
- lean = 0.12 − (0.22 × offset + 0.13 × sideways speed) rad
