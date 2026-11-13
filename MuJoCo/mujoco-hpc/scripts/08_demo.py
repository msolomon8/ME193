"""Ping-pong returns: a paddle at 0.5 m keeps a red ball going so each apex is 1.2 m off the
ground, and the ball stays over the middle. The paddle learns how to play from a reward.

    python 08_demo.py

The video shows the paddle before, during, and after learning (the reward is printed on screen).

How it works
  * Each rally starts with the ball at a random spot, drifting sideways, from a random height.
  * Policy (4 learned numbers). Before every hit the paddle predicts where/when/how fast the
    ball will arrive at 0.5 m, then:
        - slides under that spot                                (moves with the ball)
        - tilts its face by   tilt = c_pos * offset + c_vel * sideways speed
          to steer the ball back toward the middle              (c_pos, c_vel learned)
        - pulls back and swings up through 0.5 m at
              u = u0 + k * (arrival speed - V_REF)              (u0, k learned)
          V_REF is how fast a ball that peaked at exactly 1.2 m arrives.
  * Reward per bounce: -(apex - 1.2 m)^2 - 0.5 * (distance from the middle)^2.
    A missed ball scores as an apex of 0 m. Higher is better; 0 is perfect.
  * Learning: cross-entropy method: try a batch of policies, keep the best few, refit, repeat.
    No GPU, no neural network; it takes about a minute on a laptop.
"""
import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering on the cluster; set before importing mujoco

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TARGET_APEX = 1.2      # m, where the ball should peak
PADDLE_Z = 0.5         # m, paddle face height at rest
HITS = 5               # bounces per training rally
OUT = "demo.mp4"       # the video file to write (08_demo.sbatch looks for this name)

BALL_R = 0.025         # ball radius (m); a bit bigger than a real 20 mm ping-pong ball so it shows on video
PADDLE_HALF_T = 0.006  # paddle half-thickness

XML = f"""
<mujoco>
  <option timestep="0.0005"/>
  <visual><global offwidth="960" offheight="720"/></visual>
  <default>
    <!-- solref = "-stiffness -damping" of every contact. Less damping = bouncier.
         Stiff enough that the ball cannot sink through the paddle, soft enough that each hit
         spans many time steps (consistent bounces); solimp keeps the stiffness constant.
         The damping makes the ball lose about half its speed on a still paddle (restitution
         ~0.5, like soft rubber), so the paddle has to swing to get it back up to the line.
         condim=1: frictionless, so only the paddle's tilt changes the ball's sideways motion.
         (Set for all geoms: MuJoCo mixes the two touching geoms' values.) -->
    <geom solref="-100000 -136" solimp=".95 .95 .001" condim="1"/>
  </default>
  <worldbody>
    <light pos="0 -1 4"/>
    <light pos="1 1 3" diffuse=".4 .4 .4"/>

    <geom name="floor" type="plane" size="5 5 .1" rgba=".3 .4 .5 1"/>

    <!-- faint green ring at the target apex height (visual only, no collisions) -->
    <geom type="cylinder" pos="0 0 {TARGET_APEX}" size=".25 .001" rgba=".2 .9 .3 .25"
          contype="0" conaffinity="0"/>

    <body name="ball" pos="0 0 1.5">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="{BALL_R}" mass=".0027" rgba="1 .05 .05 1"/>
    </body>

    <!-- the paddle can slide in x, y, z and tilt about x and y; we drive its speeds directly -->
    <body name="paddle" pos="0 0 {PADDLE_Z - PADDLE_HALF_T}">
      <joint name="px" type="slide" axis="1 0 0"/>
      <joint name="py" type="slide" axis="0 1 0"/>
      <joint name="pz" type="slide" axis="0 0 1"/>
      <joint name="tilt_y" type="hinge" axis="0 1 0"/>
      <joint name="tilt_x" type="hinge" axis="1 0 0"/>
      <geom name="face" type="cylinder" size=".075 {PADDLE_HALF_T}" mass="2" rgba=".15 .15 .15 1"/>
      <geom type="box" pos=".115 0 0" size=".045 .014 .01" mass=".1" rgba=".75 .55 .3 1"
            contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
BALL = model.body("ball").id
J = {n: model.joint(n).dofadr[0] for n in ("px", "py", "pz", "tilt_y", "tilt_x")}
Q = {n: model.joint(n).qposadr[0] for n in J}
G = 9.81
V_REF = np.sqrt(2 * G * (TARGET_APEX - PADDLE_Z - BALL_R))  # m/s, arrival speed of an ideal ball
T_SWING = 0.12         # s, the paddle swings up at constant speed for this long before contact
T_WINDUP = 0.20        # s, time spent pulling the paddle back before the swing starts
STROKE = 0.25          # m, max follow-through before the paddle heads back to rest
MAX_TILT = 0.35        # rad (~20 degrees)
MAX_SLIDE = 2.5        # m/s, how fast the paddle can move sideways


def predict(data):
    """Where (x, y), in how long, and how fast the ball will reach the hitting height."""
    x, y, z = data.xpos[BALL]
    vx, vy, vz = data.qvel[0:3]
    h = z - BALL_R - PADDLE_Z
    if h < 0:
        return None
    v_in = np.sqrt(vz ** 2 + 2 * G * h)      # downward speed on arrival
    t = (vz + v_in) / G                       # works whether the ball is rising or falling
    return np.array([x + vx * t, y + vy * t]), t, v_in, np.array([vx, vy])


class Paddle:
    """wait -> wind up (pull back, slide under, tilt) -> swing up through 0.5 m -> return."""

    def __init__(self, params):
        self.u0, self.k, self.c_pos, self.c_vel = params
        self.mode, self.u, self.t_hit = "wait", 0.0, 0.0
        self.spot, self.tilt = np.zeros(2), np.zeros(2)

    def __call__(self, data):
        q = data.qpos[Q["pz"]]
        ball_vz = data.qvel[2]
        p = predict(data)
        if p is not None and self.mode in ("wait", "return"):
            self.spot = p[0]                  # keep sliding toward where the ball will come down

        if self.mode == "wait":
            vz = -10 * q                       # hold at rest height
            if p is not None and ball_vz < 0 and p[1] < T_WINDUP + T_SWING:
                spot, t_arrive, v_in, v_side = p
                self.spot = spot
                self.u = self.u0 + self.k * (v_in - V_REF)                       # learned
                self.tilt = np.clip(self.c_pos * spot + self.c_vel * v_side,    # learned
                                    -MAX_TILT, MAX_TILT)
                self.t_hit = data.time + t_arrive
                self.mode = "windup"
        if self.mode == "windup":
            t_left = self.t_hit - data.time - T_SWING
            if t_left <= 0:
                self.mode = "swing"
            else:
                vz = (-self.u * T_SWING - q) / t_left    # glide to the start of the swing
        if self.mode == "swing":
            vz = self.u
            if data.time > self.t_hit + 0.04 or q > STROKE:
                self.mode = "return"
                self.tilt = np.zeros(2)
        if self.mode == "return":
            vz = float(np.clip(-6 * q, -1.5, 1.5))
            if abs(q) < 2e-3:
                self.mode = "wait"

        # Sideways: chase the predicted spot. Tilt: turn toward the target angle.
        # tilt[0] > 0 should lean the face normal toward -x (sending the ball toward -x);
        # rotating about +y by a positive angle leans it toward +x, so use -tilt[0].
        # Likewise rotating about +x by a positive angle leans it toward -y, so use +tilt[1].
        pos = np.array([data.qpos[Q["px"]], data.qpos[Q["py"]]])
        vxy = np.clip(15 * (self.spot - pos), -MAX_SLIDE, MAX_SLIDE)
        ang = np.array([data.qpos[Q["tilt_y"]], data.qpos[Q["tilt_x"]]])
        want = np.array([-self.tilt[0], self.tilt[1]])
        w = np.clip(25 * (want - ang), -8, 8)
        data.qvel[J["px"]], data.qvel[J["py"]], data.qvel[J["pz"]] = vxy[0], vxy[1], vz
        data.qvel[J["tilt_y"]], data.qvel[J["tilt_x"]] = w[0], w[1]


def start_states(n, seed):
    """Random rallies: start height, position and sideways drift of the ball."""
    rng = np.random.default_rng(seed)
    return [dict(h=rng.uniform(0.9, 1.5), xy=rng.uniform(-0.12, 0.12, 2),
                 vxy=rng.uniform(-0.3, 0.3, 2)) for _ in range(n)]


def rollout(params, start, hits=HITS, renderer=None, cam=None, fps=30, label=None):
    """One rally. Returns (list of (apex height, distance from middle), video frames)."""
    data = mujoco.MjData(model)
    data.qpos[0:3] = [*start["xy"], start["h"]]
    data.qvel[0:2] = start["vxy"]
    mujoco.mj_forward(model, data)
    paddle = Paddle(params)
    apexes, frames, rising = [], [], False
    while len(apexes) < hits and data.time < 15:
        paddle(data)
        mujoco.mj_step(model, data)
        vz, (x, y, z) = data.qvel[2], data.xpos[BALL]
        if vz > 0.05:
            rising = True
        elif rising and vz <= 0:               # top of a bounce
            apexes.append((z, np.hypot(x, y)))
            rising = False
        if z < PADDLE_Z - 0.2:                 # missed: ball fell past the paddle
            break
        if renderer is not None and data.time >= len(frames) / fps:
            renderer.update_scene(data, camera=cam)
            frames.append(overlay(renderer.render(), label, apexes))
    if renderer is not None:                   # hold the last frame for half a second
        frames += [frames[-1]] * (fps // 2)
    return apexes, frames


def score(apexes, hits=HITS):
    """The reward: how close each apex was to 1.2 m and to the middle."""
    r = 0.0
    for i in range(hits):
        z, d = apexes[i] if i < len(apexes) else (0.0, 0.3)   # missed bounces score badly
        r -= (z - TARGET_APEX) ** 2 + 0.5 * d ** 2
    return r / hits


TRAIN_STARTS = start_states(4, seed=1)


def reward(params):
    return np.mean([score(rollout(params, s)[0]) for s in TRAIN_STARTS])


def train(iters=9, pop=32, elite=6, seed=0):
    """Cross-entropy method over (u0, k, c_pos, c_vel). Returns the policy after each iteration."""
    rng = np.random.default_rng(seed)
    mean, std = np.zeros(4), np.array([1.0, 1.0, 2.0, 0.5])
    history = [(mean.copy(), reward(mean))]
    for it in range(iters):
        cands = mean + std * rng.standard_normal((pop, 4))
        scores = np.array([reward(c) for c in cands])
        best = cands[np.argsort(scores)[-elite:]]
        mean, std = best.mean(0), best.std(0) + 1e-3
        r = reward(mean)
        history.append((mean.copy(), r))
        print(f"iter {it:2d}  reward {r:9.5f}  u0={mean[0]:.2f} k={mean[1]:.2f} "
              f"c_pos={mean[2]:.2f} c_vel={mean[3]:.2f}")
    return history


def font(size):
    try:
        return ImageFont.load_default(size=size)
    except TypeError:                          # older Pillow
        return ImageFont.load_default()


FONT, SMALL = font(26), font(20)


def overlay(img, label, apexes):
    if label is None:
        return img
    im = Image.fromarray(img)
    d = ImageDraw.Draw(im)
    d.text((16, 12), label, fill=(255, 255, 255), font=FONT)
    if apexes:
        z, dist = apexes[-1]
        d.text((16, 46), f"last apex {z:.2f} m (target 1.20)   off-center {100 * dist:.0f} cm",
               fill=(120, 230, 140), font=SMALL)
    return np.asarray(im)


if __name__ == "__main__":
    history = train()
    params, r = history[-1]
    u0, k, c_pos, c_vel = params
    print(f"learned policy:\n  swing speed = {u0:.2f} + {k:.2f} * (arrival speed - {V_REF:.2f}) m/s"
          f"\n  tilt        = {c_pos:.2f} * offset + {c_vel:.2f} * sideways speed (rad)")

    renderer = mujoco.Renderer(model, 720, 960)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.85]
    cam.distance, cam.azimuth, cam.elevation = 2.3, 70, -12
    show = start_states(1, seed=7)[0]          # the same tricky rally for every stage
    stages = [("before learning", 0), ("learning...", 2), ("after learning", len(history) - 1)]
    frames = []
    for name, i in stages:
        p, r = history[i]
        apexes, f = rollout(p, show, hits=6, renderer=renderer, cam=cam,
                            label=f"{name}  (iteration {i}, reward {r:.3f})")
        frames += f
        print(f"{name:16s} apexes: " + " ".join(f"{z:.2f}" for z, _ in apexes)
              + "   off-center (cm): " + " ".join(f"{100 * d:.0f}" for _, d in apexes))
    renderer.close()
    imageio.mimwrite(OUT, frames, fps=30)
    print(f"wrote {OUT}: {len(frames)} frames")
