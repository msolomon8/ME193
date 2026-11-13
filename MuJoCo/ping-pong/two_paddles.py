"""Two ping-pong paddles rally one red ball. Each paddle rests 0.5 m off the ground, 0.6 m apart;
every return should peak at 1.2 m, halfway between them. They learn how to play from a reward.

    python two_paddles.py             # on a laptop (pip install mujoco imageio imageio-ffmpeg)
    python3 runIt.py --cpus 4 ../ping-pong/two_paddles.py   # or on the cluster, from MuJoCo/mujoco-hpc/

The video shows the rally before, during, and after learning (the reward is printed on screen).

How it works
  * The rally starts with the ball dropped near the left paddle, a little off its center,
    drifting sideways, from a random height.
  * Both paddles use the same policy (5 learned numbers), mirrored left/right. When the ball is
    coming down on its side, a paddle predicts where/when/how fast it will arrive at 0.5 m, then:
        - slides under that spot                                     (moves with the ball)
        - tilts its face toward the other paddle by
              lean = t0 - (c_pos * offset + c_vel * sideways speed)  (t0, c_pos, c_vel learned)
          t0 sends the ball across; the rest corrects for where and how the ball came in
        - pulls back and swings up through 0.5 m at
              u = u0 + k * (arrival speed - V_REF)                   (u0, k learned)
          V_REF is how fast a ball that peaked at exactly 1.2 m arrives.
  * Reward per return: -(apex - 1.2 m)^2 - 0.5 * (apex distance from the middle)^2.
    A miss, or the same paddle hitting twice in a row, ends the rally; the returns that never
    happened score as an apex of 0 m. Higher is better; 0 is perfect.
  * Learning: cross-entropy method: try a batch of policies, keep the best few, refit, repeat.
    A rally is touchy (a slightly wrong tilt ends it), so a single learner sometimes gets stuck
    on a so-so policy. We run 4 learners with different random seeds side by side and keep
    the best. No GPU, no neural network; about a minute on a laptop with 4+ cores.
"""
import os
import sys

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")  # offscreen rendering on the cluster; set before importing mujoco

import imageio
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

TARGET_APEX = 1.2      # m, where the ball should peak (halfway between the paddles)
PADDLE_Z = 0.5         # m, paddle face height at rest
GAP = 0.3              # m, each paddle's home is this far left/right of the middle
HITS = 6               # returns per training rally
OUT = "two_paddles.mp4"

BALL_R = 0.025         # ball radius (m); a bit bigger than a real 20 mm ping-pong ball so it shows on video
PADDLE_HALF_T = 0.006  # paddle half-thickness
SIDES = {"L": -1, "R": +1}   # left paddle sends the ball toward +x, right toward -x


def paddle_xml(name, sign, color):
    return f"""
    <body name="{name}" pos="{sign * GAP} 0 {PADDLE_Z - PADDLE_HALF_T}" gravcomp="1">
      <joint name="{name}_px" type="slide" axis="1 0 0"/>
      <joint name="{name}_py" type="slide" axis="0 1 0"/>
      <joint name="{name}_pz" type="slide" axis="0 0 1"/>
      <joint name="{name}_ty" type="hinge" axis="0 1 0"/>
      <joint name="{name}_tx" type="hinge" axis="1 0 0"/>
      <geom type="cylinder" size=".075 {PADDLE_HALF_T}" mass="2" rgba="{color}"/>
      <geom type="box" pos="{sign * .115} 0 0" size=".045 .014 .01" mass=".1" rgba=".75 .55 .3 1"
            contype="0" conaffinity="0"/>
    </body>"""


XML = f"""
<mujoco>
  <option timestep="0.0005"/>
  <visual><global offwidth="960" offheight="720"/></visual>
  <default>
    <!-- solref = "-stiffness -damping" of every contact. Less damping = bouncier.
         Stiff enough that the ball cannot sink through a paddle, soft enough that each hit
         spans many time steps (consistent bounces); solimp keeps the stiffness constant.
         The damping makes the ball lose about half its speed on a still paddle (restitution
         ~0.5, like soft rubber), so the paddles have to swing to get it up to the line.
         condim=1: frictionless, so only a paddle's tilt changes the ball's sideways motion.
         (Set for all geoms: MuJoCo mixes the two touching geoms' values.) -->
    <geom solref="-100000 -136" solimp=".95 .95 .001" condim="1"/>
  </default>
  <worldbody>
    <light pos="0 -1 4"/>
    <light pos="1 1 3" diffuse=".4 .4 .4"/>

    <geom name="floor" type="plane" size="5 5 .1" rgba=".3 .4 .5 1"/>

    <!-- faint green ring where every apex should be (visual only, no collisions) -->
    <geom type="cylinder" pos="0 0 {TARGET_APEX}" size=".12 .001" rgba=".2 .9 .3 .35"
          contype="0" conaffinity="0"/>

    <body name="ball" pos="0 0 1.5">
      <freejoint/>
      <geom name="ball_geom" type="sphere" size="{BALL_R}" mass=".0027" rgba="1 .05 .05 1"/>
    </body>
    {paddle_xml("L", -1, ".12 .12 .12 1")}
    {paddle_xml("R", +1, ".1 .15 .35 1")}
  </worldbody>
</mujoco>
"""

model = mujoco.MjModel.from_xml_string(XML)
BALL = model.body("ball").id
G = 9.81
V_REF = np.sqrt(2 * G * (TARGET_APEX - PADDLE_Z - BALL_R))  # m/s, arrival speed of an ideal ball
T_SWING = 0.12         # s, the paddle swings up at constant speed for this long before contact
T_WINDUP = 0.20        # s, time spent pulling the paddle back before the swing starts
STROKE = 0.25          # m, max follow-through before the paddle heads back to rest
MAX_TILT = 0.45        # rad (~25 degrees)
MAX_SLIDE = 2.5        # m/s, how fast a paddle can move sideways
SUBSTEPS = 4           # physics steps per control update (control runs at 500 Hz)


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
    """wait -> wind up (pull back, slide under, tilt) -> swing up through 0.5 m -> return.

    Works in its own mirrored frame: +x' always points at the other paddle.
    """

    def __init__(self, name, params):
        self.s = SIDES[name]                   # -1 left, +1 right
        self.home = np.array([self.s * GAP, 0.0])
        self.mirror = np.array([-self.s, 1.0])  # world (x, y) -> mirrored (x', y)
        self.u0, self.k, self.t0, self.c_pos, self.c_vel = params
        self.j = {n: model.joint(f"{name}_{n}").dofadr[0] for n in ("px", "py", "pz", "ty", "tx")}
        self.qa = {n: model.joint(f"{name}_{n}").qposadr[0] for n in self.j}
        self.mode, self.u, self.t_hit = "wait", 0.0, 0.0
        self.spot, self.lean = self.home.copy(), np.zeros(2)

    def mine(self, spot):
        return np.sign(spot[0]) == self.s      # the ball is coming down on my side

    def __call__(self, data):
        q = data.qpos[self.qa["pz"]]
        ball_vz = data.qvel[2]
        p = predict(data)
        if self.mode in ("wait", "return"):    # chase the ball if it's mine, otherwise go home
            self.spot = p[0] if (p is not None and self.mine(p[0])) else self.home

        if self.mode == "wait":
            vz = -10 * q                       # hold at rest height
            if (p is not None and ball_vz < 0 and self.mine(p[0])
                    and p[1] < T_WINDUP + T_SWING):
                spot, t_arrive, v_in, v_side = p
                self.spot = spot
                off = self.mirror * (spot - self.home)       # where it lands, mirrored
                vel = self.mirror * v_side                   # sideways speed, mirrored
                self.u = self.u0 + self.k * (v_in - V_REF)                          # learned
                lean = np.array([self.t0, 0.0]) - (self.c_pos * off + self.c_vel * vel)  # learned
                self.lean = np.clip(lean, -MAX_TILT, MAX_TILT)
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
                self.lean = np.zeros(2)
        if self.mode == "return":
            vz = float(np.clip(-6 * q, -1.5, 1.5))
            if abs(q) < 2e-3:
                self.mode = "wait"

        # Sideways: chase the spot. Tilt: lean[0] > 0 leans the face toward the other paddle
        # (mirrored +x'); lean[1] > 0 leans it toward +y. Rotating about +y by a leans the normal
        # toward +x; rotating about +x by b leans it toward -y.
        pos = self.home + np.array([data.qpos[self.qa["px"]], data.qpos[self.qa["py"]]])
        vxy = np.clip(15 * (self.spot - pos), -MAX_SLIDE, MAX_SLIDE)
        ang = np.array([data.qpos[self.qa["ty"]], data.qpos[self.qa["tx"]]])
        want = np.array([-self.s * self.lean[0], -self.lean[1]])
        w = np.clip(25 * (want - ang), -8, 8)
        data.qvel[self.j["px"]], data.qvel[self.j["py"]], data.qvel[self.j["pz"]] = vxy[0], vxy[1], vz
        data.qvel[self.j["ty"]], data.qvel[self.j["tx"]] = w[0], w[1]


def start_states(n, seed):
    """Random serves: the ball drops near the left paddle, off-center and drifting."""
    rng = np.random.default_rng(seed)
    return [dict(h=rng.uniform(0.9, 1.5), xy=np.array([-GAP, 0]) + rng.uniform(-0.06, 0.06, 2),
                 vxy=rng.uniform(-0.15, 0.15, 2)) for _ in range(n)]


def rollout(params, start, hits=HITS, renderer=None, cam=None, fps=30, label=None):
    """One rally. Returns (list of (apex height, distance from middle), video frames)."""
    data = mujoco.MjData(model)
    data.qpos[0:3] = [*start["xy"], start["h"]]
    data.qvel[0:2] = start["vxy"]
    mujoco.mj_forward(model, data)
    paddles = [Paddle("L", params), Paddle("R", params)]
    apexes, frames, rising, last_hitter = [], [], False, None
    while len(apexes) < hits and data.time < 15:
        for p in paddles:
            p(data)
        mujoco.mj_step(model, data, nstep=SUBSTEPS)
        vz, (x, y, z) = data.qvel[2], data.xpos[BALL]
        if vz > 0.05 and not rising:           # just got hit
            rising = True
            hitter = "L" if x < 0 else "R"
            if hitter == last_hitter:          # same paddle twice: not a rally
                break
            last_hitter = hitter
        elif rising and vz <= 0:               # top of a return
            apexes.append((z, np.hypot(x, y)))
            rising = False
        if z < PADDLE_Z - 0.2:                 # missed: ball fell past the paddles
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
        z, d = apexes[i] if i < len(apexes) else (0.0, 0.3)   # returns that never happened
        r -= (z - TARGET_APEX) ** 2 + 0.5 * d ** 2
    return r / hits


TRAIN_STARTS = start_states(4, seed=1)
NAMES = ("u0", "k", "t0", "c_pos", "c_vel")
STD_FLOOR = np.array([0.03, 0.03, 0.01, 0.03, 0.02])


def reward(params):
    return np.mean([score(rollout(params, s)[0]) for s in TRAIN_STARTS])


RESTARTS = 4           # independent learners; the best one wins (some get stuck, see README)


def cpus():
    try:
        return len(os.sched_getaffinity(0))    # Linux/Slurm: only the cores this job was given
    except AttributeError:
        return os.cpu_count() or 1


def train(seed, iters=12, pop=32, elite=6):
    """Cross-entropy method over (u0, k, t0, c_pos, c_vel). Returns the policy after each iteration."""
    rng = np.random.default_rng(seed)
    mean, std = np.zeros(5), np.array([1.0, 1.0, 0.3, 1.0, 0.5])
    history = [(mean.copy(), reward(mean))]
    for it in range(iters):
        cands = mean + std * rng.standard_normal((pop, 5))
        scores = np.array([reward(c) for c in cands])
        best = cands[np.argsort(scores)[-elite:]]
        mean, std = best.mean(0), np.maximum(best.std(0), STD_FLOOR)   # keep exploring a little
        r = reward(mean)
        history.append((mean.copy(), r))
        print(f"learner {seed}  iter {it:2d}  reward {r:9.5f}  "
              + " ".join(f"{n}={v:.2f}" for n, v in zip(NAMES, mean)), flush=True)
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
        d.text((16, 46), f"returns {len(apexes)}   last apex {z:.2f} m (target 1.20)   "
               f"off-center {100 * dist:.0f} cm", fill=(120, 230, 140), font=SMALL)
    return np.asarray(im)


if __name__ == "__main__":
    import multiprocessing
    with multiprocessing.Pool(min(RESTARTS, cpus())) as pool:   # learners run side by side
        histories = pool.map(train, range(RESTARTS))
    for i, h in enumerate(histories):
        print(f"learner {i}: final reward {h[-1][1]:.4f}")
    best = max(range(RESTARTS), key=lambda i: histories[i][-1][1])
    history = histories[best]
    print(f"using learner {best}")
    params, r = history[-1]
    u0, k, t0, c_pos, c_vel = params
    print(f"learned policy:\n  swing speed = {u0:.2f} + {k:.2f} * (arrival speed - {V_REF:.2f}) m/s"
          f"\n  lean        = {t0:.2f} - ({c_pos:.2f} * offset + {c_vel:.2f} * sideways speed) rad")

    renderer = mujoco.Renderer(model, 720, 960)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.85]
    cam.distance, cam.azimuth, cam.elevation = 2.4, 90, -12
    show = start_states(1, seed=7)[0]          # the same serve for every stage
    stages = [("before learning", 0), ("learning...", len(history) // 3),
              ("after learning", len(history) - 1)]
    frames = []
    for name, i in stages:
        p, r = history[i]
        apexes, f = rollout(p, show, hits=8, renderer=renderer, cam=cam,
                            label=f"{name}  (iteration {i}, reward {r:.3f})")
        frames += f
        print(f"{name:16s} apexes: " + " ".join(f"{z:.2f}" for z, _ in apexes)
              + "   off-center (cm): " + " ".join(f"{100 * d:.0f}" for _, d in apexes))
    renderer.close()
    imageio.mimwrite(OUT, frames, fps=30)
    print(f"wrote {OUT}: {len(frames)} frames")
