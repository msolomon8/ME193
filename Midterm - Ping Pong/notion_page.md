[COVER IMAGE: screenshot of the game, or a photo of you mid-swing]

GitHub - msolomon8/ME193 at main, Midterm - Ping Pong folder: https://github.com/msolomon8/ME193/tree/main/Midterm%20-%20Ping%20Pong

**Mia Solomon**, ME193 Midterm

## Table of Contents

### This is what I did

[VIDEO: a full game, from picking a level with the AprilTag to YOU WIN! / YOU LOSE!]

I built a **Wii Sports-style ping-pong game** 🏓 that you play with a **LEGO Double Motor as your paddle**. You stand in front of the laptop camera holding the motor and swing it at the screen.

Four things decide whether you hit the ball:

- **MediaPipe Pose** watches your right arm to figure out *where* your paddle is on the table.
- **The Double Motor's gyroscope (IMU)** decides *whether you actually swung*: no swing, no hit, even if you're in the right spot.
- **An AprilTag** on the paddle works like a Wii remote pointer on the start page. You point it at Easy, Medium or Hard and hold it there to pick the level (ball speed) and start the game.
- **A Q-learning opponent** learns where to aim its shots to make *you* miss.

The ball rallies back and forth until someone misses, just like real ping pong, and first to **5** wins. Every time you hit the ball, your **record number of continuous hits** is posted live to `ME193/Rogers/Mia` over MQTT as a float (like `7.0`).

The new thing we haven't done in class: **haptic feedback**! The motor in your hand buzzes when you hit the ball (a short tap) and when you lose a point (a long buzz), so you can *feel* the shot.

[SCREENSHOT: start page with the Easy / Medium / Hard boxes and the AprilTag pointer]

[SCREENSHOT: mid-rally, with the scoreboard, swing meter and camera view]

| What you do | What the game uses | What happens |
| --- | --- | --- |
| Hold the AprilTag on a level box for 2 s | AprilTag (OpenCV, 36h11 family, id 0) | Picks Easy / Medium / Hard and starts the game |
| Move your right arm left/right | MediaPipe Pose (shoulder, elbow, wrist) | Moves your paddle across the table |
| Swing the paddle as the ball arrives | Double Motor gyroscope | Hit! The paddle buzzes and the ball goes back |
| Hit the ball off the edge of your paddle | Where the ball meets the paddle | Aims your return wide |
| Swing harder | Peak gyro reading | Your shot goes back faster |
| Miss | — | Long buzz, point to the computer |
| Every hit | MQTT `ME193/Rogers/Mia` | Posts your record streak, e.g. `7.0` |

| Level | Ball travel time | Computer aims randomly (ε) |
| --- | --- | --- |
| Easy | 2.1 s | 50% |
| Medium | 1.6 s | 20% |
| Hard | 1.2 s | 5% |

**The camera view** (bottom-right corner) shows:

- your pose as white lines,
- yellow lines where the table's left and right edges are,
- a dot on your paddle point that turns **green while the motor senses a swing**,
- a blue box around the AprilTag on the start page.

There's also a swing meter that shows how hard you're swinging compared with the threshold.

**First-time calibration:** everyone swings differently, so the first time you play, the game asks you to hold the paddle still and then swing 3 times. It sets the swing threshold to 45% of your typical swing.

[VIDEO: calibration, i.e. hold still, then 3 swings]

[SCREENSHOT: the MQTT messages arriving on ME193/Rogers/Mia]

### My Paddle

[PHOTO: the Double Motor paddle with the AprilTag extension]

**Code on GitHub:** Midterm - Ping Pong folder

- `ping_pong.py`: the whole game (pose, IMU, AprilTag, Q-learning, MQTT, haptics, Wii-style screen)
- `apriltag-36h11-id0.png`: the AprilTag to print
- `swing_calibration.json`: my saved swing calibration
- `q_table.json`: the computer's learned Q-table (saved after every game)
- `record.json`: my record streak
- **The policy** (`QPolicy` + `Game.cpu_hits`): where the computer aims

    ```python
    STATE_EDGES = [0.2, 0.4, 0.6, 0.8]                 # your paddle lane -> 5 states (far left .. far right)
    AIMS = [0.15, 0.32, 0.5, 0.68, 0.85]                # 5 actions: aim lanes across the table
    ALPHA = 0.3                  # learning rate
    GAMMA = 0.5                  # how much the next shot's value counts
    REWARD_YOU_MISS = 1.0        # the computer's shot made you miss: point for the computer
    REWARD_YOU_RETURN = -0.3     # you got it back

    def choose(self, s, epsilon):
        if random.random() < epsilon:
            return random.randrange(len(AIMS))                     # explore
        best = np.flatnonzero(self.Q[s] == self.Q[s].max())
        return int(random.choice(best))                            # exploit (random among ties)

    def update(self, s, a, r, s_next=None):
        future = 0.0 if s_next is None else GAMMA * self.Q[s_next].max()
        self.Q[s, a] = (1 - ALPHA) * self.Q[s, a] + ALPHA * (r + future)

    def cpu_hits(self, from_lane):
        """The computer hits: the Q-learning policy picks where to aim from where you are."""
        s = self.policy.state_of(self.paddle_lane)
        if self.pending is not None:                  # you returned its last shot
            self.policy.update(*self.pending, REWARD_YOU_RETURN, s)
        a = self.policy.choose(s, LEVELS[self.level][3])
        self.pending = (s, a)
        aim = min(0.92, max(0.08, AIMS[a] + random.gauss(0, AIM_NOISE)))
        self.shot = Shot(True, from_lane, aim, self.travel_time)
    ```

- **Swing detection** (`MotorPaddle._read_imu`): a background thread reads the gyroscope 50× per second

    ```python
    g = math.sqrt(d.gyroscopeX ** 2 + d.gyroscopeY ** 2 + d.gyroscopeZ ** 2)
    self.gyro = g
    now = time.time()
    if self.threshold is None or now < self.quiet_until:
        continue                         # ignore the motor's own buzzing
    if g > self.threshold:
        self.swing_time = now            # last moment you were swinging
    if not self._in_swing and g > self.threshold:
        self._in_swing = True
        self._swing_start, self.swing_peak = now, g
    elif self._in_swing:
        self.swing_peak = max(self.swing_peak, g)       # how hard -> how fast your shot goes
        if g < self.threshold * SWING_END or now - self._swing_start > MAX_SWING_TIME:
            self._in_swing = False
    ```

- **Hit or miss?** (pose + IMU together): checked every frame as the ball reaches you

    ```python
    if f >= HIT_WINDOW_START:
        if self.hit_window_opened is None:
            self.hit_window_opened = now
        lined_up = abs(self.shot.lane() - paddle_lane) <= PADDLE_WIDTH / 2      # POSE
        swung = self.swung_since(self.hit_window_opened - SWING_EARLY) and \
            self.swung_since(self.shot.t0)                                      # IMU
        if lined_up and swung:
            self.streak += 1
            self.poster.new_streak(self.streak)                                 # MQTT
            self.player_hits(paddle_lane, self.shot.lane())                     # + haptic buzz
        elif f > HIT_WINDOW_END:
            self.point(player_won=False)
    ```

- **Your return shot**: where it hits your paddle aims it, and swing power sets the speed

    ```python
    offset = max(-1.0, min(1.0, (from_lane - paddle_lane) / (PADDLE_WIDTH / 2)))
    aim = min(0.9, max(0.1, 0.5 + offset * AIM_SPREAD + random.gauss(0, 0.05)))
    power = self.paddle.swing_peak / self.paddle.threshold if self.paddle.threshold else 1.0
    speedup = min(RETURN_SPEEDUP_MAX, max(1.0, 0.8 + 0.2 * power))
    self.shot = Shot(False, from_lane, aim, self.travel_time / speedup)
    self.paddle.buzz(HIT_BUZZ_MS)
    ```

- **MQTT score**: posted on every hit

    ```python
    def new_streak(self, streak):
        if streak > self.record:
            self.record = streak
            RECORD_FILE.write_text(json.dumps({"record": self.record}))
        self.client.publish("ME193/Rogers/Mia", f"{float(self.record):.1f}")
    ```

[VIDEO: Easy level]

[VIDEO: Hard level]

[VIDEO: winning a game]

[VIDEO: losing a game]

### Questions

**Describe the policy: how does it make decisions?**

The computer's decision is *where to aim each shot*, and it makes it with **Q-learning**:

- **State:** where my paddle is when the computer hits the ball, split into 5 zones (far left, left, center, right, far right).
- **Action:** which of 5 lanes across the table to aim at.
- **Reward:** **+1** if I miss that shot (a point for the computer), **−0.3** if I return it.
- **Update:** after each shot it updates its Q-table with `Q(s,a) = (1 − α)·Q(s,a) + α·(r + γ·max Q(s'))`, where α = 0.3, γ = 0.5, and s′ is where I am at the computer's next shot. If the point ended, there's no s′.

It chooses with **ε-greedy**:

- With probability ε it aims randomly (exploring).
- Otherwise it picks the aim with the highest Q value for where I'm standing.
- The level sets ε: Easy 50% random, Medium 20%, Hard only 5%, so on Hard it almost always uses what it's learned.

The Q-table is saved after every game, so the computer keeps learning my weak spots from game to game. I tested it with a simulated player who drifts to the right and is slow to reach the far left. The computer learned that when the player is on the right, **"far left" is the best aim** (Q = 0.90, compared with 0.08–0.19 for the other aims).

The computer's *paddle* isn't learned. It moves toward where it predicts my shot will land, with a 0.25 s reaction delay, a top speed, and some error, so wide or fast shots can still beat it.

On my side the decisions are rule-based:

- I hit the ball only if **both** my paddle is within half a paddle-width of the ball (pose) **and** I swung during the hit window (IMU).
- Where the ball meets my paddle aims the return.
- How hard I swing (the gyro peak) makes it faster.

**What are the potential limitations of your game?**

- **Pose is only 2-D.** The camera knows left and right but not how far forward my arm reaches. MediaPipe can also lose my wrist when it moves fast, in bad lighting, or when my arm crosses my body.
- **Swing detection is one threshold.** Any fast rotation counts as a swing (even shaking the motor), and the direction doesn't matter, so there's no topspin or backhand. A different player, or a different grip, may need to recalibrate.
- **Bluetooth lag.** Gyro readings come over Bluetooth Low Energy about every 20 ms with some delay, so the hit window has to be forgiving.
- **The AprilTag has to be big and visible.** Glare, a small print, or standing too far away stops the pointer from working (keys 1/2/3 are a backup).
- **The Q-table is tiny and starts empty.** With 5 states × 5 aims it can't learn timing or spin. It takes a few games to beat random aiming, and it learns *one* player's habits.
- **Simplified ping pong.** No spin, no ball going out or into the net, no win-by-2.
- **Setup.** It needs a webcam, the Double Motor in Bluetooth range, and the public MQTT broker. If the broker is down, the game still runs; it just doesn't post.

**What different AI/ML algorithms did you use, and in two sentences or less how do they work?**

- **MediaPipe Pose (deep learning):** a neural network trained on lots of images of people predicts 33 body landmarks (shoulders, elbows, wrists…) in every camera frame. I use the right shoulder, elbow and wrist to place the paddle.
- **Q-learning (reinforcement learning):** the computer tries aims, gets rewarded when I miss, and updates a table of how good each aim is from each position. It mostly picks the best-scoring aim but sometimes explores (ε-greedy).
- **AprilTag detection (computer vision):** OpenCV finds the square black-and-white marker and decodes its pattern into an ID. Its corners tell me where it is in the image, which becomes the pointer.
- **Swing calibration (learning a threshold from data):** the game measures my resting gyro noise and 3 practice swings, then sets the swing threshold to 45% of my typical swing. That separates real swings from small wrist movements.

### Reflection questions

- What I learned: [YOUR ANSWER]
- What I am most proud of: [YOUR ANSWER]
- What took a while (was tough):
    - Getting the paddle to feel right.
    - The webcam image is mirrored, so MediaPipe tracked my *left* wrist when I wanted my right.
    - The paddle jumped around, especially mid-swing, so I blended in the elbow, smoothed it, and ignored tiny wiggles.
    - I couldn't reach the left third of the table, so I measured the table from my right shoulder instead of the middle of the camera. Then reaching across turned my shoulders and the table slid away, so now the table lines lock in place once the game starts.
- Most spectacular failure: [YOUR ANSWER]. One candidate: the first swing calibration counted two tiny wiggles as "swings", so the threshold ended up so low that the game thought I was swinging nonstop and never registered a real one. I was swinging away and nothing was getting hit 😭

```python
Thank you!
```
