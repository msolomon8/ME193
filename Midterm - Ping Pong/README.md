# Midterm: Wii-Style Ping Pong

**Mia Solomon**, ME193

A virtual table tennis game you play against the computer. You hold a LEGO Double Motor as your paddle and swing it at the screen. The camera tracks where your arm is, the motor's gyroscope senses your swing, and the motor buzzes in your hand when you hit the ball. You pick your level by pointing an AprilTag at the screen, like a Wii remote. First to 5 points wins, and your record number of continuous hits is posted live over MQTT.

## How to play

1. Turn on the Double Motor (red 1142 card). From the ME193 folder run:
   `le-venv/bin/python "Midterm - Ping Pong/ping_pong.py"`
2. **First time only:** the game asks you to hold the paddle still, then swing 3 times, to calibrate your swing.
3. **Start page:** hold the AprilTag (on the paddle) over **Easy**, **Medium** or **Hard** until the ring fills.
4. **Play:**
   - Move your right arm left and right to line your paddle up with the ball.
   - Swing as the ball reaches you.
   - Where the ball hits your paddle aims your return, and a harder swing sends it back faster.
5. **Win:** first to 5 wins. YOU WIN! or YOU LOSE! shows, then the game goes back to the start page.

## What each requirement uses

| Requirement | How the game does it | From class |
|---|---|---|
| **Pose**: is the paddle in the right place? | MediaPipe Pose finds your right shoulder, elbow and wrist. Your arm's position, measured from your shoulder, moves the paddle. | Project 1 (Pose Race) |
| **IMU**: are you swinging? | The Double Motor's gyroscope is read 50 times a second. You only hit the ball if your rotation speed passes your calibrated swing threshold while the paddle is under the ball. | Q-table Training (IMU) |
| **AprilTag**: player level and game start | One tag (id 0) is a pointer. Holding it on a level box picks the ball speed and starts the game. | Project 2 (AprilTag Parking) |
| **Score over MQTT** | Your record number of continuous hits is posted as a float (e.g. `7.0`) to `ME193/Rogers/Mia` on every hit. | MQTT, Project 4 |
| **Policy / AI** | The computer learns where to aim using Q-learning. | Q-table Training |
| **Something new** | Haptic feedback: the motor in your hand buzzes on a hit and on a miss. Also a Wii-style AprilTag pointer, and swing calibration. | New |

Levels:

| Level | Ball travel time | How often the computer aims randomly (ε) |
|---|---|---|
| Easy | 2.1 s | 50% |
| Medium | 1.6 s | 20% |
| Hard | 1.2 s | 5% |

## Describe the policy: how does it make decisions?

The computer decides **where to aim each shot** using a Q-table: a table of how good each aim has been in each situation.

- **State:** where your paddle is when the computer hits the ball, in 5 zones: far left, left, center, right, far right.
- **Action:** which of 5 lanes across the table to aim at.
- **Reward:**
  - **+1** if you miss that shot (point for the computer).
  - **−0.3** if you return it.
- **Update:** `Q(s,a) = (1 − α)·Q(s,a) + α·(r + γ·max Q(s'))`, with α = 0.3 and γ = 0.5. Here s′ is where you are at the computer's next shot; if the point ended, there is no s′.
- **Choosing (ε-greedy):**
  - With probability ε the computer aims randomly, which is exploring.
  - Otherwise it picks the aim with the highest Q value for where you're standing.
  - ε comes from the level: Easy explores a lot, so it's more random and less ruthless. Hard almost always uses what it has learned.
- **Memory:** the Q-table is saved to `q_table.json` after every game, so it keeps learning your weak spots from game to game. In a simulation where the player was slow to reach the far left, the table learned that "far left" was the best aim when the player stood on the right.

The computer's paddle movement isn't learned. It moves toward where it predicts your shot will land, with a reaction delay, a top speed and some error. That's why wide, fast shots can beat it.

Your returns follow simple physics-style rules rather than a policy:
- **Where** the ball meets your paddle sets the direction (the paddle's edge sends it wide).
- **How hard** you swing (the gyro peak) sets the speed.

## What are the potential limitations of the game?

- **Pose tracking is 2-D.** The camera only knows left and right, not how far your arm reaches forward, so you can't play near or far shots. MediaPipe can also lose your wrist when it moves fast, the lighting is poor, or your arm crosses your body.
- **Swing detection is a single threshold.** Any fast rotation counts as a swing, including a shake, and the direction of the swing doesn't matter. You may need to recalibrate if someone else plays or you change your grip.
- **Bluetooth lag.** Gyro readings arrive over Bluetooth Low Energy, about every 20 ms with some delay, so the hit window has to be generous.
- **The AprilTag must be visible and fairly large.** Glare, a small print or standing far away can stop it being detected.
- **The Q-table is small and starts empty.** With only 5 states and 5 aims, it can't learn things like timing or spin. It needs a number of games before its aims are better than random, and it learns one player's habits, so it isn't always right for a new player.
- **Simplified ping pong.** There's no spin, no ball going out or into the net, and no win-by-2 rule.
- **Hardware setup.** You need a webcam, a laptop that can see the motor over Bluetooth, and a public MQTT broker (`test.mosquitto.org`). If the broker is down, the game still runs but doesn't post.

## What AI/ML algorithms did I use, and how do they work?

- **MediaPipe Pose (deep learning):** a neural network trained on many images of people. It predicts 33 body landmarks, such as shoulders, elbows and wrists, in each camera frame.
- **Q-learning (reinforcement learning):** the computer tries aims, gets a reward when you miss, and updates a table of how good each aim is from each position. It mostly picks the best-scoring aim but sometimes explores (ε-greedy).
- **AprilTag detection (computer vision, not ML):** OpenCV finds square black-and-white markers in the image and decodes their bit pattern into an ID. Its corners give the tag's position in the image.
- **Swing calibration (learning a threshold from data):** the game records your resting gyro noise and 3 practice swings, then sets the swing threshold at 45% of your typical swing. That separates real swings from small wrist movements for you specifically.

## Template questions

*Notes for Mia: these are your reflections, so answer them in your own words. The bullets are things that happened while building the game, to jog your memory.*

**What did you make?**
- A Wii-style ping-pong game played with a LEGO Double Motor as the paddle, against a Q-learning opponent.

**What are you proud of?**
- *(your answer)*
- Possible points:
  - The Wii look.
  - The haptic paddle.
  - The AprilTag pointer.
  - That it works as a real rally game.

**What was hardest / what went wrong?**
- *(your answer)*
- Things that came up:
  - The webcam mirroring made MediaPipe track the wrong wrist.
  - The first swing calibration counted tiny wiggles as swings.
  - Reaching the left side of the table was hard until the table was measured from your shoulder and locked during a game.
  - The paddle twitched until it was smoothed.
  - The Double Motor lost its connection card serial after a firmware check.

**What did you learn?**
- *(your answer)*

**What would you do next?**
- *(your answer)*
- Ideas:
  - Show the game on the UNO Q LED matrix.
  - Use the swing direction for topspin or backspin.
  - Pre-train the Q-table in simulation.
  - Add two-player mode with two motors.

## Files

| File | What it is |
|---|---|
| `ping_pong.py` | The game |
| `apriltag-36h11-id0.png` | The AprilTag to print (about 8–10 cm across) |
| `swing_calibration.json` | Your saved swing calibration (`--calibrate` redoes it) |
| `q_table.json` | The computer's learned Q-table |
| `record.json` | Your record streak |
