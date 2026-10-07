"""Q-table (reinforcement learning) to drive the LEGO Double Motor in a straight line.

Follows the "Q-tables" Notion page (Silly Walks):
  - State  = yaw error (IMU yaw since reset_heading(), target 0°) put into 9 bins
  - Action = speed difference between the motors (5 choices), average speed stays BASE_SPEED
  - Reward = +1 on target (|error| < 2°), 0 acceptable (< 10°), -1 off course (>= 10°)
  - Update = Q(s,a) = (1 - α)·Q(s,a) + α·(r + γ·max Q(s'))
  - ε-greedy: explore with probability ε, else pick the best action; ε *= 0.98 every step

The robot drives for STEPS_PER_EPISODE steps, stops, and waits for you to put it back at
the start (press Enter). The Q-table is saved to q_table.json after every episode and loaded
again next time, so training picks up where it left off (--sim uses q_table_sim.json).

Usage:
    python q_table_drive.py            # train on the robot (red card 1142)
    python q_table_drive.py --run      # no exploring, no learning: drive with the learned table
    python q_table_drive.py --sim      # train on a simulated robot (no LEGO needed)
    python q_table_drive.py --reset    # start over with a table of zeros

Ctrl+C stops the motors and saves the table.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np

# lelib.py is shared across projects and lives one folder up, in ME193/.
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

# ---- LEGO connection ----
CARD_COLOR = "red"          # Connection Card: red, serial 1142
CARD_SERIAL = "1142"

# ---- States: yaw error bins (Notion table) ----
#  0: < -20°   1: -20..-10   2: -10..-5   3: -5..-2   4: -2..+2 (goal)
#  5: +2..+5   6: +5..+10    7: +10..+20  8: > +20°
BIN_EDGES = [-20, -10, -5, -2, 2, 5, 10, 20]
N_STATES = len(BIN_EDGES) + 1

# ---- Actions: speed difference, right minus left (Notion table, halved to turn slower) ----
#  A0 -20 (right turn)  A1 -10 (gentle right)  A2 0 (straight)  A3 +10 (gentle left)  A4 +20 (left turn)
ACTIONS = [-20, -10, 0, 10, 20]
STRAIGHT = ACTIONS.index(0)  # every episode starts with this: both sides at the same speed
BASE_SPEED = 25             # average motor speed (%), stays the same for every action

# ---- Learning ----
ALPHA = 0.2                 # learning rate: how much a new result overrides what we knew
GAMMA = 0.9                 # discount: how much the next state's best value counts
EPSILON_START = 1.0         # start fully exploring
EPSILON_DECAY = 0.98        # multiply ε by this every step
EPSILON_MIN = 0.05          # always explore a little

# ---- Timing ----
STEP_TIME = 0.3             # seconds each action runs before reading the new yaw
STEPS_PER_EPISODE = 30      # ~9 s of driving, then stop and reset the robot

TABLE_FILE = HERE / "q_table.json"          # real robot
SIM_TABLE_FILE = HERE / "q_table_sim.json"  # --sim keeps its own table so it never mixes with the real one


def state_of(yaw_error):
    """Yaw error (degrees) -> state bin 0..8."""
    return int(np.digitize(yaw_error, BIN_EDGES))


def reward_of(yaw_error):
    """Reward for where the action left us (Notion rewards table, checked in order)."""
    e = abs(yaw_error)
    if e < 2:
        return 1.0      # on target
    if e < 10:
        return 0.0      # acceptable
    return -1.0         # off course


def wrap(angle):
    """Keep an angle in -180..180 so 350° reads as -10°."""
    return (angle + 180) % 360 - 180


def choose_action(Q, s, epsilon):
    if random.random() < epsilon:
        return random.randrange(len(ACTIONS))                         # explore
    best = np.flatnonzero(Q[s] == Q[s].max())
    return int(random.choice(best))                                   # exploit (random among ties)


def update(Q, s, a, r, s_next):
    Q[s, a] = (1 - ALPHA) * Q[s, a] + ALPHA * (r + GAMMA * Q[s_next].max())


def speeds_for(a):
    diff = ACTIONS[a]
    left = BASE_SPEED - diff // 2
    right = BASE_SPEED + diff // 2
    return left, right


def print_table(Q):
    names = ["<-20", "-20..-10", "-10..-5", "-5..-2", "-2..+2", "+2..+5", "+5..+10", "+10..+20", ">+20"]
    print("\nState      " + "".join(f"{a:>+8}" for a in ACTIONS) + "   best")
    for s in range(N_STATES):
        row = "".join(f"{q:8.2f}" for q in Q[s])
        best = ACTIONS[int(Q[s].argmax())] if Q[s].any() else "-"
        print(f"{s} {names[s]:>9}{row}   {best}")
    print()


def load_table(path, reset):
    if path.exists() and not reset:
        data = json.loads(path.read_text())
        if data.get("actions") == ACTIONS and data.get("base_speed") == BASE_SPEED:
            print(f"Loaded Q-table from {path.name} ({data['steps']} steps trained)")
            return np.array(data["Q"]), data["epsilon"], data["steps"]
        # Scores learned with other speeds don't apply to these actions
        print(f"{path.name} was trained with different ACTIONS/BASE_SPEED -- starting a new table")
    return np.zeros((N_STATES, len(ACTIONS))), EPSILON_START, 0


def save_table(path, Q, epsilon, steps):
    path.write_text(json.dumps({"Q": Q.round(4).tolist(), "epsilon": epsilon, "steps": steps,
                                "actions": ACTIONS, "base_speed": BASE_SPEED}, indent=1))


class LegoRobot:
    """Double Motor with IMU yaw, connected with the red 1142 Connection Card."""

    def __init__(self):
        import legoeducation as le
        from lelib import doubleMotor
        self.le = le
        print(f"Connecting to LEGO Double Motor ({CARD_COLOR} {CARD_SERIAL} card)...")
        self.motor = doubleMotor()
        self.motor.connect(card_serial=CARD_SERIAL, card_color=le.LEGO_COLOR_RED)
        print("Connected.")

    def reset_heading(self):
        self.motor.reset_heading()
        time.sleep(0.2)

    def yaw_error(self):
        return wrap(self.motor.yaw() - 0.0)     # ψ_err = ψ - 0°

    def drive(self, left, right):
        # One tank command sets both sides at once (two separate motor_run calls start the
        # left wheel slightly first, which twists the robot). It also handles the mirrored
        # right motor, so (25, 25) drives straight forward.
        self.motor.movement_move_tank(int(left), int(right), blocking=False)

    def stop(self):
        self.motor.movement_stop()

    def close(self):
        try:
            self.stop()
        finally:
            self.motor.disconnect()


class SimRobot:
    """Pretend robot for testing without LEGO: it naturally drifts right, the speed
    difference turns it, and there's some noise. Not physics -- just enough to learn on."""

    DRIFT = 4.0          # degrees per step it veers right on its own
    TURN = 0.4           # degrees per step per % of speed difference (right - left)

    def __init__(self):
        self.yaw = 0.0
        self.left = self.right = 0

    def reset_heading(self):
        self.yaw = random.uniform(-25, 25)   # start each run at a random heading so every state gets visited

    def yaw_error(self):
        return wrap(self.yaw)

    def drive(self, left, right):
        self.left, self.right = left, right
        # positive yaw = drifted right (clockwise); right faster turns left (negative)
        self.yaw += self.DRIFT - self.TURN * (right - left) + random.gauss(0, 1.5)

    def stop(self):
        pass

    def close(self):
        pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", action="store_true", help="use the learned table only (no exploring/learning)")
    parser.add_argument("--sim", action="store_true", help="simulated robot, no LEGO needed")
    parser.add_argument("--reset", action="store_true", help="start from a table of zeros")
    parser.add_argument("--episodes", type=int, default=20)
    args = parser.parse_args()

    table = SIM_TABLE_FILE if args.sim else TABLE_FILE
    Q, epsilon, steps = load_table(table, args.reset)
    if args.run:
        epsilon = 0.0
    robot = SimRobot() if args.sim else LegoRobot()

    try:
        for episode in range(1, args.episodes + 1):
            if not args.sim:
                input(f"\nEpisode {episode}: put the robot at the start, pointing where it should go, then press Enter...")
            robot.reset_heading()

            e = robot.yaw_error()
            s = state_of(e)
            total = 0.0
            for step in range(STEPS_PER_EPISODE):
                # 3. choose -- the first step is always straight, both sides at the same speed
                a = STRAIGHT if step == 0 else choose_action(Q, s, epsilon)
                robot.drive(*speeds_for(a))                      # 4. execute
                if not args.sim:
                    time.sleep(STEP_TIME)
                e = robot.yaw_error()                            # 5. observe
                s_next = state_of(e)
                r = reward_of(e)
                if not args.run:
                    update(Q, s, a, r, s_next)                   # 6. update Q
                    epsilon = max(EPSILON_MIN, epsilon * EPSILON_DECAY)   # 7. decay ε
                    steps += 1
                total += r
                s = s_next                                       # 8. repeat from the new state

            robot.stop()
            print(f"Episode {episode}: total reward {total:+.0f} / {STEPS_PER_EPISODE}, "
                  f"final error {e:+.1f}°, ε = {epsilon:.2f}")
            if not args.run:
                save_table(table, Q, epsilon, steps)
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        robot.close()
        if not args.run:
            save_table(table, Q, epsilon, steps)
            print(f"Saved {table.name}")
        print_table(Q)


if __name__ == "__main__":
    main()
