import random
import numpy as np
# Initialize robot position (center of the grid)
robot_pos = [2, 0]  # [x, y] coordinates
iceCream = [2,4]
iceCream2 = [2,2]

# Define rewards and forbidden states
forbidden_state = [[1,1],[2,1],[1,3],[2,3]]
RW_loc = [[4,0],[4,1],[4,2],[4,3],[4,4]]
RD_loc = [[2,2]]
RS_loc = [[2,4]]
RW_val = -1
RD_val = 1
RS_val = 10
r = 0 
Pe = 0.3

def display_grid():
    """Display the 5x5 grid with robot, rewards, and forbidden states"""
    print("\n" + "="*30)
    for y in range(5):
        row = ""
        for x in range(5):
            pos = [x, y]

            # Priority: Robot overrides any other symbol
            if pos == robot_pos:
                cell = " R "
            elif pos in RD_loc:
                cell = "RD "
            elif pos in RS_loc:
                cell = "RS "
            elif pos in RW_loc:
                cell = "RW "
            elif pos in forbidden_state:
                cell = " X "
            else:
                cell = " . "
            row += cell
        print(row)
    print(f"Robot position: ({robot_pos[0]}, {robot_pos[1]})")
    print("="*30)

    
def move_robot(a):
    """Move the robot with 30% chance of not following user input"""
    if random.random() < Pe:
        # Robot disobeys and picks a random action
        possible_actions = ["up", "down", "left", "right", "stay"]
        possible_actions.remove(a)
        actual_direction = random.choice(possible_actions)
        print(f"Robot disobeyed! Instead of {a.upper()}, it went {actual_direction.upper()}")
    else:
        actual_direction = a
        print(f"Robot followed command: {actual_direction.upper()}")

    # Execute the actual movemendt
    global robot_pos
    old_pos = robot_pos.copy()

    if actual_direction== "up" and robot_pos[1] > 0:
        robot_pos[1] -= 1
    elif actual_direction == "down" and robot_pos[1] < 4:
        robot_pos[1] += 1
    elif actual_direction == "left" and robot_pos[0] > 0:
        robot_pos[0] -= 1
    elif actual_direction == "right" and robot_pos[0] < 4:
        robot_pos[0] += 1
    elif actual_direction =="tay":
        robot_pos[0] = robot_pos[0]
        robot_pos[1] = robot_pos[1]
    for state in forbidden_state:
        if robot_pos == state:
            robot_pos = old_pos.copy()
            print("Uh oh, robot went to a forbidden place!")

def P(next_state, s_prev, a):
    """
    Transition probability P(next_state | s_prev, a)
    where s_prev is ANY state, not the actual robot_pos.
    """
    intended_next = s_prev.copy()

    # Compute intended move from s_prev (not robot_pos!)
    if a == "up" and s_prev[1] > 0:
        intended_next[1] -= 1
    elif a == "down" and s_prev[1] < 4:
        intended_next[1] += 1
    elif a == "left" and s_prev[0] > 0:
        intended_next[0] -= 1
    elif a == "right" and s_prev[0] < 4:
        intended_next[0] += 1
    # 'stay' leaves intended_next unchanged

    # Probability distribution:
    #  - intended move: 1 - Pe
    #  - four unintended moves: Pe/4 each
    if next_state == intended_next:
        return 1 - Pe
    else:
        return Pe / 4


def compute_o():
    # Ensure inputs are numpy arrays
    curr_pos = np.array(robot_pos)
    R_D_pos = np.array(iceCream)
    R_S_pos = np.array(iceCream2)

    # Distances
    d_D = np.linalg.norm(curr_pos - R_D_pos)  # Euclidean distance
    d_S = np.linalg.norm(curr_pos - R_S_pos)

    # Harmonic mean
    if d_D == 0 or d_S == 0:
        # Avoid division by zero if position coincides with one of the reference points
        h = 0
    else:
        h = 2 / (1/d_D + 1/d_S)
        #print(h)

    # Probabilistic rounding
    ceil_h = np.ceil(h)
    floor_h = np.floor(h)
    prob_floor = ceil_h - h  # probability of rounding down

    if np.random.rand() < prob_floor:
        o = int(floor_h)
    else:
        o = int(ceil_h)
    return o

def check_rewards():
    """Check if the robot has reached a reward location and update total reward."""
    global r  # use the global reward variable
    
    if robot_pos in RW_loc:
        r = RW_val
        print(f"Robot reached RW location! Reward = {RW_val}. Total reward: {r}")
    elif robot_pos in RD_loc:
        r = RD_val
        print(f"Robot reached RD location! Reward = {RD_val}. Total reward: {r}")
    elif robot_pos in RS_loc:
        r = RS_val
        print(f"Robot reached RS location! Reward = {RS_val}. Total reward: {r}")

def P_o_given_s(o, s):
    # Convert to numpy arrays for distance math
    curr_pos = np.array(s)
    R_D_pos = np.array(iceCream)
    R_S_pos = np.array(iceCream2)

    # Compute harmonic mean h = 2 / (d_D^-1 + d_S^-1)
    d_D = np.linalg.norm(curr_pos - R_D_pos)
    d_S = np.linalg.norm(curr_pos - R_S_pos)

    if d_D == 0 or d_S == 0:
        h = 0
    else:
        h = 2 / (1/d_D + 1/d_S)

    ceil_h = np.ceil(h)
    floor_h = np.floor(h)

    # Probabilities according to your rule
    if np.isclose(o, ceil_h):        # observation equals ceil(h)
        return 1 - (ceil_h - h)
    elif np.isclose(o, floor_h):     # observation equals floor(h)
        return (ceil_h - h)
    else:
        return 0.0

def P_s(s):
    start_state = [2, 0]
    return 1.0 if s == start_state else 0.0

def P_o(o):
    total = 0.0
    for x in range(5):
        for y in range(5):
            s = [x, y]
            # Skip forbidden states if you like (optional)
            if s in forbidden_state:
                continue
            total += P_o_given_s(o, s) * P_s(s)
    return total

def P_s_given_o(s, o):
    """
    Compute P(s | o) = (P(o|s) * P(s)) / P(o)
    """
    numerator = P_o_given_s(o, s) * P_s(s)
    denominator = P_o(o)
    if denominator == 0:
        return 0.0
    return numerator / denominator

def Bel_t_update(prev_bel, action, observation):
    """
    Update the belief state Bel_t given the previous belief, action, and new observation.

    prev_bel: dict mapping tuple(state) -> probability
    action: string ("up", "down", "left", "right", "stay")
    observation: numeric value (the observed o)
    """
    # Define all possible states
    states = [[x, y] for x in range(5) for y in range(5) if [x, y] not in forbidden_state]

    # ----- Prediction step: Bel_t^-(s_t) -----
    bel_pred = {}
    for s_next in states:
        total = 0.0
        for s_prev in states:
            total += P(s_next, s_prev, action) * prev_bel[tuple(s_prev)]
        bel_pred[tuple(s_next)] = total

    # Normalize predicted belief
    norm_pred = sum(bel_pred.values())
    if norm_pred > 0:
        for s in bel_pred:
            bel_pred[s] /= norm_pred

    # ----- Correction step: Bel_t(s_t) -----
    bel_new = {}
    for s in states:
        bel_new[tuple(s)] = P_o_given_s(observation, s) * bel_pred[tuple(s)]

    # Normalize to make it a valid probability distribution
    norm = sum(bel_new.values())
    if norm > 0:
        for s in bel_new:
            bel_new[s] /= norm

    return bel_new

def display_belief(bel):
    """
    Display the belief distribution Bel(s) as a 5x5 grid.
    Each cell shows the probability of being in that state.
    """
    print("\nBelief Distribution (P(s)):")
    print("=" * 40)
    for y in range(5):
        row = ""
        for x in range(5):
            s = (x, y)
            if [x, y] in forbidden_state:
                cell = "  X   "
            else:
                prob = bel.get(s, 0.0)
                cell = f"{prob:5.2f} "
            row += cell
        print(row)
    print("=" * 40)
        
def main():
    print("Robot Grid Movement System")
    print("WARNING: Robot has 30% chance to disobey commands!")
    print("Commands: up, down, left, right, stay, quit")

    # ---- Initialize belief ----
    # Initial observation (you could use a real compute_o() value)
    o0 = 2
    Bel = {tuple([x, y]): P_s_given_o([x, y], o0)
           for x in range(5) for y in range(5)
           if [x, y] not in forbidden_state}

    while True:
        display_grid()
        display_belief(Bel)  # <-- show belief right below your grid

        command = input("\nEnter command: ").strip().lower()

        if command == "quit":
            print("Goodbye!")
            break
        elif command in ["up", "down", "left", "right", "stay"]:
            move_robot(command)
            check_rewards()

            # Compute observation automatically
            new_o = compute_o()

            # ---- Belief update ----
            Bel = Bel_t_update(Bel, command, new_o)

        else:
            print("Invalid command! Use: up, down, left, right, stay, quit")


# Run the program
if __name__ == "__main__":
    main()