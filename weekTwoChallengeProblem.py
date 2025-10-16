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

# Transition and discount parameters
Pe = 0.3   # Probability of error (disobedience)
gamma = 0.7  # Discount factor


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

    global robot_pos
    old_pos = robot_pos.copy()

    if actual_direction == "up" and robot_pos[1] > 0:
        robot_pos[1] -= 1
    elif actual_direction == "down" and robot_pos[1] < 4:
        robot_pos[1] += 1
    elif actual_direction == "left" and robot_pos[0] > 0:
        robot_pos[0] -= 1
    elif actual_direction == "right" and robot_pos[0] < 4:
        robot_pos[0] += 1
    elif actual_direction == "stay":
        pass  # No change

    # Prevent movement into forbidden states
    for state in forbidden_state:
        if robot_pos == state:
            robot_pos = old_pos.copy()
            print("Uh oh, robot went to a forbidden place!")


def compute_o():
    """Compute a harmonic mean-based observation measure (for testing)"""
    curr_pos = np.array(robot_pos)
    R_D_pos = np.array(iceCream)
    R_S_pos = np.array(iceCream2)

    # Distances
    d_D = np.linalg.norm(curr_pos - R_D_pos)
    d_S = np.linalg.norm(curr_pos - R_S_pos)

    # Harmonic mean
    if d_D == 0 or d_S == 0:
        h = 0
    else:
        h = 2 / (1/d_D + 1/d_S)
        print(h)

    # Probabilistic rounding
    ceil_h = np.ceil(h)
    floor_h = np.floor(h)
    prob_floor = ceil_h - h

    if np.random.rand() < prob_floor:
        print(int(floor_h))
    else:
        print(int(ceil_h))


def r(robot_pos):
    """
    Return the reward at the given robot position.
    Does not accumulate global rewards.
    """
    if robot_pos in RW_loc:
        return RW_val
    elif robot_pos in RD_loc:
        return RD_val
    elif robot_pos in RS_loc:
        return RS_val
    else:
        return 0  # No reward at this position



def P(next_state, robot_pos, a):
    """
    Compute the transition probability P(next_state | robot_pos, a).
    next_state, robot_pos: [x, y] lists
    a: action string ("up", "down", "left", "right", "stay")
    """
    intended_next = robot_pos.copy()

    if a == "up" and robot_pos[1] > 0:
        intended_next[1] -= 1
    elif a == "down" and robot_pos[1] < 4:
        intended_next[1] += 1
    elif a == "left" and robot_pos[0] > 0:
        intended_next[0] -= 1
    elif a == "right" and robot_pos[0] < 4:
        intended_next[0] += 1
    elif a == "stay":
        pass  # No change

    # If intended next state equals the queried next state
    if next_state == intended_next:
        return 1 - Pe
    else:
        return Pe / 4

def value_iteration():
    """
    Compute optimal value function V*(s) and policy pi*(s)
    using Bellman optimality backups.
    """
    actions = ["up", "down", "left", "right", "stay"]
    states = [[x, y] for x in range(5) for y in range(5)
              if [x, y] not in forbidden_state]

    # Initialize value function V_0(s) = 0 for all s
    V = {tuple(s): 0.0 for s in states}
    theta = 1e-6  # convergence threshold

    while True:
        delta = 0.0
        V_new = {}

        # Bellman optimality backup: V_{i+1}(s) = max_a Q(s,a)
        for s in states:
            s_tup = tuple(s)
            Q_values = []

            for a in actions:
                expected_value = 0.0
                for s_next in states:
                    p = P(s_next, s, a)
                    expected_value += p * (r(s) + gamma * V[tuple(s_next)])
                Q_values.append(expected_value)

            V_new[s_tup] = max(Q_values)  # Bellman backup
            delta = max(delta, abs(V_new[s_tup] - V[s_tup]))

        V = V_new.copy()

        # Stopping criterion
        if delta < theta:
            break

    # Derive Q*(s,a) and policy pi*(s)
    Q_star = {}
    policy = {}

    for s in states:
        s_tup = tuple(s)
        Q_star[s_tup] = {}

        best_action = None
        best_value = -float("inf")

        for a in actions:
            q_val = 0.0
            for s_next in states:
                p = P(s_next, s, a)
                q_val += p * (r(s) + gamma * V[tuple(s_next)])
            Q_star[s_tup][a] = q_val

            if q_val > best_value:
                best_value = q_val
                best_action = a

        policy[s_tup] = best_action

    return V, Q_star, policy


if __name__ == "__main__":
    print("Running Value Iteration...\n")
    V_opt, Q_opt, policy_opt = value_iteration()

    print("Optimal Value Function:")
    for y in range(5):
        row = ""
        for x in range(5):
            s = (x, y)
            if [x, y] in forbidden_state:
                row += "  X   "
            else:
                row += f"{V_opt[s]:5.2f} "
        print(row)

    print("\nOptimal Policy:")
    for y in range(5):
        row = ""
        for x in range(5):
            s = (x, y)
            if [x, y] in forbidden_state:
                row += "  X   "
            else:
                a = policy_opt[s]
                row += f"{a[0].upper()}   "  # print first letter of action
        print(row)