import numpy as np
from itertools import product

# ------------------------------
# Environment definitions
# ------------------------------

forbidden_state = [[1,1],[2,1],[1,3],[2,3]]
RW_loc = [[4,0],[4,1],[4,2],[4,3],[4,4]]
RD_loc = [[2,2]]
RS_loc = [[2,4]]

RW_val = -1
RD_val = 1
RS_val = 10
R_crash = -1  # collision penalty

Pe = 0.3   # probability of disobedience
gamma = 0.2  # discount factor

actions = ["up", "down", "left", "right", "stay"]

# ------------------------------
# Helper: single-robot movement
# ------------------------------

def step_single(pos, act):
    """Return next position for a single robot applying action act."""
    x, y = pos
    nx, ny = x, y

    if act == "up" and y > 0:
        ny -= 1
    elif act == "down" and y < 4:
        ny += 1
    elif act == "left" and x > 0:
        nx -= 1
    elif act == "right" and x < 4:
        nx += 1
    elif act == "stay":
        pass  # no change

    candidate = [nx, ny]

    # Treat forbidden states as walls
    if candidate in forbidden_state:
        return tuple(pos)

    return tuple(candidate)

# ------------------------------
# Transition model P_single
# ------------------------------

def P_single(next_state, pos, act):
    """
    Local stochastic transition model for a single robot.
    With prob 1-Pe, robot attempts act.
    With prob Pe, robot takes one of the OTHER 4 actions.
    """
    intended = step_single(pos, act)

    # Compute probability from intended movement
    prob = 0.0
    if next_state == intended:
        prob += 1 - Pe

    # Error actions
    other_acts = [a for a in actions if a != act]
    err_prob = Pe / len(other_acts)  # always 4

    for err in other_acts:
        err_next = step_single(pos, err)
        if next_state == err_next:
            prob += err_prob

    return prob

# ------------------------------
# Joint transition model
# ------------------------------

def P_joint(next_state, state, action_pair):
    """
    P( (pos1',pos2') | (pos1,pos2), (a1,a2) )
    Since noise is independent, this factorizes.
    """
    (pos1, pos2) = state
    (next1, next2) = next_state
    (a1, a2) = action_pair

    p1 = P_single(next1, pos1, a1)
    p2 = P_single(next2, pos2, a2)

    return p1 * p2

# ------------------------------
# Reward function
# ------------------------------

def reward_single(pos):
    if list(pos) in RW_loc: return RW_val
    if list(pos) in RD_loc: return RD_val
    if list(pos) in RS_loc: return RS_val
    return 0

def r_joint(pos1, pos2):
    """Reward for the joint state (pos1,pos2), including crash penalty."""
    r = reward_single(pos1) + reward_single(pos2)
    if pos1 == pos2:
        r += R_crash
    return r

# ------------------------------
# Build joint state space
# ------------------------------

valid_cells = [
    (x, y)
    for x in range(5)
    for y in range(5)
    if [x, y] not in forbidden_state
]

joint_states = [(s1, s2) for s1 in valid_cells for s2 in valid_cells]

joint_actions = [(a1, a2) for a1 in actions for a2 in actions]
# 25 joint actions

# ------------------------------
# VALUE ITERATION
# ------------------------------

def value_iteration_two_robots():
    V = {state: 0.0 for state in joint_states}
    theta = 1e-6

    while True:
        delta = 0.0
        V_new = {}

        for s in joint_states:
            (pos1, pos2) = s

            best_q = -float("inf")

            # Bellman max over joint actions
            for a in joint_actions:
                q = 0.0
                for s_next in joint_states:
                    p = P_joint(s_next, s, a)
                    if p > 0:
                        r = r_joint(*s_next)
                        q += p * (r + gamma * V[s_next])

                if q > best_q:
                    best_q = q

            V_new[s] = best_q
            delta = max(delta, abs(V_new[s] - V[s]))

        V = V_new

        if delta < theta:
            break

    # Derive optimal policy π*(s)
    policy = {}

    for s in joint_states:
        best_a = None
        best_q = -float("inf")

        for a in joint_actions:
            q = 0.0
            for s_next in joint_states:
                p = P_joint(s_next, s, a)
                if p > 0:
                    r = r_joint(*s_next)
                    q += p * (r + gamma * V[s_next])

            if q > best_q:
                best_q = q
                best_a = a

        policy[s] = best_a

    return V, policy

# ------------------------------
# RUN VALUE ITERATION
# ------------------------------

if __name__ == "__main__":
    print("Running 2-Robot Value Iteration...\n")
    V_opt, policy_opt = value_iteration_two_robots()

    print(f"Computed {len(V_opt)} joint-state values.")
    print(f"Example policy entry:")
    example_state = ((0,0), (4,4))
    print(f"State {example_state}: best action = {policy_opt[example_state]}")
    print("")
    print("Done.")
