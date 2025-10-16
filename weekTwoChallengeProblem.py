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


def main():
    print("Robot Grid Movement System")
    print("WARNING: Robot has 30% chance to disobey commands!")
    print("Commands: up, down, left, right, stay, quit")
    
    while True:
        display_grid()
        
        command = input("\nEnter command: ").strip().lower()
        
        if command == "quit":
            print("Goodbye!")
            break
        elif command in ["up", "down", "left", "right", "stay"]:
            move_robot(command)
            compute_o()

            # Example demonstration of P:
            test_next = [robot_pos[0], robot_pos[1]]
            print(f"P({test_next} | current={robot_pos}, action='{command}') = {P(test_next, robot_pos, command)}")
        else:
            print("Invalid command! Use: up, down, left, right, stay, quit")

# Run the program
if __name__ == "__main__":
    main()