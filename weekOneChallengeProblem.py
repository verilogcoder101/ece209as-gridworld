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

    
def move_robot(intended_direction):
    """Move the robot with 30% chance of not following user input"""
    # 30% chance the robot disobeys
    if random.random() < 0.3:
        # Robot chooses a random direction (including staying)
        possible_actions = ["up", "down", "left", "right", "tay"]
        possible_actions.remove(intended_direction)
        actual_direction = random.choice(possible_actions)
        print(f"Robot disobeyed! Instead of {intended_direction.upper()}, it went {actual_direction.upper()}")
    else:
        # Robot follows the command
        actual_direction = intended_direction
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
        print(h)

    # Probabilistic rounding
    ceil_h = np.ceil(h)
    floor_h = np.floor(h)
    prob_floor = ceil_h - h  # probability of rounding down

    if np.random.rand() < prob_floor:
        print (int(floor_h))
    else:
        print (int(ceil_h))

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
            check_rewards()
            compute_o()
        else:
            print("Invalid command! Use: up, down, left, right, stay, quit")
        


# Run the program
if __name__ == "__main__":
    main()