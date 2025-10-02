import random

# Initialize robot position (center of the grid)
robot_pos = [2, 0]  # [x, y] coordinates
iceCream = [1,4]
iceCream2 = [1,2]
def display_grid():
    """Display the 5x5 grid with the robot's current position"""
    print("\n" + "="*30)
    for y in range(5):
        row = ""
        for x in range(5):
            if x == robot_pos[0] and y == robot_pos[1]:
                row += " R "  # Robot position
            elif x == iceCream[0] and y == iceCream[1]:
                row += " I "  # Robot position:
            elif x == iceCream2[0] and y == iceCream2[1]:
                row += " I "  # Robot position:
            else:
                row += " . "  # Empty cell
        print(row)
    print(f"Robot position: ({robot_pos[0]}, {robot_pos[1]})")
    print("="*30)

def move_robot(intended_direction):
    """Move the robot with 30% chance of not following user input"""
    # 30% chance the robot disobeys
    if random.random() < 0.3:
        # Robot chooses a random direction (including staying)
        possible_actions = ["up", "down", "left", "right", "stay"]
        actual_direction = random.choice(possible_actions)
        print(f"Robot disobeyed! Instead of {intended_direction.upper()}, it went {actual_direction.upper()}")
    else:
        # Robot follows the command
        actual_direction = intended_direction
        print(f"Robot followed command: {actual_direction.upper()}")

    # Execute the actual movement
    if actual_direction == "up" and robot_pos[1] > 0:
        robot_pos[1] -= 1
    elif actual_direction == "down" and robot_pos[1] < 4:
        robot_pos[1] += 1
    elif actual_direction == "left" and robot_pos[0] > 0:
        robot_pos[0] -= 1
    elif actual_direction == "right" and robot_pos[0] < 4:
        robot_pos[0] += 1
    elif actualdirection =="tay":
        robot_pos[0] = robot_pos[0]
        robot_pos[1] = robot_pos[1]

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
        else:
            print("Invalid command! Use: up, down, left, right, stay, quit")
        

# Run the program
if __name__ == "__main__":
    main()