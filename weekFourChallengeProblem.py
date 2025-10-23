import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# -------------------------------
# Knight movement + graph setup
# -------------------------------

def is_valid(x, y):
    """Check if a position is within the 8x8 chessboard."""
    return 0 <= x < 8 and 0 <= y < 8

def knight_moves(x, y):
    """Return all valid knight moves from (x, y)."""
    moves = [
        (x + 2, y + 1), (x + 1, y + 2),
        (x - 1, y + 2), (x - 2, y + 1),
        (x - 2, y - 1), (x - 1, y - 2),
        (x + 1, y - 2), (x + 2, y - 1)
    ]
    return [(nx, ny) for nx, ny in moves if is_valid(nx, ny)]

def build_knight_graph():
    """Create a graph of knight moves on an 8x8 chessboard."""
    G = nx.Graph()
    for x in range(8):
        for y in range(8):
            for move in knight_moves(x, y):
                G.add_edge((x, y), move)
    return G

# -------------------------------
# BFS and DFS search algorithms
# -------------------------------

def bfs_path(G, start, goal):
    """Return the shortest path from start to goal using BFS."""
    queue = deque([(start, [start])])
    visited = set()

    while queue:
        node, path = queue.popleft()
        if node == goal:
            return path
        if node in visited:
            continue
        visited.add(node)
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                queue.append((neighbor, path + [neighbor]))
    return None

def dfs_path(G, start, goal):
    """Return a path (not necessarily shortest) from start to goal using DFS."""
    stack = [(start, [start])]
    visited = set()

    while stack:
        node, path = stack.pop()
        if node == goal:
            return path
        if node in visited:
            continue
        visited.add(node)
        for neighbor in G.neighbors(node):
            if neighbor not in visited:
                stack.append((neighbor, path + [neighbor]))
    return None

# -------------------------------
# Visualization
# -------------------------------

def draw_path(G, path, title):
    pos = {(x, y): (x, y) for x in range(8) for y in range(8)}
    plt.figure(figsize=(7, 7))
    nx.draw(G, pos, node_color='lightgray', node_size=300, with_labels=False)
    
    if path:
        nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='skyblue', node_size=500)
        nx.draw_networkx_edges(G, pos, edgelist=list(zip(path[:-1], path[1:])), width=2.5, edge_color='blue')
        nx.draw_networkx_labels(G, pos, {p: str(p) for p in path}, font_size=8)
    
    plt.title(title)
    plt.gca().invert_yaxis()
    plt.show()

# -------------------------------
# Main
# -------------------------------

if __name__ == "__main__":
    G = build_knight_graph()
    start = (0, 0)
    goal = (7, 7)

    bfs_result = bfs_path(G, start, goal)
    dfs_result = dfs_path(G, start, goal)

    print("BFS path:", bfs_result)
    print("DFS path:", dfs_result)

    draw_path(G, bfs_result, "Knight Path using BFS (Shortest Path)")
    draw_path(G, dfs_result, "Knight Path using DFS (Non-optimal Path)")
