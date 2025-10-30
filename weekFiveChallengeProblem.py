# import random
# import numpy as np
# # System parameters (global)
# m = 1.0  # mass of the particle
# phi = lambda y: 0  # potential field function (example: quadratic potential)

# # Simulation parameters (global)
# time_steps = 10
# f_i = 0.5  # constant applied force

# # Initial conditions (global)
# y_0 = 0.0  # initial position
# v_0 = 0.0  # initial velocity
# v_max = 10.0  # maximum velocity
# p_c = 0.1  # crash probability factor


# class State:
#     def __init__(self, y: float, v: float):
#         self.y = y
#         self.v = v

#     def __str__(self):
#         return f"State(y={self.y:.3g}, v={self.v:.3g})"


# def update_state(state: State, f_i: float):
#     """Update the state of the particle based on the state equations.

#     Args:
#         state: Current state (y, v)
#         f_i: Applied force (input)
#     """
#     f_net = compute_net_force(f_i, state.y)
#     state.y = update_position(state.y, state.v)
#     state.v = update_velocity(state.v, f_net)

    


# def update_position(y: float, v: float) -> float:
#     return y + v

# def noisy_position(y: float) -> float:
#     mu = 0
#     sigma = 0.5 * y
#     return y + random.gauss(mu, sigma)

# def print_state(t, x: State):
#     y_noisy = noisy_position(x.y)
#     print(t, f"State(y={x.y:.3g}, v={x.v:.3g}, y_noisy={y_noisy:.3g})")


# def update_velocity(v: float, f_net: float) -> float:
#     """Update velocity: v'(t) = v(t) + (1/m) * f_net(t)

#     For pure particle without potential field: f_net(t) = f_i(t)
#     """
#     crash_prob = p_c * abs(v)/v_max
#     normal_dist = np.random.uniform()
#     if normal_dist <= crash_prob:
#         v_n = v + 1/m * f_net + random.gauss(0, 0.1 * v)
#     else :
#         v_n = 0
#     return v_n


# def compute_net_force(f_i: float, y: float) -> float:
#     """Compute net force: f_net(t) = f_i(t) - d/dy(phi(y))

#     Args:
#         f_i: Applied force (input)
#         y: Current position

#     Returns:
#         Net force acting on the particle
#     """
#     # Numerical derivative of phi with respect to y
#     return f_i 


# def main():
#     # Initialize state
#     state = State(y_0, v_0)

#     print("Pure Particle Dynamics Simulation")
#     print("=" * 50)
#     print(f"Initial state: {state}")
#     print(f"Mass: {m:.3g}, Applied force: {f_i:.3g}")
#     print(f"Potential field: phi(y) = {phi(y_0):.3g} * y^2")
#     print("=" * 50)

#     # Simulation loop
#     for t in range(time_steps):
#         print_state(t, state)
#         update_state(state, f_i)

#     print(f"t={time_steps}: {state}")


# if __name__ == "__main__":
#     main()

import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# Compute LQR gain using explicit Riccati recursion
# -------------------------------------------------------
def compute_lqr_gain(A, B, Q, R, H=200, tol=1e-8):
    if isinstance(R, (int, float)):
        R = np.array([[R]])
    P = Q.copy()

    for _ in range(H):
        K = np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)
        P_new = Q + K.T @ R @ K + (A - B @ K).T @ P @ (A - B @ K)
        if np.linalg.norm(P_new - P, ord='fro') < tol:
            break
        P = P_new

    return K, P

# -------------------------------------------------------
# Simulate x[t+1] = A x[t] + B u[t] + noise
# with noise only on the velocity component
# -------------------------------------------------------
def simulate(A, B, K, x0, sigma_d=0.1, T=50):
    n = A.shape[0]
    x_hist = np.zeros((n, T+1))
    u_hist = np.zeros(T)
    x = x0.copy()

    for t in range(T):
        u = -K @ x
        noise = np.array([[0], [np.random.normal(0, sigma_d)]])  # AWGN on velocity
        x = A @ x + B @ u + noise
        x_hist[:, t] = x.flatten()
        u_hist[t] = u
    x_hist[:, T] = x.flatten()

    return x_hist, u_hist

# -------------------------------------------------------
# Main experiment
# -------------------------------------------------------
A = np.array([[1, 1],
              [0, 1]])
B = np.array([[0],
              [1]])

x0 = np.array([[5.0],
               [2.0]])

sigma_d = 0.1  # process noise on velocity

configs = [
    {"name": "Q = [1,1], R = 1 (Balanced Q & R)", "Q": np.diag([1, 1]), "R": 1.0},
    {"name": "Q = [10,10], R = 0.1 (High Q & Low R)", "Q": np.diag([10, 10]), "R": 0.1},
    {"name": "Q = [1,1], R = 10 (Low Q & High R)", "Q": np.diag([1, 1]), "R": 10.0},
    {"name": "Q = [10,1], R = 1 (Position-Prioritized Q)", "Q": np.diag([10, 1]), "R": 1.0}
]

plt.figure(figsize=(10, 8))
for cfg in configs:
    K, _ = compute_lqr_gain(A, B, cfg["Q"], cfg["R"])
    x_hist, u_hist = simulate(A, B, K, x0, sigma_d=sigma_d, T=50)
    t = np.arange(len(u_hist))
    plt.plot(t, u_hist, label=f"{cfg['name']}")

plt.title(f"Control Input u(t) with σ_d = {sigma_d}")
plt.xlabel("Time step")
plt.ylabel("Control input u")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

# -------------------------------------------------------
# Optional: plot one full trajectory (y, v, u)
# -------------------------------------------------------
Q, R = np.diag([1, 1]), 1.0
K, _ = compute_lqr_gain(A, B, Q, R)
x_hist, u_hist = simulate(A, B, K, x0, sigma_d=sigma_d, T=50)
t = np.arange(len(u_hist)+1)

plt.figure(figsize=(10, 6))
plt.plot(t, x_hist[0, :], 'b-', label='Position y(t)')
plt.plot(t, x_hist[1, :], 'g-', label='Velocity v(t)')
plt.plot(t[:-1], u_hist, 'r--', label='Control u(t)')
plt.axhline(0, color='k', linestyle=':', alpha=0.5)
plt.title(f"State and Control Evolution (σ_d = {sigma_d})")
plt.xlabel("Time step")
plt.ylabel("Value")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
