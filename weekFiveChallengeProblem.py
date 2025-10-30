
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
# Second figure: individual trajectories for each config
# -------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for idx, cfg in enumerate(configs):
    Q, R = cfg["Q"], cfg["R"]
    K, _ = compute_lqr_gain(A, B, Q, R)
    x_hist, u_hist = simulate(A, B, K, x0, sigma_d=sigma_d, T=50)
    t = np.arange(len(u_hist)+1)

    ax = axes[idx]
    ax.plot(t, x_hist[0, :], 'b-', label='Position y(t)')
    ax.plot(t, x_hist[1, :], 'g-', label='Velocity v(t)')
    ax.plot(t[:-1], u_hist, 'r--', label='Control u(t)')
    ax.axhline(0, color='k', linestyle=':', alpha=0.5)

    # Handle R as float or array
    R_val = R if np.isscalar(R) else R[0,0]

    ax.set_title(
        f"{cfg['name']}\nσ_d={sigma_d}, Q=[{Q[0,0]}, {Q[1,1]}], R={R_val}"
    )
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()