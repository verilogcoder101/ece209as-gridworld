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
from scipy.linalg import solve_discrete_are

class ParticleOnNumberline:
    """
    Week 5: LQR Control for Particle on Numberline
    
    Simplified system:
    - No potential field (phi = 0)
    - Only AWGN on velocity
    - No input bounds
    - Full state knowledge
    """
    
    def __init__(self, dt=1.0, sigma_d=0.1):
        """
        Initialize the system.
        
        Parameters:
        -----------
        dt : float
            Time step for discrete dynamics
        sigma_d : float
            Standard deviation of velocity noise
        """
        self.dt = dt
        self.sigma_d = sigma_d
        
        # System matrices: x[t+1] = Ax[t] + Bu[t] + noise
        # State x = [y, v]'
        self.A = np.array([[1, dt],
                           [0, 1]])
        self.B = np.array([[0],
                           [dt]])
        
        self.K = None  # LQR gain matrix
        self.P = None  # Solution to DARE
        
    def dynamics(self, y, v, f_i, add_noise=True):
        """
        System dynamics with optional noise.
        
        Parameters:
        -----------
        y : float
            Current position
        v : float
            Current velocity
        f_i : float
            Input force
        add_noise : bool
            Whether to add process noise
            
        Returns:
        --------
        y_next, v_next : tuple of floats
        """
        # Process noise
        d = np.random.normal(0, self.sigma_d) if add_noise else 0
        
        # Discrete dynamics
        y_next = y + v * self.dt
        v_next = v + f_i * self.dt + d
        
        return y_next, v_next
    
    def compute_lqr_gain(self, Q, R, H=200, tol=1e-8):
        """
        Compute LQR gain matrix K by iteratively solving the Riccati recursion.

        Parameters
        ----------
        Q : np.array (2x2)
            State cost matrix
        R : np.array (1x1) or float
            Control cost
        H : int
            Maximum number of iterations (horizon length)
        tol : float
            Convergence tolerance for P

        Returns
        -------
        K : np.array (1x2)
            LQR gain matrix
        """
        if isinstance(R, (int, float)):
            R = np.array([[R]])

        # Initialize
        P = Q.copy()
        
        for i in range(H):
            # Compute K from current P
            K = np.linalg.inv(R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)

            # Riccati recursion
            P_new = Q + K.T @ R @ K + (self.A - self.B @ K).T @ P @ (self.A - self.B @ K)

            # Check for convergence
            if np.linalg.norm(P_new - P, ord='fro') < tol:
                print(f"Converged at iteration {i}")
                P = P_new
                break

            P = P_new

        # Store results
        self.P = P
        self.K = np.linalg.inv(R + self.B.T @ P @ self.B) @ (self.B.T @ P @ self.A)
        return self.K

    
    def control(self, y, v, y_d=0.0, v_d=0.0):
        """
        Compute LQR control input.
        
        Parameters:
        -----------
        y, v : float
            Current state
        y_d, v_d : float
            Desired state
            
        Returns:
        --------
        u : float
            Control input
        """
        if self.K is None:
            raise ValueError("LQR gain not computed. Call compute_lqr_gain() first.")
        
        # State error
        x_error = np.array([[y - y_d],
                            [v - v_d]])
        
        # Control law: u = -Kx
        u = -(self.K @ x_error)[0, 0]
        
        return u
    
    def simulate(self, y0, v0, y_d=0.0, v_d=0.0, T=100):
        """
        Simulate closed-loop system with LQR control.
        
        Parameters:
        -----------
        y0, v0 : float
            Initial state
        y_d, v_d : float
            Desired state
        T : int
            Number of time steps
            
        Returns:
        --------
        trajectory : dict
            Dictionary containing time series of state and control
        """
        if self.K is None:
            raise ValueError("LQR gain not computed. Call compute_lqr_gain() first.")
        
        # Initialize storage
        trajectory = {
            't': np.zeros(T+1),
            'y': np.zeros(T+1),
            'v': np.zeros(T+1),
            'u': np.zeros(T+1)
        }
        
        # Initial conditions
        y, v = y0, v0
        trajectory['y'][0] = y
        trajectory['v'][0] = v
        
        # Simulate
        for t in range(T):
            trajectory['t'][t] = t
            
            # Compute control
            u = self.control(y, v, y_d, v_d)
            trajectory['u'][t] = u
            
            # Apply dynamics
            y, v = self.dynamics(y, v, u)
            
            # Store
            trajectory['y'][t+1] = y
            trajectory['v'][t+1] = v
        
        trajectory['t'][T] = T
        
        return trajectory
    
    def plot_results(self, trajectory, y_d=0.0, v_d=0.0):
        """
        Plot simulation results.
        
        Parameters:
        -----------
        trajectory : dict
            Trajectory data from simulate()
        y_d, v_d : float
            Desired state for reference lines
        """
        fig, axes = plt.subplots(3, 1, figsize=(10, 8))
        
        # Position
        axes[0].plot(trajectory['t'], trajectory['y'], 'b-', label='Position y(t)', linewidth=2)
        axes[0].axhline(y=y_d, color='b', linestyle='--', alpha=0.5, label=f'Goal: y_d = {y_d}')
        axes[0].set_ylabel('Position (y)')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_title('LQR Control: Particle on Numberline')
        
        # Velocity
        axes[1].plot(trajectory['t'], trajectory['v'], 'g-', label='Velocity v(t)', linewidth=2)
        axes[1].axhline(y=v_d, color='g', linestyle='--', alpha=0.5, label=f'Goal: v_d = {v_d}')
        axes[1].set_ylabel('Velocity (v)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Control input
        axes[2].plot(trajectory['t'][:-1], trajectory['u'][:-1], 'r-', label='Control input u(t)', linewidth=2)
        axes[2].set_ylabel('Force (f_i)')
        axes[2].set_xlabel('Time Step')
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        plt.tight_layout()
        return fig


# ============================================================================
# EXAMPLE USAGE AND EXPERIMENTS
# ============================================================================

def part1_drive_to_origin():
    """
    Part 1: Drive system to rest at origin (y_d = 0, v_d = 0)
    Explore effects of varying Q and R.
    """
    print("=" * 70)
    print("PART 1: Drive to Rest at Origin")
    print("=" * 70)
    
    system = ParticleOnNumberline(dt=1.0, sigma_d=0.1)
    
    # Initial conditions
    y0, v0 = 5.0, 2.0
    y_d, v_d = 0.0, 0.0
    
    # Different Q and R configurations
    configs = [
        {"name": "Balanced", "Q_y": 1.0, "Q_v": 1.0, "R": 1.0},
        {"name": "High state cost (aggressive)", "Q_y": 10.0, "Q_v": 10.0, "R": 0.1},
        {"name": "High control cost (smooth)", "Q_y": 1.0, "Q_v": 1.0, "R": 10.0},
        {"name": "Prioritize position", "Q_y": 10.0, "Q_v": 1.0, "R": 1.0},
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, config in enumerate(configs):
        print(f"\n{config['name']}: Q_y={config['Q_y']}, Q_v={config['Q_v']}, R={config['R']}")
        
        # Setup LQR
        Q = np.diag([config['Q_y'], config['Q_v']])
        R = config['R']
        K = system.compute_lqr_gain(Q, R)
        
        print(f"  Gain K = [{K[0,0]:.4f}, {K[0,1]:.4f}]")
        
        # Simulate
        traj = system.simulate(y0, v0, y_d, v_d, T=50)
        
        # Calculate max control
        max_u = np.max(np.abs(traj['u']))
        print(f"  Max |u| = {max_u:.4f}")
        
        # Plot on subplot
        ax = axes[idx]
        ax.plot(traj['t'], traj['y'], 'b-', label='y(t)', linewidth=2)
        ax.plot(traj['t'], traj['v'], 'g-', label='v(t)', linewidth=2)
        ax.plot(traj['t'][:-1], traj['u'][:-1], 'r--', label='u(t)', linewidth=1.5, alpha=0.7)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title(f"{config['name']}\nK=[{K[0,0]:.2f}, {K[0,1]:.2f}], max|u|={max_u:.2f}")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lqr_part1_comparison.png', dpi=150)
    print("\nPlot saved as 'lqr_part1_comparison.png'")
    plt.show()


def part2_bounded_control():
    """
    Part 2: Predict Q, R to ensure |f_i| < 1 for bounded initial states.
    Given |y[0]| <= y_max, v[0] = 0, find Q, R s.t. |u[t]| < 1 for all t.
    """
    print("\n" + "=" * 70)
    print("PART 2: Bounded Control Input")
    print("=" * 70)
    
    system = ParticleOnNumberline(dt=1.0, sigma_d=0.0)  # No noise for analysis
    
    y_max = 5.0
    y0_values = [y_max, y_max/2, -y_max]  # Test different initial positions
    v0 = 0.0
    y_d, v_d = 0.0, 0.0
    
    # Try different R values (increasing control cost)
    R_values = [0.1, 1.0, 5.0, 10.0]
    Q_y, Q_v = 1.0, 1.0
    Q = np.diag([Q_y, Q_v])
    
    print(f"\nTesting with |y[0]| <= {y_max}, v[0] = 0")
    print(f"Fixed Q: diag([{Q_y}, {Q_v}])")
    print("\nVarying R to find bounds on |u|:")
    print("-" * 70)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for idx, R in enumerate(R_values):
        K = system.compute_lqr_gain(Q, R)
        
        max_controls = []
        
        ax = axes[idx]
        
        for y0 in y0_values:
            traj = system.simulate(y0, v0, y_d, v_d, T=50)
            max_u = np.max(np.abs(traj['u']))
            max_controls.append(max_u)
            
            ax.plot(traj['t'][:-1], traj['u'][:-1], label=f'y[0]={y0:.1f}, max|u|={max_u:.3f}')
        
        # Show constraint
        ax.axhline(y=1, color='r', linestyle='--', linewidth=2, label='Constraint |u| < 1')
        ax.axhline(y=-1, color='r', linestyle='--', linewidth=2)
        
        overall_max = np.max(max_controls)
        satisfies = "✓ SATISFIES" if overall_max < 1.0 else "✗ VIOLATES"
        
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Control Input u')
        ax.set_title(f'R = {R:.1f}, K=[{K[0,0]:.3f}, {K[0,1]:.3f}]\nmax|u|={overall_max:.3f} {satisfies}')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([-2, 2])
        
        print(f"R = {R:5.1f}: K = [{K[0,0]:7.4f}, {K[0,1]:7.4f}], max|u| = {overall_max:.4f} {satisfies}")
    
    plt.tight_layout()
    plt.savefig('lqr_part2_bounded_control.png', dpi=150)
    print("\nPlot saved as 'lqr_part2_bounded_control.png'")
    plt.show()
    
    print("\n" + "=" * 70)
    print("ANALYTICAL PREDICTION:")
    print("For worst case y[0] = y_max, v[0] = 0, the initial control is:")
    print("  u[0] = -K @ [y_max, 0]' = -K[0,0] * y_max")
    print(f"To ensure |u[0]| < 1, we need: K[0,0] * {y_max} < 1")
    print(f"Therefore: K[0,0] < {1/y_max:.3f}")
    print("\nIncrease R to reduce K[0,0]. From results above, R >= 5.0 works.")
    print("=" * 70)


def additional_experiments():
    """
    Additional experiments for deeper understanding.
    """
    print("\n" + "=" * 70)
    print("ADDITIONAL EXPERIMENTS")
    print("=" * 70)
    
    system = ParticleOnNumberline(dt=1.0, sigma_d=0.1)
    
    # Experiment: Track non-zero goal
    print("\nExperiment: Track non-zero goal state")
    y0, v0 = 0.0, 0.0
    y_d, v_d = 10.0, 1.0
    
    Q = np.diag([1.0, 1.0])
    R = 1.0
    system.compute_lqr_gain(Q, R)
    
    traj = system.simulate(y0, v0, y_d, v_d, T=80)
    
    fig = system.plot_results(traj, y_d, v_d)
    plt.savefig('lqr_nonzero_goal.png', dpi=150)
    print("Plot saved as 'lqr_nonzero_goal.png'")
    plt.show()
    
    # Experiment: Effect of noise
    print("\nExperiment: Effect of process noise")
    noise_levels = [0.0, 0.05, 0.1, 0.2]
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    Q = np.diag([1.0, 1.0])
    R = 1.0
    
    for idx, sigma in enumerate(noise_levels):
        system_noise = ParticleOnNumberline(dt=1.0, sigma_d=sigma)
        system_noise.compute_lqr_gain(Q, R)
        
        traj = system_noise.simulate(5.0, 2.0, 0.0, 0.0, T=50)
        
        ax = axes[idx]
        ax.plot(traj['t'], traj['y'], 'b-', label='Position y', linewidth=2)
        ax.plot(traj['t'], traj['v'], 'g-', label='Velocity v', linewidth=2)
        ax.axhline(y=0, color='k', linestyle=':', alpha=0.3)
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Value')
        ax.set_title(f'Process Noise σ_d = {sigma}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('lqr_noise_effect.png', dpi=150)
    print("Plot saved as 'lqr_noise_effect.png'")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("ECE209AS - Week 5: Linear Quadratic Regulator")
    print("Particle on Numberline System")
    print("=" * 70)
    
    # Run experiments
    part1_drive_to_origin()
    part2_bounded_control()
    additional_experiments()
    
    print("\n" + "=" * 70)
    print("All experiments completed!")
    print("=" * 70)