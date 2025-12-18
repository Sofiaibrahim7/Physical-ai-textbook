# Chapter 10: Safety and Robustness in Physical AI

Safety and robustness are paramount concerns in Physical AI systems, where failures can result in physical damage, injury, or environmental harm. Unlike purely digital systems, physical AI agents operate in real environments with real consequences, making safety considerations fundamental to system design rather than optional add-ons.

## 10.1 Safety in Physical AI Systems

Safety in Physical AI encompasses multiple dimensions: operational safety (avoiding harm during normal operation), fail-safe behavior (graceful degradation when components fail), and robustness against environmental uncertainties. The challenge is to maintain safety while enabling complex, adaptive behaviors.

Formally, safety can be defined as ensuring the system remains within a set of safe states $S_{safe} \subset S$:

$$\forall t \geq 0, s_t \in S_{safe}$$

where $s_t$ is the system state at time $t$.

![Figure 10.1: Safety Framework for Physical AI Systems](placeholder)

## 10.2 Formal Methods for Safety

### 10.2.1 Control Barrier Functions

Control Barrier Functions (CBFs) provide a mathematical framework for ensuring safety constraints in control systems. A function $h: \mathbb{R}^n \to \mathbb{R}$ is a Control Barrier Function if:

$$\exists \alpha \in \mathcal{K}_\infty \text{ such that } \sup_{u \in U} [\nabla h(x)^T f(x) + \nabla h(x)^T g(x)u + \alpha(h(x))] \geq 0$$

where $f(x)$ and $g(x)$ define the system dynamics $\dot{x} = f(x) + g(x)u$.

```python
import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import minimize

class ControlBarrierFunction:
    def __init__(self, state_dim, safe_distance=0.5):
        self.state_dim = state_dim
        self.safe_distance = safe_distance

    def barrier_function(self, state):
        """Define barrier function h(x) > 0 means safe"""
        # For collision avoidance: h(x) = distance - safe_distance
        # Assuming state contains position information
        if self.state_dim >= 4:  # At least 2D position for each agent
            pos1 = state[:2]
            pos2 = state[2:4]
            distance = np.linalg.norm(pos1 - pos2)
            return distance - self.safe_distance
        return 1.0  # Default safe state

    def barrier_gradient(self, state):
        """Compute gradient of barrier function"""
        # Numerical gradient for simplicity
        eps = 1e-6
        grad = np.zeros_like(state)

        for i in range(len(state)):
            state_plus = state.copy()
            state_minus = state.copy()
            state_plus[i] += eps
            state_minus[i] -= eps

            grad[i] = (self.barrier_function(state_plus) -
                      self.barrier_function(state_minus)) / (2 * eps)

        return grad

class SafeController:
    def __init__(self, nominal_controller, cbf, alpha_gain=1.0):
        self.nominal_controller = nominal_controller
        self.cbf = cbf
        self.alpha_gain = alpha_gain

    def safe_control(self, state, unsafe_action=None):
        """Compute safe control action using CBF"""
        if unsafe_action is None:
            unsafe_action = self.nominal_controller(state)

        # Formulate optimization problem to find safe action
        def safety_constraint(action):
            grad_h = self.cbf.barrier_gradient(state)
            h_val = self.cbf.barrier_function(state)

            # CBF constraint: L_f h(x) + L_g h(x) u + alpha(h(x)) >= 0
            # For simplicity, assume L_g h(x) = 1 (direct control)
            constraint_val = grad_h @ unsafe_action + self.alpha_gain * max(0, h_val)
            return constraint_val

        # Optimization to find safe action closest to nominal
        def objective(action):
            return np.sum((action - unsafe_action) ** 2)

        # Use scipy optimization (in practice, this would be solved more efficiently)
        result = minimize(
            objective,
            x0=unsafe_action,
            constraints={'type': 'ineq', 'fun': safety_constraint},
            method='SLSQP'
        )

        return result.x if result.success else unsafe_action
```

### 10.2.2 Reachability Analysis

Reachability analysis determines the set of states that a system can reach from a given initial set, enabling safety verification:

```python
class ReachabilityAnalyzer:
    def __init__(self, dynamics_model, state_dim):
        self.dynamics_model = dynamics_model
        self.state_dim = state_dim

    def compute_reachable_set(self, initial_set, time_horizon, dt=0.01):
        """Compute reachable set using forward simulation"""
        reachable_states = []

        # Sample from initial set
        for _ in range(100):  # Sample 100 initial states
            init_state = self.sample_initial_set(initial_set)
            trajectory = [init_state]

            for t in range(int(time_horizon / dt)):
                current_state = trajectory[-1]
                # Apply dynamics
                next_state = self.dynamics_model(current_state)
                trajectory.append(next_state)

            reachable_states.extend(trajectory)

        return np.array(reachable_states)

    def sample_initial_set(self, initial_set_bounds):
        """Sample from initial set (assumed to be hyperrectangle)"""
        state = np.zeros(self.state_dim)
        for i in range(self.state_dim):
            state[i] = np.random.uniform(initial_set_bounds[i][0],
                                       initial_set_bounds[i][1])
        return state

    def verify_safety(self, reachable_set, unsafe_set):
        """Check if reachable set intersects unsafe set"""
        # Simple collision detection
        for state in reachable_set:
            if self.in_unsafe_set(state, unsafe_set):
                return False
        return True

    def in_unsafe_set(self, state, unsafe_set):
        """Check if state is in unsafe set"""
        # Example: circular unsafe region
        center = unsafe_set['center']
        radius = unsafe_set['radius']
        distance = np.linalg.norm(state[:2] - center)
        return distance < radius
```

## 10.3 Robust Control Design

### 10.3.1 H-infinity Control

H-infinity control minimizes the worst-case effect of disturbances on system performance:

```python
import control  # python-control package

class HInfinityController:
    def __init__(self, system_matrices, disturbance_weight=1.0, control_weight=1.0):
        """
        System: dx/dt = Ax + B1*w + B2*u
        Output: z = C1*x + D12*u, y = C2*x + D21*w
        """
        self.A = system_matrices['A']
        self.B1 = system_matrices['B1']  # Disturbance input
        self.B2 = system_matrices['B2']  # Control input
        self.C1 = system_matrices['C1']  # Performance output
        self.C2 = system_matrices['C2']  # Measurement output
        self.D12 = system_matrices['D12']
        self.D21 = system_matrices['D21']

        # Design weights
        self.W_disturbance = disturbance_weight
        self.W_control = control_weight

    def synthesize_controller(self, gamma=1.0):
        """Synthesize H-infinity controller"""
        # Augmented system for H-infinity design
        P = control.ss(self.A,
                      np.hstack([self.B1, self.B2]),
                      np.vstack([self.C1, self.C2]),
                      np.vstack([np.hstack([np.zeros((self.C1.shape[0], self.B1.shape[1])),
                                          self.D12]),
                               np.hstack([self.D21, np.zeros((self.C2.shape[0], self.B2.shape[1])))]))

        # Use control synthesis tools
        try:
            K, cl, info = control.hinfsyn(P, 1, 1)  # 1 control input, 1 disturbance input
            return K
        except:
            # Fallback to LQR if H-infinity fails
            return self.synthesize_lqr_fallback()

    def synthesize_lqr_fallback(self):
        """Fallback LQR controller"""
        Q = np.eye(self.A.shape[0])  # State cost
        R = self.W_control * np.eye(self.B2.shape[1])  # Control cost
        K, S, E = control.lqr(self.A, self.B2, Q, R)
        return K

class RobustNeuralController(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.nominal_policy = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Uncertainty estimator
        self.uncertainty_estimator = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)  # Uncertainty compensation
        )

    def forward(self, state, uncertainty_scale=1.0):
        nominal_action = self.nominal_policy(state)
        uncertainty_compensation = self.uncertainty_estimator(state)

        # Robust action with uncertainty compensation
        robust_action = nominal_action + uncertainty_scale * uncertainty_compensation
        return torch.tanh(robust_action)  # Ensure action bounds
```

## 10.4 Safe Exploration

Safe exploration is crucial during learning, as random exploration can lead to unsafe states in physical systems.

### 10.4.1 Constrained Exploration

```python
class SafeExplorationAgent:
    def __init__(self, policy_network, cbf, safety_threshold=0.1):
        self.policy_network = policy_network
        self.cbf = cbf
        self.safety_threshold = safety_threshold

    def safe_exploration_action(self, state, exploration_noise=0.1):
        """Generate exploration action while maintaining safety"""
        # Get nominal action from policy
        with torch.no_grad():
            nominal_action = self.policy_network(torch.FloatTensor(state)).numpy()

        # Add exploration noise
        noise = np.random.normal(0, exploration_noise, size=nominal_action.shape)
        unsafe_action = nominal_action + noise

        # Check if action maintains safety
        predicted_next_state = self.predict_next_state(state, unsafe_action)
        safety_margin = self.cbf.barrier_function(predicted_next_state)

        if safety_margin < self.safety_threshold:
            # Project to safe action space
            safe_action = self.project_to_safe_action(state, unsafe_action)
        else:
            safe_action = unsafe_action

        return safe_action

    def project_to_safe_action(self, state, unsafe_action):
        """Project action to safe space using optimization"""
        def safety_constraint(action):
            next_state = self.predict_next_state(state, action)
            return self.cbf.barrier_function(next_state) - self.safety_threshold

        def objective(action):
            return np.sum((action - unsafe_action) ** 2)

        result = minimize(
            objective,
            x0=unsafe_action,
            constraints={'type': 'ineq', 'fun': safety_constraint},
            method='SLSQP'
        )

        return result.x if result.success else unsafe_action

    def predict_next_state(self, state, action, dt=0.01):
        """Predict next state given action (simplified dynamics)"""
        # Placeholder: implement based on specific system dynamics
        next_state = state + dt * action  # Simplified Euler integration
        return next_state
```

## 10.5 Fault Detection and Tolerance

### 10.5.1 Anomaly Detection in Physical Systems

```python
class AnomalyDetector:
    def __init__(self, state_dim, threshold=0.05):
        self.state_dim = state_dim
        self.threshold = threshold
        self.normalizer = None

        # Autoencoder for anomaly detection
        self.autoencoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)
        )
        self.optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=1e-3)

    def train_normal_behavior(self, normal_data):
        """Train autoencoder on normal system behavior"""
        states = torch.FloatTensor(normal_data)

        for epoch in range(1000):
            reconstructed = self.autoencoder(states)
            loss = nn.MSELoss()(reconstructed, states)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def detect_anomaly(self, current_state):
        """Detect anomalies in current state"""
        state_tensor = torch.FloatTensor(current_state).unsqueeze(0)
        reconstructed = self.autoencoder(state_tensor)

        reconstruction_error = nn.MSELoss()(reconstructed, state_tensor)
        is_anomaly = reconstruction_error.item() > self.threshold

        return is_anomaly, reconstruction_error.item()

class FaultTolerantController:
    def __init__(self, nominal_controller, backup_controllers, anomaly_detector):
        self.nominal_controller = nominal_controller
        self.backup_controllers = backup_controllers
        self.anomaly_detector = anomaly_detector
        self.fault_detected = False
        self.active_controller_idx = 0

    def get_action(self, state):
        """Get action with fault tolerance"""
        is_anomaly, anomaly_score = self.anomaly_detector.detect_anomaly(state)

        if is_anomaly and not self.fault_detected:
            print(f"Fault detected with anomaly score: {anomaly_score}")
            self.fault_detected = True
            self.activate_backup_controller()

        if self.fault_detected:
            return self.backup_controllers[self.active_controller_idx](state)
        else:
            return self.nominal_controller(state)

    def activate_backup_controller(self):
        """Activate backup controller"""
        self.active_controller_idx = min(self.active_controller_idx + 1,
                                       len(self.backup_controllers) - 1)
```

## 10.6 Verification and Validation

### 10.6.1 Statistical Verification

```python
class StatisticalSafetyVerifier:
    def __init__(self, system_model, confidence_level=0.95):
        self.system_model = system_model
        self.confidence_level = confidence_level

    def verify_safety_probability(self, initial_state, time_horizon, num_samples=10000):
        """Estimate probability of safe operation"""
        safe_trajectories = 0

        for _ in range(num_samples):
            trajectory_safe = True
            state = initial_state.copy()

            for t in range(int(time_horizon / 0.01)):  # 0.01s time step
                action = self.system_model.get_safe_action(state)
                state = self.system_model.step(state, action)

                if not self.system_model.is_safe(state):
                    trajectory_safe = False
                    break

            if trajectory_safe:
                safe_trajectories += 1

        safety_probability = safe_trajectories / num_samples
        confidence_interval = self.compute_confidence_interval(
            safe_trajectories, num_samples
        )

        return safety_probability, confidence_interval

    def compute_confidence_interval(self, successes, n):
        """Compute confidence interval for binomial proportion"""
        p_hat = successes / n
        z = 1.96  # 95% confidence level

        se = np.sqrt(p_hat * (1 - p_hat) / n)
        margin_error = z * se

        lower = max(0, p_hat - margin_error)
        upper = min(1, p_hat + margin_error)

        return (lower, upper)
```

## 10.7 Practical Safety Implementations

### 10.7.1 Safety-Critical Control Architecture

```python
class SafetyCriticalController:
    def __init__(self, nominal_controller, safety_controller,
                 state_dim, action_dim):
        self.nominal_controller = nominal_controller
        self.safety_controller = safety_controller
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Safety monitor
        self.safety_monitor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Safety probability
        )
        self.safety_threshold = 0.8

    def get_action(self, state):
        """Get action with safety monitoring"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Get safety assessment
        safety_prob = self.safety_monitor(state_tensor).item()

        if safety_prob < self.safety_threshold:
            # Use safety controller
            action = self.safety_controller(state)
            print("Safety controller activated")
        else:
            # Use nominal controller
            action = self.nominal_controller(state)

        return action

    def update_safety_monitor(self, safe_states, unsafe_states):
        """Update safety monitor with new data"""
        safe_tensor = torch.FloatTensor(safe_states)
        unsafe_tensor = torch.FloatTensor(unsafe_states)

        # Labels: 1 for safe, 0 for unsafe
        safe_labels = torch.ones(len(safe_states), 1)
        unsafe_labels = torch.zeros(len(unsafe_states), 1)

        all_states = torch.cat([safe_tensor, unsafe_tensor])
        all_labels = torch.cat([safe_labels, unsafe_labels])

        optimizer = torch.optim.Adam(self.safety_monitor.parameters(), lr=1e-3)
        criterion = nn.BCELoss()

        for epoch in range(100):
            predictions = self.safety_monitor(all_states)
            loss = criterion(predictions, all_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

## Key Takeaways

- Safety in Physical AI requires formal methods like Control Barrier Functions to guarantee constraint satisfaction
- Robust control techniques (H-infinity, adaptive control) handle uncertainties and disturbances
- Safe exploration ensures learning doesn't violate safety constraints
- Fault detection and tolerance mechanisms maintain operation during component failures
- Statistical verification provides probabilistic safety guarantees
- Hierarchical safety architectures separate nominal control from safety-critical functions

## Exercises

1. **Coding**: Implement a Control Barrier Function for a simple 2D robot navigation task with obstacle avoidance.

2. **Theoretical**: Prove that a Control Barrier Function guarantees forward invariance of the safe set.

3. **Coding**: Design a fault-tolerant controller for a quadrotor and test its performance under actuator failures.

4. **Theoretical**: Explain how the composition of multiple safety constraints affects the feasible control space.

5. **Coding**: Implement statistical verification for a simple safety-critical system and analyze the confidence intervals.

## Further Reading

1. Ames, A. D., et al. (2019). "Control barrier functions: Theory and applications." *ECC*.

2. Zhu, Q., et al. (2020). "Safety-critical control of stochastic systems using risk-aware control barrier functions." *ACC*.

3. Fisac, J. F., et al. (2018). "A general safety framework for learning-based control in uncertain robotic systems." *TAC*.