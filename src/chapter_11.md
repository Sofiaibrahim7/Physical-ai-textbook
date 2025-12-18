
# Chapter 11: Energy Efficiency and Sustainable Physical AI

Energy efficiency and sustainability are critical considerations in Physical AI systems, where computational and physical resources must be optimized to extend operational lifetime, reduce environmental impact, and enable deployment in resource-constrained environments. As Physical AI systems become more prevalent, sustainable design principles must be integrated from the ground up.

## 11.1 Energy-Aware Physical AI Systems

Energy efficiency in Physical AI encompasses both computational efficiency (minimizing processing power) and physical efficiency (optimizing movement, sensing, and actuation). The total energy consumption of a Physical AI system includes:

- **Computational Energy**: Energy for processing, learning, and decision-making
- **Sensing Energy**: Energy for data acquisition and processing
- **Actuation Energy**: Energy for physical movement and interaction
- **Communication Energy**: Energy for data transmission

The total energy consumption can be modeled as:

$$E_{total} = E_{comp} + E_{sensing} + E_{actuation} + E_{comm}$$

![Figure 11.1: Energy Consumption in Physical AI Systems](placeholder)

## 11.2 Energy-Efficient Computing Architectures

### 11.2.1 Neuromorphic Computing for Physical AI

Neuromorphic computing architectures mimic neural structures to achieve energy-efficient computation:

```python
import torch
import torch.nn as nn
import numpy as np

class SpikingNeuron(nn.Module):
    def __init__(self, threshold=1.0, decay=0.9):
        super().__init__()
        self.threshold = threshold
        self.decay = decay
        self.membrane_potential = 0.0

    def forward(self, input_current):
        # Update membrane potential
        self.membrane_potential = self.decay * self.membrane_potential + input_current

        # Generate spike if threshold exceeded
        spike = (self.membrane_potential >= self.threshold).float()

        # Reset potential after spike
        self.membrane_potential = self.membrane_potential * (1 - spike)

        return spike

class SpikingNeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_timesteps=10):
        super().__init__()
        self.num_timesteps = num_timesteps

        # Spiking layers
        self.input_layer = nn.Linear(input_size, hidden_size)
        self.hidden_layer = nn.Linear(hidden_size, hidden_size)
        self.output_layer = nn.Linear(hidden_size, output_size)

        # Spiking neurons
        self.spiking_neurons = nn.ModuleList([
            SpikingNeuron() for _ in range(hidden_size + output_size)
        ])

    def forward(self, x):
        spike_count = torch.zeros(x.shape[0], self.output_layer.out_features)

        for t in range(self.num_timesteps):
            # Process through layers
            hidden_input = torch.relu(self.input_layer(x))
            hidden_output = torch.relu(self.hidden_layer(hidden_input))
            final_output = self.output_layer(hidden_output)

            # Count spikes over time
            spike_count += final_output > 0

        # Return spike rate as output
        return spike_count / self.num_timesteps

class EnergyEfficientPolicy(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.snn = SpikingNeuralNetwork(state_dim, 128, action_dim)
        self.energy_budget = 1.0  # Energy constraint

    def forward(self, state):
        action = self.snn(state)
        return torch.tanh(action)  # Bound actions
```

### 11.2.2 Model Compression and Quantization

```python
class QuantizedNeuralNetwork(nn.Module):
    def __init__(self, original_model, bit_width=8):
        super().__init__()
        self.original_model = original_model
        self.bit_width = bit_width
        self.scale_factor = 2 ** (bit_width - 1) - 1

        # Quantization parameters
        self.register_buffer('input_scale', torch.tensor(1.0))
        self.register_buffer('output_scale', torch.tensor(1.0))

    def quantize_tensor(self, tensor):
        """Quantize tensor to specified bit width"""
        # Clamp values to range [-1, 1]
        clamped = torch.clamp(tensor, -1, 1)
        # Scale to quantization range
        scaled = clamped * self.scale_factor
        # Quantize to integer
        quantized = torch.round(scaled)
        # Scale back
        dequantized = quantized / self.scale_factor
        return dequantized

    def forward(self, x):
        # Quantize input
        x_quantized = self.quantize_tensor(x)

        # Forward through original model with quantized operations
        with torch.no_grad():
            output = self.original_model(x_quantized)

        # Quantize output
        output_quantized = self.quantize_tensor(output)
        return output_quantized

def prune_network(model, sparsity=0.5):
    """Prune network to reduce computational requirements"""
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Apply magnitude-based pruning
            weight = module.weight.data
            threshold = torch.topk(torch.abs(weight.flatten()),
                                 int(sparsity * weight.numel())).values[-1]

            # Create mask
            mask = torch.abs(weight) > threshold
            module.weight.data *= mask.float()

    return model

class EnergyAwareTraining:
    def __init__(self, model, energy_cost_per_flop=1e-12):
        self.model = model
        self.energy_cost_per_flop = energy_cost_per_flop

    def compute_energy_cost(self, state):
        """Estimate energy cost of computation"""
        # Count floating point operations
        with torch.no_grad():
            flops = self.estimate_flops(state)
            energy_cost = flops * self.energy_cost_per_flop
        return energy_cost

    def estimate_flops(self, state):
        """Estimate FLOPs for given input"""
        # Simplified FLOP estimation
        flops = 0
        state_size = state.numel()

        for module in self.model.modules():
            if isinstance(module, nn.Linear):
                flops += module.in_features * module.out_features

        return flops * state_size
```

## 11.3 Efficient Sensing Strategies

### 11.3.1 Adaptive Sensing

```python
class AdaptiveSensorManager:
    def __init__(self, base_sensors, energy_budget=100.0):
        self.base_sensors = base_sensors
        self.energy_budget = energy_budget
        self.current_energy = energy_budget

        # Sensor efficiency models
        self.sensor_efficiency = {
            'camera': {'energy_per_use': 5.0, 'information_gain': 8.0},
            'lidar': {'energy_per_use': 8.0, 'information_gain': 10.0},
            'imu': {'energy_per_use': 1.0, 'information_gain': 3.0},
            'gps': {'energy_per_use': 2.0, 'information_gain': 4.0}
        }

    def select_optimal_sensors(self, task_requirements, state_uncertainty):
        """Select sensors based on energy efficiency and task needs"""
        available_sensors = []

        for sensor_name, specs in self.sensor_efficiency.items():
            if specs['energy_per_use'] <= self.current_energy:
                # Calculate efficiency ratio
                efficiency = specs['information_gain'] / specs['energy_per_use']

                # Adjust based on task requirements and uncertainty
                adjusted_efficiency = efficiency * self.uncertainty_weight(
                    state_uncertainty, sensor_name
                )

                available_sensors.append((sensor_name, adjusted_efficiency))

        # Sort by efficiency and select top sensors within budget
        available_sensors.sort(key=lambda x: x[1], reverse=True)

        selected_sensors = []
        energy_spent = 0

        for sensor_name, efficiency in available_sensors:
            sensor_cost = self.sensor_efficiency[sensor_name]['energy_per_use']
            if energy_spent + sensor_cost <= self.energy_budget:
                selected_sensors.append(sensor_name)
                energy_spent += sensor_cost

        return selected_sensors

    def uncertainty_weight(self, uncertainty, sensor_type):
        """Weight sensor selection based on current uncertainty"""
        # Higher uncertainty increases need for precise sensing
        base_weight = 1.0
        uncertainty_factor = min(2.0, 1.0 + uncertainty)
        return base_weight * uncertainty_factor

class EventBasedSensor(nn.Module):
    def __init__(self, sensor_threshold=0.1):
        super().__init__()
        self.sensor_threshold = sensor_threshold
        self.last_reading = None

    def forward(self, current_reading):
        """Only update when change exceeds threshold"""
        if self.last_reading is None:
            self.last_reading = current_reading
            return current_reading, True  # Always trigger first reading

        # Calculate change magnitude
        change = torch.norm(current_reading - self.last_reading)

        if change > self.sensor_threshold:
            self.last_reading = current_reading
            return current_reading, True  # Trigger update
        else:
            return self.last_reading, False  # No update needed
```

## 11.4 Energy-Efficient Motion Planning

### 11.4.1 Optimal Control for Energy Minimization

```python
class EnergyOptimalController:
    def __init__(self, system_dynamics, energy_cost_matrix=None):
        self.system_dynamics = system_dynamics
        if energy_cost_matrix is None:
            self.energy_cost_matrix = np.eye(2)  # Default: equal energy cost
        else:
            self.energy_cost_matrix = energy_cost_matrix

    def compute_energy_optimal_trajectory(self, start_state, goal_state,
                                        time_horizon=10, dt=0.1):
        """Compute energy-optimal trajectory using direct collocation"""
        import cvxpy as cp

        n_states = len(start_state)
        n_controls = 2  # Assuming 2D control inputs

        # Decision variables
        X = cp.Variable((n_states, time_horizon + 1))
        U = cp.Variable((n_controls, time_horizon))

        # Objective: minimize energy consumption
        energy_cost = 0
        for k in range(time_horizon):
            energy_cost += cp.quad_form(U[:, k], self.energy_cost_matrix)

        # Constraints

        # Initial and final conditions
        constraints = []
        constraints.append(X[:, 0] == start_state)
        constraints.append(X[:, -1] == goal_state)

        # Dynamics constraints
        for k in range(time_horizon):
            # Simple linearized dynamics: x_{k+1} = Ax_k + Bu_k
            next_state = self.system_dynamics(X[:, k], U[:, k], dt)
            constraints.append(X[:, k + 1] == next_state)

        # Control limits
        for k in range(time_horizon):
            constraints.append(cp.norm(U[:, k], 'inf') <= 1.0)

        # Solve optimization
        problem = cp.Problem(cp.Minimize(energy_cost), constraints)
        problem.solve()

        if problem.status == cp.OPTIMAL:
            return X.value, U.value
        else:
            # Fallback to simple energy-efficient path
            return self.fallback_trajectory(start_state, goal_state, time_horizon)

    def fallback_trajectory(self, start, goal, steps):
        """Simple linear interpolation as fallback"""
        trajectory = np.zeros((len(start), steps + 1))
        controls = np.zeros((2, steps))

        for i in range(steps + 1):
            t = i / steps
            trajectory[:, i] = start * (1 - t) + goal * t

        for i in range(steps):
            controls[:, i] = (trajectory[:, i + 1] - trajectory[:, i]) / 0.1

        return trajectory, controls

class MinimumJerkTrajectory:
    def __init__(self, start_pos, end_pos, duration=2.0):
        self.start_pos = np.array(start_pos)
        self.end_pos = np.array(end_pos)
        self.duration = duration
        self.coeffs = self.compute_minimum_jerk_coefficients()

    def compute_minimum_jerk_coefficients(self):
        """Compute coefficients for minimum jerk trajectory"""
        # Minimum jerk trajectory: s(t) = a0 + a1*t + a2*t^2 + a3*t^3 + a4*t^4 + a5*t^5
        # Boundary conditions: s(0)=0, s(T)=1, s'(0)=s'(T)=s''(0)=s''(T)=0
        T = self.duration

        a0 = 0
        a1 = 0
        a2 = 0
        a3 = 10 / (T**3)
        a4 = -15 / (T**4)
        a5 = 6 / (T**5)

        return [a0, a1, a2, a3, a4, a5]

    def position(self, t):
        """Get position at time t"""
        if t < 0:
            return self.start_pos
        elif t > self.duration:
            return self.end_pos
        else:
            s = sum(c * (t**i) for i, c in enumerate(self.coeffs))
            return self.start_pos + s * (self.end_pos - self.start_pos)

    def velocity(self, t):
        """Get velocity at time t"""
        if t < 0 or t > self.duration:
            return np.zeros_like(self.start_pos)
        else:
            # Derivative of minimum jerk: v(t) = a1 + 2*a2*t + 3*a3*t^2 + 4*a4*t^3 + 5*a5*t^4
            T = t
            v = (self.coeffs[1] +
                 2*self.coeffs[2]*T +
                 3*self.coeffs[3]*T**2 +
                 4*self.coeffs[4]*T**3 +
                 5*self.coeffs[5]*T**4)
            return v * (self.end_pos - self.start_pos)

    def energy_consumption(self):
        """Estimate energy consumption of trajectory"""
        # Integrate squared velocity over time (proportional to energy)
        dt = 0.01
        t = 0
        energy = 0

        while t <= self.duration:
            vel = self.velocity(t)
            energy += np.sum(vel**2) * dt
            t += dt

        return energy
```

## 11.5 Sustainable Learning Algorithms

### 11.5.1 Green Reinforcement Learning

```python
class GreenRLAgent:
    def __init__(self, state_dim, action_dim, energy_budget=1000.0):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.energy_budget = energy_budget
        self.current_energy = energy_budget

        # Main policy network
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        # Energy prediction network
        self.energy_predictor = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # Predict energy cost
        )

        self.optimizer = torch.optim.Adam(
            list(self.policy_network.parameters()) +
            list(self.energy_predictor.parameters()),
            lr=1e-4
        )

    def select_action(self, state, energy_conscious=True):
        """Select action considering energy constraints"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        if energy_conscious and self.current_energy < 0.1 * self.energy_budget:
            # Enter energy-saving mode
            return self.energy_saving_action(state)

        action = self.policy_network(state_tensor).squeeze(0).numpy()

        if energy_conscious:
            # Predict energy cost and adjust if necessary
            action_tensor = torch.FloatTensor(action).unsqueeze(0)
            predicted_energy = self.energy_predictor(
                torch.cat([state_tensor, action_tensor], dim=1)
            ).item()

            if self.current_energy - predicted_energy < 0:
                # Choose less energy-intensive action
                action = self.find_low_energy_action(state)

        return action

    def energy_saving_action(self, state):
        """Return energy-conserving action"""
        # Return minimal action (e.g., stop, hover)
        return np.zeros(self.action_dim)

    def find_low_energy_action(self, state):
        """Find action that minimizes energy consumption"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Optimize for low energy action
        best_action = np.zeros(self.action_dim)
        min_energy = float('inf')

        # Sample actions to find low-energy option
        for _ in range(100):
            candidate_action = np.random.uniform(-1, 1, self.action_dim)
            action_tensor = torch.FloatTensor(candidate_action).unsqueeze(0)
            energy_cost = self.energy_predictor(
                torch.cat([state_tensor, action_tensor], dim=1)
            ).item()

            if energy_cost < min_energy:
                min_energy = energy_cost
                best_action = candidate_action

        return best_action

    def update(self, state, action, reward, next_state, done):
        """Update with energy-aware learning"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        # Energy prediction loss
        predicted_energy = self.energy_predictor(
            torch.cat([state_tensor, action_tensor], dim=1)
        )

        # Use actual energy consumption as target (simplified)
        actual_energy = np.sum(np.abs(action))  # Placeholder energy model
        energy_loss = nn.MSELoss()(predicted_energy,
                                  torch.FloatTensor([[actual_energy]]))

        # Policy loss with energy penalty
        policy_output = self.policy_network(state_tensor)
        energy_penalty = self.energy_predictor(
            torch.cat([state_tensor, policy_output], dim=1)
        )

        # Combined loss: maximize reward, minimize energy
        combined_loss = -reward + 0.1 * energy_penalty.mean()

        total_loss = combined_loss + energy_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
```

## 11.6 Energy Harvesting and Management

### 11.6.1 Energy Harvesting Integration

```python
class EnergyHarvestingSystem:
    def __init__(self, initial_energy=100.0, max_energy=1000.0):
        self.current_energy = initial_energy
        self.max_energy = max_energy
        self.energy_sources = {
            'solar': {'rate': 0.1, 'availability': 0.5},  # 0.1 units/sec, 50% available
            'kinetic': {'rate': 0.05, 'availability': 0.8},  # 0.05 units/sec, 80% available
            'thermal': {'rate': 0.02, 'availability': 0.3}   # 0.02 units/sec, 30% available
        }

    def harvest_energy(self, dt=1.0):
        """Harvest energy from available sources"""
        harvested_energy = 0

        for source, specs in self.energy_sources.items():
            if np.random.random() < specs['availability']:
                harvested = specs['rate'] * dt
                harvested_energy += harvested

        # Update energy level
        self.current_energy = min(self.max_energy,
                                self.current_energy + harvested_energy)

        return harvested_energy

    def consume_energy(self, amount):
        """Consume energy, return whether sufficient available"""
        if self.current_energy >= amount:
            self.current_energy -= amount
            return True
        else:
            return False

    def get_energy_level(self):
        """Get normalized energy level [0, 1]"""
        return self.current_energy / self.max_energy

class AdaptivePolicyScheduler:
    def __init__(self, energy_system, policy_complexity_levels):
        self.energy_system = energy_system
        self.policy_complexity_levels = policy_complexity_levels  # [simple, medium, complex]
        self.current_level = 1  # Start with medium complexity

    def select_policy_level(self):
        """Select policy complexity based on energy level"""
        energy_level = self.energy_system.get_energy_level()

        if energy_level > 0.7:
            # High energy: use complex policy
            self.current_level = 2
        elif energy_level > 0.3:
            # Medium energy: use medium policy
            self.current_level = 1
        else:
            # Low energy: use simple policy
            self.current_level = 0

        return self.current_level

    def get_energy_consumption(self, policy_level):
        """Get energy consumption for different policy levels"""
        consumption_rates = [1.0, 2.5, 5.0]  # Simple, medium, complex
        return consumption_rates[policy_level]
```

## 11.7 Lifecycle Assessment and Environmental Impact

### 11.7.1 Carbon Footprint Estimation

```python
class CarbonFootprintEstimator:
    def __init__(self):
        # Carbon intensity factors (kg CO2/kWh)
        self.carbon_factors = {
            'coal': 0.82,
            'gas': 0.37,
            'renewable': 0.05,
            'average': 0.475
        }
        self.current_grid_mix = 'average'

    def estimate_computational_footprint(self, energy_consumption_kwh):
        """Estimate carbon footprint of computational energy"""
        carbon_factor = self.carbon_factors[self.current_grid_mix]
        return energy_consumption_kwh * carbon_factor

    def estimate_hardware_footprint(self, device_mass_kg, material_type='silicon'):
        """Estimate carbon footprint of hardware manufacturing"""
        # Approximate manufacturing carbon intensity (kg CO2/kg)
        material_factors = {
            'silicon': 10.0,  # High for semiconductor processing
            'aluminum': 15.0,
            'steel': 2.0,
            'plastic': 3.0
        }
        factor = material_factors.get(material_type, 10.0)
        return device_mass_kg * factor

    def total_footprint(self, energy_kwh, device_mass_kg, device_material='silicon'):
        """Calculate total carbon footprint"""
        computational = self.estimate_computational_footprint(energy_kwh)
        manufacturing = self.estimate_hardware_footprint(device_mass_kg, device_material)
        return computational + manufacturing

class SustainableDesignOptimizer:
    def __init__(self, carbon_budget=1000.0):  # kg CO2
        self.carbon_budget = carbon_budget
        self.carbon_estimator = CarbonFootprintEstimator()

    def optimize_design(self, design_parameters):
        """Optimize design parameters for minimal carbon footprint"""
        # This would typically involve complex optimization
        # For simplicity, we'll implement a basic trade-off analysis

        best_design = None
        min_footprint = float('inf')

        # Evaluate different design configurations
        for config in self.generate_design_space(design_parameters):
            footprint = self.evaluate_design_footprint(config)

            if footprint < min_footprint and footprint <= self.carbon_budget:
                min_footprint = footprint
                best_design = config

        return best_design, min_footprint

    def generate_design_space(self, base_params):
        """Generate design space for optimization"""
        configs = []

        # Vary computational complexity
        for comp_level in [0.5, 1.0, 1.5]:  # Relative complexity
            # Vary hardware efficiency
            for hw_efficiency in [0.8, 1.0, 1.2]:  # Relative efficiency
                configs.append({
                    'computational_complexity': comp_level,
                    'hardware_efficiency': hw_efficiency
                })

        return configs

    def evaluate_design_footprint(self, config):
        """Evaluate carbon footprint of design configuration"""
        # Simplified model
        base_energy = 100.0  # kWh
        energy_adjusted = base_energy * config['computational_complexity']

        base_material = 5.0  # kg
        material_adjusted = base_material / config['hardware_efficiency']

        return self.carbon_estimator.total_footprint(
            energy_adjusted, material_adjusted
        )
```

## Key Takeaways

- Energy efficiency requires optimization across computation, sensing, actuation, and communication
- Neuromorphic and spiking neural networks offer significant energy savings for certain tasks
- Adaptive sensing strategies reduce energy consumption by activating sensors only when needed
- Energy-optimal control minimizes energy consumption while achieving task objectives
- Green RL algorithms incorporate energy constraints into learning objectives
- Energy harvesting can extend operational lifetime in resource-constrained environments
- Lifecycle assessment is crucial for understanding total environmental impact

## Exercises

1. **Coding**: Implement a spiking neural network controller and compare its energy consumption with a traditional neural network.

2. **Theoretical**: Derive the minimum jerk trajectory equations and prove their optimality for smooth motion.

3. **Coding**: Design an energy-aware motion planner that optimizes for both time and energy consumption.

4. **Theoretical**: Analyze the trade-offs between model compression and control performance in Physical AI systems.

5. **Coding**: Implement a carbon footprint estimator for a Physical AI system and optimize its design parameters.

## Further Reading

1. Merolla, P. A., et al. (2014). "A million spiking-neuron integrated circuit with a scalable communication network and interface." *Science*.

2. Strukov, D. B., et al. (2008). "The missing memristor found." *Nature*.

3. Zhang, Y., et al. (2020). "Energy-efficient neural networks: A survey." *Journal of Low Power Electronics and Applications*.