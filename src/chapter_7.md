# Chapter 7: Sim-to-Real Transfer

Sim-to-real transfer is a critical technique in Physical AI that enables policies learned in simulation to be successfully deployed on real-world robots. This approach addresses the fundamental challenge of sample efficiency by leveraging the speed and safety of simulation while attempting to bridge the reality gap between synthetic and physical environments.

## 7.1 The Reality Gap Problem

The reality gap refers to the discrepancy between simulated and real environments that prevents policies trained in simulation from performing well in the physical world. This gap arises from several sources:

- **Dynamics Mismatch**: Simulated physics often fail to capture real-world complexities like friction, compliance, and contact dynamics
- **Visual Differences**: Lighting, textures, and sensor noise differ between simulation and reality
- **Actuator Limitations**: Real actuators have delays, saturation, and non-linear responses
- **Unmodeled Effects**: Factors like air resistance, temperature changes, and wear are often omitted

The transfer problem can be formulated as finding a policy $\pi_{real}$ that performs well in the real environment $M_{real}$ using knowledge from a simulated environment $M_{sim}$:

$$\pi_{real} = \arg\max_{\pi} \mathbb{E}_{\tau \sim M_{real}, \pi} \left[ \sum_{t=0}^T r_t \right]$$

subject to having access only to samples from $M_{sim}$.

![Figure 7.1: The Sim-to-Real Transfer Problem](placeholder)

## 7.2 Domain Randomization

Domain randomization is one of the most successful approaches to sim-to-real transfer. Instead of creating an accurate simulation, this technique randomizes simulation parameters to train robust policies that can handle variations:

```python
import numpy as np
import torch
import torch.nn as nn

class RandomizedDynamicsModel:
    def __init__(self, base_params):
        self.base_params = base_params
        self.randomization_bounds = {
            'mass': [0.5, 1.5],      # 50% to 150% of base mass
            'friction': [0.01, 0.1], # Random friction range
            'gravity': [8.5, 9.8],   # Gravity variations
            'com_offset': [0.0, 0.05] # Center of mass offset
        }

    def randomize_params(self):
        """Generate randomized parameters for simulation"""
        randomized_params = {}
        for param, bounds in self.randomization_bounds.items():
            if param == 'com_offset':
                # 3D offset
                randomized_params[param] = np.random.uniform(
                    -bounds[1], bounds[1], size=3
                )
            else:
                randomized_params[param] = np.random.uniform(
                    bounds[0], bounds[1]
                )
        return randomized_params

class DomainRandomizedEnvironment:
    def __init__(self, base_env):
        self.base_env = base_env
        self.dynamics_model = RandomizedDynamicsModel(base_env.get_base_params())
        self.reset_dynamics()

    def reset_dynamics(self):
        """Randomize dynamics parameters"""
        self.current_params = self.dynamics_model.randomize_params()
        self.base_env.update_params(self.current_params)

    def step(self, action):
        """Execute action in current randomized environment"""
        return self.base_env.step(action)

    def reset(self):
        """Reset environment with new random dynamics"""
        self.reset_dynamics()
        return self.base_env.reset()

# Training with domain randomization
def train_with_domain_randomization(agent, env, episodes=10000):
    """Train agent with domain randomization"""
    for episode in range(episodes):
        state = env.reset()  # New random dynamics each episode
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if episode % 1000 == 0:
            print(f"Episode {episode}, Reward: {total_reward:.2f}")
```

## 7.3 Domain Adaptation Techniques

Domain adaptation methods learn to map between simulation and reality using techniques from transfer learning:

### 7.3.1 Adversarial Domain Adaptation

```python
class DomainAdaptationNetwork(nn.Module):
    def __init__(self, state_dim, hidden_dim=256):
        super().__init__()
        # Feature extractor
        self.feature_extractor = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # Task-specific classifier
        self.task_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)  # Action prediction
        )

        # Domain classifier
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 2)  # sim vs real
        )

    def forward(self, x, domain_label=None):
        features = self.feature_extractor(x)
        task_output = self.task_classifier(features)

        if domain_label is not None:
            domain_output = self.domain_classifier(features)
            return task_output, domain_output
        else:
            return task_output

def adversarial_training(sim_data, real_data, model, epochs=100):
    """Train with adversarial domain adaptation"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    task_criterion = nn.MSELoss()
    domain_criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        # Train on simulation data
        sim_states, sim_actions = sim_data
        sim_domain_labels = torch.zeros(len(sim_states))  # Label 0 for sim

        # Train on real data
        real_states, real_actions = real_data
        real_domain_labels = torch.ones(len(real_states))  # Label 1 for real

        # Combine data
        all_states = torch.cat([sim_states, real_states])
        all_actions = torch.cat([sim_actions, real_actions])
        all_domains = torch.cat([sim_domain_labels, real_domain_labels]).long()

        task_pred, domain_pred = model(all_states, domain_labels=all_domains)

        # Task loss (want to predict actions well)
        task_loss = task_criterion(task_pred, all_actions)

        # Domain loss (want to confuse domain classifier)
        domain_loss = domain_criterion(domain_pred, all_domains)

        # Total loss (minimize task loss, maximize domain confusion)
        total_loss = task_loss - domain_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        if epoch % 20 == 0:
            print(f"Epoch {epoch}, Task Loss: {task_loss:.4f}, Domain Loss: {domain_loss:.4f}")
```

### 7.3.2 System Identification

System identification techniques learn the mapping between simulation and reality:

```python
class SystemIdentifier:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.correction_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_dim)  # Correction term
        )
        self.optimizer = torch.optim.Adam(self.correction_model.parameters(), lr=1e-3)

    def collect_data_pairs(self, sim_env, real_env, policy, num_pairs=1000):
        """Collect state-action-state transitions from both domains"""
        sim_transitions = []
        real_transitions = []

        for _ in range(num_pairs):
            # Reset both environments to same initial state
            sim_state = sim_env.reset()
            real_state = real_env.reset_to_state(sim_state.copy())

            action = policy.select_action(sim_state)

            # Get next states
            sim_next_state, _, _, _ = sim_env.step(action)
            real_next_state, _, _, _ = real_env.step(action)

            sim_transitions.append((sim_state, action, sim_next_state))
            real_transitions.append((real_state, action, real_next_state))

        return sim_transitions, real_transitions

    def train_correction_model(self, sim_transitions, real_transitions):
        """Train model to predict real dynamics from sim dynamics"""
        for epoch in range(1000):
            total_loss = 0

            for (sim_state, action, sim_next), (real_state, _, real_next) in \
                zip(sim_transitions, real_transitions):

                # Predict correction from sim to real
                input_features = torch.cat([
                    torch.FloatTensor(sim_state),
                    torch.FloatTensor(action)
                ])

                predicted_correction = self.correction_model(input_features)
                predicted_real_next = torch.FloatTensor(sim_next) + predicted_correction

                # Loss is difference between predicted and actual real next state
                loss = nn.MSELoss()(predicted_real_next, torch.FloatTensor(real_next))

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            if epoch % 100 == 0:
                print(f"Correction model epoch {epoch}, Loss: {total_loss/len(sim_transitions):.6f}")

    def corrected_sim_step(self, sim_env, action):
        """Step simulation with correction applied"""
        current_state = sim_env.get_state()
        sim_next_state = sim_env.step_no_render(action)

        # Apply correction
        input_features = torch.cat([
            torch.FloatTensor(current_state),
            torch.FloatTensor(action)
        ])

        correction = self.correction_model(input_features).detach().numpy()
        corrected_next_state = sim_next_state + correction

        sim_env.set_state(corrected_next_state)
        return corrected_next_state
```

## 7.4 Sim-to-Real Transfer Methods

### 7.4.1 Progressive Domain Randomization

Progressive domain randomization gradually increases the range of randomization:

```python
class ProgressiveRandomization:
    def __init__(self, base_params, max_bounds):
        self.base_params = base_params
        self.max_bounds = max_bounds
        self.current_progress = 0.0  # 0.0 to 1.0
        self.progress_rate = 0.01

    def update_progress(self, performance_metric):
        """Increase randomization based on performance"""
        if performance_metric > 0.8:  # If agent is doing well
            self.current_progress = min(1.0, self.current_progress + self.progress_rate)

    def get_current_bounds(self):
        """Get current randomization bounds based on progress"""
        current_bounds = {}
        for param, max_bound in self.max_bounds.items():
            if isinstance(max_bound, (list, tuple)):
                # For ranges like [min_val, max_val]
                base_val = self.base_params.get(param, (max_bound[0] + max_bound[1]) / 2)
                range_size = (max_bound[1] - max_bound[0]) * self.current_progress
                current_bounds[param] = [
                    base_val - range_size / 2,
                    base_val + range_size / 2
                ]
            else:
                # For single values
                current_bounds[param] = max_bound * self.current_progress
        return current_bounds

def progressive_training(agent, base_env, epochs=5000):
    """Train with progressive domain randomization"""
    randomizer = ProgressiveRandomization(
        base_env.get_base_params(),
        max_bounds={
            'mass': [0.5, 2.0],
            'friction': [0.0, 0.2],
            'gravity': [8.0, 10.0]
        }
    )

    for epoch in range(epochs):
        # Update randomization based on progress
        if epoch > 0 and epoch % 100 == 0:
            # Evaluate performance and update progress
            performance = evaluate_agent(agent, base_env, num_episodes=10)
            randomizer.update_progress(performance)

        # Create environment with current randomization
        env = DomainRandomizedEnvironment(base_env)
        env.current_params = randomizer.get_current_bounds()

        # Train for one episode
        state = env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Progress: {randomizer.current_progress:.2f}, Reward: {total_reward:.2f}")
```

### 7.4.2 Domain Randomization with GANs

Using Generative Adversarial Networks to learn realistic simulation parameters:

```python
class SimulationGAN(nn.Module):
    def __init__(self, state_dim, action_dim, param_dim):
        super().__init__()
        # Generator: generates realistic simulation parameters
        self.generator = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, param_dim),
            nn.Tanh()  # Output in [-1, 1] range
        )

        # Discriminator: distinguishes real from generated simulation data
        self.discriminator = nn.Sequential(
            nn.Linear(state_dim + action_dim + param_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()  # Probability of being real
        )

    def generate_params(self, state, action):
        """Generate simulation parameters given state and action"""
        x = torch.cat([torch.FloatTensor(state), torch.FloatTensor(action)], dim=-1)
        return self.generator(x)

    def discriminate(self, state, action, params):
        """Discriminate between real and generated parameters"""
        x = torch.cat([
            torch.FloatTensor(state),
            torch.FloatTensor(action),
            torch.FloatTensor(params)
        ], dim=-1)
        return self.discriminator(x)

def train_simulation_gan(gan, real_data_loader, epochs=1000):
    """Train GAN to generate realistic simulation parameters"""
    gen_optimizer = torch.optim.Adam(gan.generator.parameters(), lr=2e-4)
    disc_optimizer = torch.optim.Adam(gan.discriminator.parameters(), lr=2e-4)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        total_gen_loss = 0
        total_disc_loss = 0

        for real_states, real_actions, real_params in real_data_loader:
            batch_size = len(real_states)

            # Train discriminator
            # Real data
            real_labels = torch.ones(batch_size, 1)
            real_output = gan.discriminate(real_states, real_actions, real_params)
            real_loss = criterion(real_output, real_labels)

            # Generated data
            fake_params = gan.generate_params(real_states, real_actions)
            fake_labels = torch.zeros(batch_size, 1)
            fake_output = gan.discriminate(real_states, real_actions, fake_params)
            fake_loss = criterion(fake_output, fake_labels)

            disc_loss = real_loss + fake_loss

            disc_optimizer.zero_grad()
            disc_loss.backward()
            disc_optimizer.step()

            # Train generator
            fake_params = gan.generate_params(real_states, real_actions)
            fake_output = gan.discriminate(real_states, real_actions, fake_params)
            gen_loss = criterion(fake_output, real_labels)  # Want to fool discriminator

            gen_optimizer.zero_grad()
            gen_loss.backward()
            gen_optimizer.step()

            total_gen_loss += gen_loss.item()
            total_disc_loss += disc_loss.item()

        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Gen Loss: {total_gen_loss:.4f}, Disc Loss: {total_disc_loss:.4f}")
```

## 7.5 Systematic Approaches to Sim-to-Real Transfer

### 7.5.1 System Identification Pipeline

A systematic approach to identify and correct simulation discrepancies:

```python
class SystematicSimToReal:
    def __init__(self, sim_env, real_env):
        self.sim_env = sim_env
        self.real_env = real_env
        self.discrepancy_models = {}
        self.correction_functions = {}

    def identify_discrepancies(self, policy, num_rollouts=50):
        """Systematically identify discrepancies between sim and real"""
        discrepancies = {
            'dynamics': [],
            'sensor': [],
            'actuation': []
        }

        for _ in range(num_rollouts):
            # Reset both environments to same state
            sim_state = self.sim_env.reset()
            real_state = self.real_env.reset_to_state(sim_state.copy())

            rollout_discrepancies = []

            for t in range(100):  # Fixed horizon rollout
                action = policy.select_action(sim_state)

                # Get transitions
                sim_next_state, sim_reward, sim_done, _ = self.sim_env.step(action)
                real_next_state, real_reward, real_done, _ = self.real_env.step(action)

                # Calculate discrepancies
                state_diff = sim_next_state - real_next_state
                reward_diff = sim_reward - real_reward

                discrepancies['dynamics'].append(state_diff)
                discrepancies['sensor'].append(reward_diff)

                # Store for model training
                rollout_discrepancies.append({
                    'state': sim_state,
                    'action': action,
                    'state_diff': state_diff
                })

                if sim_done or real_done:
                    break

                sim_state = sim_next_state

            # Train discrepancy prediction models
            self.train_discrepancy_models(rollout_discrepancies)

        return discrepancies

    def train_discrepancy_models(self, discrepancy_data):
        """Train models to predict discrepancies"""
        states = torch.FloatTensor([d['state'] for d in discrepancy_data])
        actions = torch.FloatTensor([d['action'] for d in discrepancy_data])
        state_diffs = torch.FloatTensor([d['state_diff'] for d in discrepancy_data])

        # Create discrepancy prediction model
        discrepancy_model = nn.Sequential(
            nn.Linear(states.shape[1] + actions.shape[1], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, state_diffs.shape[1])
        )

        optimizer = torch.optim.Adam(discrepancy_model.parameters(), lr=1e-3)

        for epoch in range(500):
            pred_diffs = discrepancy_model(torch.cat([states, actions], dim=1))
            loss = nn.MSELoss()(pred_diffs, state_diffs)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        self.discrepancy_models['dynamics'] = discrepancy_model

    def apply_correction(self, sim_state, action):
        """Apply learned correction to simulation"""
        if 'dynamics' in self.discrepancy_models:
            input_features = torch.cat([
                torch.FloatTensor(sim_state),
                torch.FloatTensor(action)
            ]).unsqueeze(0)

            predicted_diff = self.discrepancy_models['dynamics'](input_features).squeeze(0).detach().numpy()
            corrected_state = sim_state - predicted_diff  # Subtract the predicted discrepancy
            return corrected_state
        else:
            return sim_state

def systematic_transfer_training(agent, systematic_transfer, epochs=2000):
    """Train with systematic sim-to-real approach"""
    for epoch in range(epochs):
        if epoch % 500 == 0:  # Re-identify discrepancies periodically
            print(f"Re-identifying discrepancies at epoch {epoch}")
            systematic_transfer.identify_discrepancies(agent)

        # Use corrected simulation for training
        sim_env = systematic_transfer.sim_env
        state = sim_env.reset()
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)

            # Apply correction if available
            next_state = sim_env.step(action)
            if epoch > 100:  # Start applying corrections after initial training
                corrected_state = systematic_transfer.apply_correction(state, action)
                sim_env.set_state(corrected_state)

            # Continue with corrected state
            agent.update(state, action, 0, next_state, done)  # Using dummy reward
            state = next_state
            total_reward += 0  # Using dummy reward during sim training

        if epoch % 500 == 0:
            print(f"Epoch {epoch}, Total Reward: {total_reward:.2f}")
```

## 7.6 Practical Considerations

### 7.6.1 Evaluation Metrics for Transfer

```python
def evaluate_sim_to_real_transfer(sim_agent, real_env, num_episodes=20):
    """Evaluate how well sim-trained agent performs in real"""
    real_rewards = []
    success_rates = []

    for episode in range(num_episodes):
        state = real_env.reset()
        total_reward = 0
        steps = 0
        max_steps = 1000

        while steps < max_steps:
            action = sim_agent.select_action(state)
            next_state, reward, done, info = real_env.step(action)

            total_reward += reward
            state = next_state
            steps += 1

            if done:
                break

        real_rewards.append(total_reward)
        success_rates.append(info.get('success', 0))

    return {
        'mean_reward': np.mean(real_rewards),
        'std_reward': np.std(real_rewards),
        'success_rate': np.mean(success_rates),
        'transfer_efficiency': np.mean(real_rewards) / max(1, np.mean(real_rewards))  # Normalized
    }

def compute_domain_gap(sim_performance, real_performance):
    """Compute domain gap metric"""
    return abs(sim_performance - real_performance) / max(abs(sim_performance), 1)
```

## Key Takeaways

- The reality gap between simulation and reality is the primary challenge in sim-to-real transfer
- Domain randomization is highly effective by training policies on diverse simulation parameters
- Systematic approaches involve identifying and correcting specific discrepancies between domains
- Progressive domain randomization gradually increases randomization based on agent performance
- Adversarial techniques can learn realistic simulation parameters from real data
- Evaluation of transfer success requires careful metrics comparing sim and real performance

## Exercises

1. **Coding**: Implement domain randomization on a simple pendulum swing-up task and measure the transfer gap as you increase the randomization range.

2. **Theoretical**: Prove that domain randomization can be viewed as training a robust policy that is invariant to parameter variations.

3. **Coding**: Design a system identification pipeline for a 2D point-mass navigation task and evaluate the improvement in sim-to-real transfer.

4. **Theoretical**: Explain why progressive domain randomization might be more effective than uniform randomization across the full range.

5. **Coding**: Implement an adversarial domain adaptation approach for a robotic grasping simulation and evaluate its effectiveness on a real robot.

## Further Reading

1. Tobin, J., et al. (2017). "Domain randomization for transferring deep neural networks from simulation to the real world." *IROS*.

2. Sadeghi, F., & Levine, S. (2017). "CAD2RL: Real single-image flight without a single real image." *RSS*.

3. Peng, X. B., et al. (2018). "Sim-to-real transfer of robotic control with dynamics randomization." *ICRA*.