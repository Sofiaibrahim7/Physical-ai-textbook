# Chapter 6: Reinforcement Learning in Physical Domains

Reinforcement Learning (RL) has emerged as a powerful paradigm for learning control policies in physical systems. Unlike traditional control methods that rely on precise mathematical models, RL enables agents to learn optimal behaviors through trial and error, making it particularly suitable for complex physical environments where analytical solutions are difficult or impossible to obtain.

## 6.1 Introduction to RL in Physical Systems

In physical domains, reinforcement learning faces unique challenges compared to classical discrete environments. The continuous nature of physical spaces, the presence of noise and uncertainty, safety constraints, and the cost of real-world trials all influence the design and implementation of RL algorithms for physical AI systems.

The core RL framework consists of an agent interacting with an environment through actions $a_t$, receiving observations $o_t$ and rewards $r_t$. The goal is to learn a policy $\pi(a|s)$ that maximizes the expected cumulative reward:

$$J(\pi) = \mathbb{E}_{\tau \sim \pi}\left[\sum_{t=0}^{T} \gamma^t r(s_t, a_t)\right]$$

where $\tau = (s_0, a_0, r_0, s_1, a_1, r_1, ...)$ represents a trajectory, and $\gamma \in [0,1]$ is the discount factor.

![Figure 6.1: Reinforcement Learning Loop in Physical Systems](placeholder)

## 6.2 Continuous Action Spaces and Policy Optimization

Physical systems typically operate in continuous action spaces, requiring specialized RL algorithms that can handle continuous control. Traditional Q-learning methods struggle with continuous actions due to the infinite action space. Instead, policy gradient methods and actor-critic algorithms are preferred.

### 6.2.1 Policy Gradient Methods

Policy gradient methods directly optimize the policy parameters $\theta$ by following the gradient of the expected return:

$$\nabla_\theta J(\pi_\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[\nabla_\theta \log \pi_\theta(a|s) Q^{\pi_\theta}(s,a)]$$

The REINFORCE algorithm implements this approach:

```python
import torch
import torch.nn as nn
import numpy as np

class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Output actions in [-1, 1] range
        )

    def forward(self, state):
        return self.network(state)

def compute_returns(rewards, gamma=0.99):
    """Compute discounted returns"""
    returns = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns)

def reinforce_update(policy, optimizer, states, actions, returns):
    """REINFORCE update step"""
    states = torch.stack(states)
    actions = torch.stack(actions)
    returns = torch.stack(returns)

    # Normalize returns for stability
    returns = (returns - returns.mean()) / (returns.std() + 1e-8)

    # Compute log probabilities
    mu = policy(states)
    log_probs = -((actions - mu) ** 2).sum(dim=1) / 2

    # Policy gradient loss
    loss = -(log_probs * returns).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

### 6.2.2 Actor-Critic Methods

Actor-critic methods combine policy-based and value-based approaches, using a critic to estimate the value function and guide the policy improvement. Deep Deterministic Policy Gradient (DDPG) and Twin Delayed DDPG (TD3) are popular for continuous control.

```python
import torch.optim as optim

class CriticNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(CriticNetwork, self).__init__()
        self.q_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.q_network(x)

class DDPGAgent:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3):
        # Actor networks (main and target)
        self.actor = PolicyNetwork(state_dim, action_dim)
        self.actor_target = PolicyNetwork(state_dim, action_dim)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic networks (main and target)
        self.critic = CriticNetwork(state_dim, action_dim)
        self.critic_target = CriticNetwork(state_dim, action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr_critic)

        # Initialize target networks
        self.hard_update(self.actor_target, self.actor)
        self.hard_update(self.critic_target, self.critic)

        self.tau = 0.005  # Target network update rate

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) +
                                   param.data * self.tau)

    def select_action(self, state, noise_scale=0.1):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).cpu().data.numpy()[0]

        # Add noise for exploration
        noise = np.random.normal(0, noise_scale, size=action.shape)
        action = np.clip(action + noise, -1, 1)
        return action

    def update(self, states, actions, rewards, next_states, dones, gamma=0.99):
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.BoolTensor(dones).unsqueeze(1)

        # Compute target Q-values
        next_actions = self.actor_target(next_states)
        next_q_values = self.critic_target(next_states, next_actions)
        target_q_values = rewards + (gamma * next_q_values * ~dones)

        # Critic loss
        current_q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values.detach())

        # Actor loss
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        # Update networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        self.soft_update(self.critic_target, self.critic)
        self.soft_update(self.actor_target, self.actor)

        return actor_loss.item(), critic_loss.item()
```

## 6.3 Exploration Strategies in Physical Environments

Exploration in physical systems is challenging due to safety constraints and the cost of real-world trials. Several strategies address these challenges:

### 6.3.1 Noise-Based Exploration

Adding noise to actions is a common exploration strategy. However, in physical systems, this noise should be structured to respect safety constraints:

```python
class OrnsteinUhlenbeckNoise:
    def __init__(self, action_dim, theta=0.15, sigma=0.2, dt=1e-2):
        self.action_dim = action_dim
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.reset()

    def reset(self):
        self.noise_state = np.zeros(self.action_dim)

    def __call__(self):
        x = self.noise_state
        dx = self.theta * (-x) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.randn(self.action_dim)
        self.noise_state = x + dx
        return self.noise_state

# Usage in DDPG agent
ou_noise = OrnsteinUhlenbeckNoise(action_dim=4)  # For 4-DOF manipulator

def add_exploration_noise(action, noise_process, epsilon=0.3):
    """Add exploration noise to action"""
    noise = noise_process()
    noisy_action = action + epsilon * noise
    return np.clip(noisy_action, -1, 1)
```

### 6.3.2 Curiosity-Driven Exploration

Curiosity-driven methods encourage exploration of novel states by rewarding prediction errors:

```python
class IntrinsicReward(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(IntrinsicReward, self).__init__()
        # Forward dynamics model: predict next state given current state and action
        self.forward_model = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim)
        )

        # Inverse dynamics model: predict action given state transition
        self.inverse_model = nn.Sequential(
            nn.Linear(2 * state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state, action, next_state):
        # Predict next state using forward model
        pred_next_state = self.forward_model(torch.cat([state, action], dim=1))

        # Calculate intrinsic reward as prediction error
        intrinsic_reward = ((next_state - pred_next_state) ** 2).mean(dim=1)

        return intrinsic_reward

class CuriosityDrivenAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        self.intrinsic_reward_module = IntrinsicReward(state_dim, action_dim)
        self.intrinsic_optimizer = optim.Adam(self.intrinsic_reward_module.parameters(), lr=1e-3)

    def compute_total_reward(self, extrinsic_reward, state, action, next_state, beta=0.1):
        """Combine extrinsic and intrinsic rewards"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action_tensor = torch.FloatTensor(action).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        intrinsic_reward = self.intrinsic_reward_module(
            state_tensor, action_tensor, next_state_tensor
        ).item()

        total_reward = extrinsic_reward + beta * intrinsic_reward
        return total_reward
```

## 6.4 Safe RL in Physical Systems

Safety is paramount in physical AI systems. Safe RL methods ensure that agents operate within predefined safety constraints during learning:

### 6.4.1 Constrained Markov Decision Processes

Safe RL can be formulated as a constrained optimization problem:

$$\max_\pi \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t r(s_t, a_t)\right]$$
$$\text{subject to: } \mathbb{E}\left[\sum_{t=0}^{\infty} \gamma^t c_i(s_t, a_t)\right] \leq d_i, \forall i$$

where $c_i$ represents the $i$-th constraint cost and $d_i$ is the threshold.

```python
class ConstrainedDDPGAgent(DDPGAgent):
    def __init__(self, state_dim, action_dim, constraint_dims):
        super().__init__(state_dim, action_dim)
        self.constraint_dims = constraint_dims
        self.constraint_thresholds = [0.5] * constraint_dims  # Default thresholds

        # Constraint critic networks
        self.constraint_critics = nn.ModuleList([
            CriticNetwork(state_dim, action_dim) for _ in range(constraint_dims)
        ])
        self.constraint_targets = nn.ModuleList([
            CriticNetwork(state_dim, action_dim) for _ in range(constraint_dims)
        ])

        # Initialize target networks
        for target, critic in zip(self.constraint_targets, self.constraint_critics):
            self.hard_update(target, critic)

    def compute_safe_action(self, state, safety_weight=1.0):
        """Compute action considering safety constraints"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Get action from policy
        action = self.actor(state_tensor)

        # Evaluate constraint violations
        constraint_violations = []
        for i, critic in enumerate(self.constraint_critics):
            q_constraint = critic(state_tensor, action).item()
            violation = max(0, q_constraint - self.constraint_thresholds[i])
            constraint_violations.append(violation)

        # Adjust action based on constraint violations
        if sum(constraint_violations) > 0:
            # Project action to safe region (simplified approach)
            action = self.project_to_safe_region(action, constraint_violations)

        return action.squeeze(0).detach().numpy()

    def project_to_safe_region(self, action, violations):
        """Simple projection to safe region"""
        # Scale action magnitude based on violation severity
        violation_penalty = sum(violations) / len(violations) if violations else 0
        scaling_factor = max(0.1, 1.0 - violation_penalty)  # At least 10% of original action
        return torch.tanh(action) * scaling_factor
```

## 6.5 Sample-Efficient RL Methods

Physical systems often have limited opportunities for trial-and-error learning. Sample-efficient methods are crucial:

### 6.5.1 Model-Based RL

Model-based approaches learn a model of the environment dynamics and use it for planning:

```python
class DynamicsModel(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(DynamicsModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
            nn.Softplus()  # Ensure positive differences
        )
        self.state_dim = state_dim

    def forward(self, state, action):
        """Predict state difference: s_{t+1} - s_t"""
        combined = torch.cat([state, action], dim=1)
        delta_state = self.network(combined)
        return delta_state

class ModelBasedAgent:
    def __init__(self, state_dim, action_dim):
        self.dynamics_model = DynamicsModel(state_dim, action_dim)
        self.policy = PolicyNetwork(state_dim, action_dim)
        self.model_optimizer = optim.Adam(self.dynamics_model.parameters(), lr=1e-3)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=1e-4)

    def train_dynamics_model(self, states, actions, next_states):
        """Train dynamics model to predict state transitions"""
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        next_states = torch.FloatTensor(next_states)

        # Predict state differences
        delta_states_pred = self.dynamics_model(states, actions)
        delta_states_true = next_states - states

        # Model loss
        model_loss = nn.MSELoss()(delta_states_pred, delta_states_true)

        self.model_optimizer.zero_grad()
        model_loss.backward()
        self.model_optimizer.step()

        return model_loss.item()

    def plan_with_model(self, current_state, horizon=10):
        """Plan using the learned dynamics model"""
        state = torch.FloatTensor(current_state).unsqueeze(0)
        total_reward = 0

        for _ in range(horizon):
            # Select action using current policy
            action = self.policy(state)

            # Predict next state using dynamics model
            delta_state = self.dynamics_model(state, action)
            next_state = state + delta_state

            # Estimate reward (simplified)
            reward = self.estimate_reward(next_state, action)
            total_reward += reward

            state = next_state

        return total_reward

    def estimate_reward(self, state, action):
        """Estimate reward function"""
        # Placeholder - implement based on specific task
        return torch.sum(state**2)  # Example: penalize large states
```

## 6.6 Practical Considerations

Several practical aspects are crucial when applying RL to physical systems:

### 6.6.1 Reward Shaping

Designing effective reward functions is critical for learning in physical systems:

```python
def shaped_reward(task, current_state, target_state, action, prev_state=None):
    """
    Comprehensive reward function for physical tasks
    """
    # Distance to goal
    distance_to_goal = np.linalg.norm(current_state[:2] - target_state[:2])

    # Smoothness penalty (action regularization)
    action_penalty = 0.01 * np.sum(np.square(action))

    # Velocity penalty to encourage smooth motion
    if prev_state is not None:
        velocity_penalty = 0.001 * np.sum(np.square(current_state - prev_state))
    else:
        velocity_penalty = 0

    # Goal achievement bonus
    goal_bonus = 10.0 if distance_to_goal < 0.1 else 0

    # Combine components
    reward = -distance_to_goal - action_penalty - velocity_penalty + goal_bonus

    return reward
```

### 6.6.2 Hyperparameter Tuning

Physical systems often require careful hyperparameter tuning:

```python
def hyperparameter_search(env, agent_class, param_grid):
    """Grid search for optimal hyperparameters"""
    best_score = float('-inf')
    best_params = {}

    for params in param_grid:
        agent = agent_class(**params)
        score = evaluate_agent(env, agent, num_episodes=100)

        if score > best_score:
            best_score = score
            best_params = params

    return best_params, best_score

def evaluate_agent(env, agent, num_episodes=100):
    """Evaluate agent performance"""
    total_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state

        total_rewards.append(episode_reward)

    return np.mean(total_rewards)
```

## Key Takeaways

- Reinforcement learning in physical domains requires specialized algorithms for continuous action spaces
- Policy gradient and actor-critic methods are preferred over value-based methods
- Exploration strategies must balance between discovery and safety constraints
- Model-based approaches can significantly improve sample efficiency
- Safe RL techniques are essential to prevent damage to physical systems during learning
- Careful reward design is crucial for successful learning in physical tasks

## Exercises

1. **Theoretical**: Prove that the policy gradient theorem holds for continuous action spaces with Gaussian policies.

2. **Coding**: Implement the TD3 (Twin Delayed DDPG) algorithm and compare its performance with DDPG on a simulated robotic manipulation task.

3. **Theoretical**: Explain why the Ornstein-Uhlenbeck process is preferred over white noise for exploration in physical control tasks.

4. **Coding**: Design a curiosity-driven exploration module for a 2D point-mass navigation task and demonstrate its effectiveness compared to random exploration.

5. **Coding**: Implement a constrained RL algorithm that prevents a simulated robot from exceeding joint angle limits during learning.

## Further Reading

1. Lillicrap, T. P., et al. (2016). "Continuous control with deep reinforcement learning." *ICLR*.

2. Fujimoto, S., et al. (2018). "Addressing function approximation error in actor-critic methods." *ICML*.

3. Achiam, J., et al. (2017). "Constrained policy optimization." *ICML*.