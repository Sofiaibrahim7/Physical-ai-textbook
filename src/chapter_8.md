# Chapter 8: Learning from Demonstration and Imitation

Learning from Demonstration (LfD) and imitation learning are powerful paradigms that enable physical AI systems to acquire skills by observing and replicating expert behavior. These approaches address the challenge of specifying complex behaviors through direct programming, instead leveraging human expertise or demonstrations to guide learning.

## 8.1 Introduction to Imitation Learning

Imitation learning, also known as learning from demonstration, involves learning a policy $\pi(a|s)$ that mimics expert behavior demonstrated through trajectories $\tau = \{(s_0, a_0), (s_1, a_1), ..., (s_T, a_T)\}$. The fundamental assumption is that expert demonstrations provide high-quality examples of desired behavior.

The imitation learning problem can be formulated as:

$$\pi^* = \arg\min_\pi \mathbb{E}_{s \sim D_{expert}}[D_{KL}(π_{expert}(\cdot|s) || π(\cdot|s))]$$

where $D_{KL}$ is the Kullback-Leibler divergence and $D_{expert}$ represents the state distribution under expert demonstrations.

![Figure 8.1: Imitation Learning Framework](placeholder)

## 8.2 Behavioral Cloning

Behavioral cloning (BC) is the simplest approach to imitation learning, treating it as a supervised learning problem where actions are predicted from states using expert demonstrations.

### 8.2.1 Basic Behavioral Cloning

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class BehavioralCloningNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(BehavioralCloningNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class BehavioralCloning:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.network = BehavioralCloningNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()

    def train(self, expert_states, expert_actions, epochs=1000):
        """Train the behavioral cloning network"""
        states = torch.FloatTensor(expert_states)
        actions = torch.FloatTensor(expert_actions)

        for epoch in range(epochs):
            pred_actions = self.network(states)
            loss = self.criterion(pred_actions, actions)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f"BC Epoch {epoch}, Loss: {loss.item():.6f}")

    def predict(self, state):
        """Predict action for given state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.network(state_tensor).squeeze(0).numpy()
        return action

# Example usage for robotic manipulation
def collect_expert_demonstrations(env, expert_policy, num_demos=100):
    """Collect expert demonstrations"""
    states = []
    actions = []

    for demo in range(num_demos):
        state = env.reset()
        done = False

        while not done:
            action = expert_policy(state)  # Expert provides action
            next_state, reward, done, _ = env.step(action)

            states.append(state)
            actions.append(action)

            state = next_state

    return np.array(states), np.array(actions)

# Train behavioral cloning agent
def train_bc_agent(env, expert_policy, state_dim, action_dim):
    """Train behavioral cloning agent"""
    expert_states, expert_actions = collect_expert_demonstrations(
        env, expert_policy, num_demos=200
    )

    bc_agent = BehavioralCloning(state_dim, action_dim)
    bc_agent.train(expert_states, expert_actions)

    return bc_agent
```

### 8.2.2 Data Aggregation (DAgger)

Behavioral cloning suffers from covariate shift - the agent encounters states not seen in expert demonstrations. DAgger addresses this by iteratively collecting new data:

```python
class DAgger:
    def __init__(self, state_dim, action_dim, lr=1e-3):
        self.network = BehavioralCloningNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.network.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.all_states = []
        self.all_actions = []

    def train_iteration(self, env, expert_policy, bc_agent, num_rollouts=10):
        """Perform one iteration of DAgger"""
        # Collect new data using current policy
        new_states = []
        new_actions = []

        for rollout in range(num_rollouts):
            state = env.reset()
            done = False

            while not done:
                # Use current policy to collect states
                action = bc_agent.predict(state)
                new_states.append(state.copy())

                # Get expert action for these states
                expert_action = expert_policy(state)
                new_actions.append(expert_action)

                # Take expert action to stay on expert trajectory
                state, _, done, _ = env.step(expert_action)

        # Add new data to dataset
        self.all_states.extend(new_states)
        self.all_actions.extend(new_actions)

        # Retrain network with all data
        states_tensor = torch.FloatTensor(self.all_states)
        actions_tensor = torch.FloatTensor(self.all_actions)

        for epoch in range(500):  # Train on combined dataset
            pred_actions = self.network(states_tensor)
            loss = self.criterion(pred_actions, actions_tensor)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        print(f"DAgger iteration complete, total data points: {len(self.all_states)}")

        return loss.item()

    def predict(self, state):
        """Predict action for given state"""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = self.network(state_tensor).squeeze(0).numpy()
        return action
```

## 8.3 Inverse Reinforcement Learning

Inverse Reinforcement Learning (IRL) aims to infer the reward function that the expert is optimizing, rather than directly mimicking actions.

### 8.3.1 Maximum Causal Entropy IRL

```python
class MaxCausalEntropyIRL(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(MaxCausalEntropyIRL, self).__init__()
        # Reward network
        self.reward_network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Policy network (for generating rollouts)
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def reward(self, state, action):
        """Compute reward for state-action pair"""
        x = torch.cat([state, action], dim=-1)
        return self.reward_network(x)

    def policy(self, state):
        """Compute policy for state"""
        return self.policy_network(state)

def max_causal_entropy_irl(expert_trajectories, env, state_dim, action_dim, epochs=1000):
    """Maximum Causal Entropy IRL implementation"""
    irl_model = MaxCausalEntropyIRL(state_dim, action_dim)
    reward_optimizer = optim.Adam(irl_model.reward_network.parameters(), lr=1e-3)
    policy_optimizer = optim.Adam(irl_model.policy_network.parameters(), lr=1e-3)

    expert_states = []
    expert_actions = []

    for traj in expert_trajectories:
        for (state, action) in traj:
            expert_states.append(state)
            expert_actions.append(action)

    expert_states = torch.FloatTensor(expert_states)
    expert_actions = torch.FloatTensor(expert_actions)

    for epoch in range(epochs):
        # Generate trajectories using current policy
        policy_trajectories = []
        for _ in range(10):  # Generate 10 policy trajectories
            traj = []
            state = env.reset()
            done = False

            while not done:
                action = irl_model.policy(torch.FloatTensor(state).unsqueeze(0)).squeeze(0).numpy()
                next_state, _, done, _ = env.step(action)
                traj.append((state, action))
                state = next_state

            policy_trajectories.extend(traj)

        # Compute discriminator loss (discriminate expert vs policy)
        expert_rewards = []
        for state, action in zip(expert_states, expert_actions):
            r = irl_model.reward(state.unsqueeze(0), action.unsqueeze(0))
            expert_rewards.append(r)

        policy_rewards = []
        for state, action in policy_trajectories:
            r = irl_model.reward(
                torch.FloatTensor(state).unsqueeze(0),
                torch.FloatTensor(action).unsqueeze(0)
            )
            policy_rewards.append(r)

        # Discriminator loss: expert should have high reward, policy should have low reward
        expert_loss = -torch.mean(torch.stack(expert_rewards))
        policy_loss = torch.mean(torch.stack(policy_rewards))
        discriminator_loss = expert_loss + policy_loss

        reward_optimizer.zero_grad()
        discriminator_loss.backward()
        reward_optimizer.step()

        # Update policy to maximize expected reward
        policy_loss = 0
        for state, action in policy_trajectories:
            r = irl_model.reward(
                torch.FloatTensor(state).unsqueeze(0),
                torch.FloatTensor(action).unsqueeze(0)
            )
            policy_loss -= r  # Minimize negative reward (maximize reward)

        policy_optimizer.zero_grad()
        policy_loss.backward()
        policy_optimizer.step()

        if epoch % 100 == 0:
            print(f"IRL Epoch {epoch}, Discriminator Loss: {discriminator_loss.item():.4f}")
```

## 8.4 Generative Adversarial Imitation Learning

Generative Adversarial Imitation Learning (GAIL) uses adversarial training to match the state-action distribution between expert and agent:

```python
class GAILDiscriminator(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super(GAILDiscriminator, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Probability that trajectory is expert
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        return self.network(x)

class GAIL:
    def __init__(self, state_dim, action_dim, lr_actor=1e-4, lr_disc=1e-3):
        self.policy_network = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Actions in [-1, 1]
        )

        self.discriminator = GAILDiscriminator(state_dim, action_dim)

        self.actor_optimizer = optim.Adam(self.policy_network.parameters(), lr=lr_actor)
        self.disc_optimizer = optim.Adam(self.discriminator.parameters(), lr=lr_disc)

        self.criterion = nn.BCELoss()

    def compute_reward(self, state, action):
        """Compute reward as -log(1-D(s,a)) for policy learning"""
        with torch.no_grad():
            disc_output = self.discriminator(state, action)
            # Use log(D) as reward (maximizing this is equivalent to fooling discriminator)
            reward = torch.log(disc_output + 1e-8) - torch.log(1 - disc_output + 1e-8)
        return reward.squeeze().numpy()

    def update_discriminator(self, expert_states, expert_actions, policy_states, policy_actions):
        """Update discriminator to distinguish expert vs policy"""
        # Convert to tensors
        expert_s = torch.FloatTensor(expert_states)
        expert_a = torch.FloatTensor(expert_actions)
        policy_s = torch.FloatTensor(policy_states)
        policy_a = torch.FloatTensor(policy_actions)

        # Labels: 1 for expert, 0 for policy
        expert_labels = torch.ones(len(expert_states), 1)
        policy_labels = torch.zeros(len(policy_states), 1)

        # Discriminator loss
        expert_pred = self.discriminator(expert_s, expert_a)
        policy_pred = self.discriminator(policy_s, policy_a)

        expert_loss = self.criterion(expert_pred, expert_labels)
        policy_loss = self.criterion(policy_pred, policy_labels)

        disc_loss = expert_loss + policy_loss

        self.disc_optimizer.zero_grad()
        disc_loss.backward()
        self.disc_optimizer.step()

        return disc_loss.item()

    def update_policy(self, states, actions, rewards):
        """Update policy using computed rewards"""
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards)

        # Normalize rewards for stability
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + 1e-8)

        # Policy gradient update
        log_probs = -((actions_tensor - self.policy_network(states_tensor)) ** 2).sum(dim=1) / 2
        policy_loss = -(log_probs * rewards_tensor).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        return policy_loss.item()

def train_gail(env, expert_trajectories, state_dim, action_dim, epochs=5000):
    """Train GAIL agent"""
    gail_agent = GAIL(state_dim, action_dim)

    # Extract expert data
    expert_states = []
    expert_actions = []
    for traj in expert_trajectories:
        for (state, action) in traj:
            expert_states.append(state)
            expert_actions.append(action)

    for epoch in range(epochs):
        # Collect policy trajectories
        policy_states = []
        policy_actions = []
        policy_rewards = []

        state = env.reset()
        done = False

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            action = gail_agent.policy_network(state_tensor).squeeze(0).numpy()

            next_state, _, done, _ = env.step(action)

            policy_states.append(state)
            policy_actions.append(action)

            # Compute GAIL reward
            reward = gail_agent.compute_reward(
                state_tensor,
                torch.FloatTensor(action).unsqueeze(0)
            )
            policy_rewards.append(reward)

            state = next_state

        # Update discriminator
        disc_loss = gail_agent.update_discriminator(
            expert_states, expert_actions,
            policy_states, policy_actions
        )

        # Update policy
        policy_loss = gail_agent.update_policy(
            policy_states, policy_actions, policy_rewards
        )

        if epoch % 500 == 0:
            print(f"GAIL Epoch {epoch}, Disc Loss: {disc_loss:.4f}, Policy Loss: {policy_loss:.4f}")

    return gail_agent
```

## 8.5 Learning from Observational Data

In many physical systems, we only have observational data (state sequences) rather than action demonstrations. This requires more sophisticated techniques:

### 8.5.1 Video-to-Action Mapping

```python
import torch.nn.functional as F

class VideoImitationNetwork(nn.Module):
    def __init__(self, image_channels, action_dim, hidden_dim=256):
        super(VideoImitationNetwork, self).__init__()
        # CNN for image processing
        self.cnn = nn.Sequential(
            nn.Conv2d(image_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        # Calculate CNN output size
        self.cnn_output_size = self._get_cnn_output_size((image_channels, 84, 84))

        # Combine CNN features with state features
        self.network = nn.Sequential(
            nn.Linear(self.cnn_output_size, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def _get_cnn_output_size(self, input_shape):
        """Calculate output size of CNN"""
        with torch.no_grad():
            dummy_input = torch.zeros(1, *input_shape)
            output = self.cnn(dummy_input)
            return output.shape[1]

    def forward(self, image):
        features = self.cnn(image)
        return self.network(features)

def train_visual_imitation(visual_expert_data, action_labels, image_shape, action_dim):
    """Train imitation learning from visual demonstrations"""
    agent = VideoImitationNetwork(image_shape[0], action_dim)
    optimizer = optim.Adam(agent.parameters(), lr=1e-4)
    criterion = nn.MSELoss()

    images = torch.FloatTensor(visual_expert_data)
    actions = torch.FloatTensor(action_labels)

    for epoch in range(1000):
        pred_actions = agent(images)
        loss = criterion(pred_actions, actions)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Visual Imitation Epoch {epoch}, Loss: {loss.item():.6f}")

    return agent
```

## 8.6 Hierarchical Imitation Learning

Complex physical tasks often have hierarchical structure. Hierarchical imitation learning learns both high-level goals and low-level skills:

```python
class HierarchicalImitationLearner:
    def __init__(self, state_dim, action_dim, num_skills, hidden_dim=256):
        # Skill selector (high-level policy)
        self.skill_selector = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_skills),
            nn.Softmax(dim=-1)
        )

        # Skill-specific policies (low-level policies)
        self.skill_policies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, action_dim)
            ) for _ in range(num_skills)
        ])

        self.num_skills = num_skills
        self.optimizer = optim.Adam(
            list(self.skill_selector.parameters()) +
            list(self.skill_policies.parameters()),
            lr=1e-3
        )

    def forward(self, state):
        """Get action by selecting skill and executing skill policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Select skill
        skill_probs = self.skill_selector(state_tensor)
        skill_idx = torch.multinomial(skill_probs.squeeze(0), 1).item()

        # Execute selected skill
        action = self.skill_policies[skill_idx](state_tensor).squeeze(0)

        return action.numpy(), skill_idx

    def train_hierarchical(self, expert_data, epochs=1000):
        """Train hierarchical imitation learning"""
        states = torch.FloatTensor([d[0] for d in expert_data])
        actions = torch.FloatTensor([d[1] for d in expert_data])
        skills = torch.LongTensor([d[2] for d in expert_data])  # Skill labels

        for epoch in range(epochs):
            # Skill selection loss
            skill_probs = self.skill_selector(states)
            skill_loss = F.cross_entropy(skill_probs, skills)

            # Action prediction loss for each skill
            total_action_loss = 0
            for skill_idx in range(self.num_skills):
                skill_mask = (skills == skill_idx)
                if skill_mask.sum() > 0:
                    skill_states = states[skill_mask]
                    skill_actions = actions[skill_mask]
                    pred_actions = self.skill_policies[skill_idx](skill_states)
                    skill_action_loss = F.mse_loss(pred_actions, skill_actions)
                    total_action_loss += skill_action_loss

            total_loss = skill_loss + total_action_loss

            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            if epoch % 100 == 0:
                print(f"Hierarchical Imitation Epoch {epoch}, Total Loss: {total_loss.item():.4f}")
```

## 8.7 Practical Considerations

### 8.7.1 Data Quality and Preprocessing

```python
def preprocess_demonstration_data(states, actions, window_size=5):
    """Preprocess demonstration data with smoothing and filtering"""
    # Smooth actions to reduce noise
    smoothed_actions = []
    for i in range(len(actions)):
        start_idx = max(0, i - window_size // 2)
        end_idx = min(len(actions), i + window_size // 2 + 1)
        smoothed_action = np.mean(actions[start_idx:end_idx], axis=0)
        smoothed_actions.append(smoothed_action)

    # Normalize states
    states = np.array(states)
    actions = np.array(smoothed_actions)

    state_mean = np.mean(states, axis=0)
    state_std = np.std(states, axis=0) + 1e-8
    states = (states - state_mean) / state_std

    return states, actions, state_mean, state_std

def detect_expert_suboptimalities(trajectories):
    """Detect suboptimal behaviors in expert demonstrations"""
    suboptimal_trajectories = []

    for traj in trajectories:
        # Simple heuristic: detect oscillations or inefficient paths
        path_length = 0
        for i in range(1, len(traj)):
            path_length += np.linalg.norm(traj[i][0] - traj[i-1][0])

        # Compare with straight-line distance
        start_state = traj[0][0]
        end_state = traj[-1][0]
        straight_line_dist = np.linalg.norm(end_state - start_state)

        efficiency_ratio = straight_line_dist / (path_length + 1e-8)

        if efficiency_ratio < 0.5:  # If path is more than 2x longer than straight line
            suboptimal_trajectories.append((traj, efficiency_ratio))

    return suboptimal_trajectories
```

## Key Takeaways

- Behavioral cloning is simple but suffers from covariate shift; DAgger addresses this with iterative data collection
- Inverse reinforcement learning infers reward functions from demonstrations rather than mimicking actions directly
- GAIL uses adversarial training to match state-action distributions between expert and agent
- Visual imitation learning enables learning from video demonstrations
- Hierarchical imitation learning decomposes complex tasks into skill-based sub-problems
- Data quality and preprocessing are crucial for successful imitation learning
- Expert demonstrations should be optimal and consistent for best results

## Exercises

1. **Coding**: Implement behavioral cloning and DAgger on a simple robotic manipulation task and compare their performance.

2. **Theoretical**: Prove that the DAgger algorithm converges to the expert policy under certain conditions.

3. **Coding**: Design a GAIL implementation for a navigation task and compare it with behavioral cloning.

4. **Theoretical**: Explain why inverse reinforcement learning can be more sample-efficient than direct imitation.

5. **Coding**: Implement hierarchical imitation learning for a multi-step manipulation task with clear sub-goals.

## Further Reading

1. Ross, S., Gordon, G., & Bagnell, J. (2011). "A reduction of imitation learning and structured prediction to no-regret online learning." *AISTATS*.

2. Ho, J., & Ermon, S. (2016). "Generative adversarial imitation learning." *NIPS*.

3. Finn, C., et al. (2017). "A connection between generative adversarial networks, inverse reinforcement learning, and energy-based models." *NIPS*.