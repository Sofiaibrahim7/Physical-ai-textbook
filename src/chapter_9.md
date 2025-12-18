# Chapter 9: Multi-Agent Physical AI and Interaction

Multi-Agent Physical AI deals with scenarios where multiple embodied agents interact with each other and the environment. These systems present unique challenges including coordination, communication, competition, and cooperation in physical spaces. Unlike single-agent systems, multi-agent physical AI must handle decentralized decision-making, partial observability, and dynamic interaction patterns.

## 9.1 Fundamentals of Multi-Agent Systems

In multi-agent physical systems, we have $N$ agents, each with its own state $s_i$, action $a_i$, and policy $\pi_i$. The joint state is $s = (s_1, s_2, ..., s_N)$, and the joint action is $a = (a_1, a_2, ..., a_N)$. The challenge lies in learning policies that account for the interdependencies between agents.

The multi-agent decision problem can be formulated as a Decentralized Partially Observable Markov Decision Process (Dec-POMDP):

$$\max_{\pi_1, \pi_2, ..., \pi_N} \mathbb{E}\left[\sum_{t=0}^T \sum_{i=1}^N r_i(s_t, a_t)\right]$$

where each agent $i$ only has access to its local observation $o_i^t$ and must act based on its local policy $\pi_i$.

![Figure 9.1: Multi-Agent Physical AI System Architecture](placeholder)

## 9.2 Coordination and Communication

### 9.2.1 Communication Protocols

Agents in physical systems can communicate through various channels:

```python
import torch
import torch.nn as nn
import numpy as np

class CommunicationNetwork(nn.Module):
    def __init__(self, agent_id, state_dim, action_dim, message_dim=32, num_agents=3):
        super().__init__()
        self.agent_id = agent_id
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.message_dim = message_dim
        self.num_agents = num_agents

        # Encoder: convert local state to message
        self.encoder = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, message_dim)
        )

        # Decoder: process received messages
        self.decoder = nn.Sequential(
            nn.Linear(message_dim * (num_agents - 1), 128),  # All other agents' messages
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim)  # Augmented state representation
        )

        # Policy network with communication
        self.policy = nn.Sequential(
            nn.Linear(state_dim + state_dim, 256),  # Original state + communication state
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim)
        )

    def encode_message(self, local_state):
        """Encode local state into message"""
        return self.encoder(torch.FloatTensor(local_state))

    def decode_messages(self, received_messages):
        """Decode messages from other agents"""
        if len(received_messages) == 0:
            return torch.zeros(self.state_dim)

        # Concatenate all received messages
        all_messages = torch.cat(received_messages, dim=-1)
        return self.decoder(all_messages)

    def forward(self, local_state, received_messages):
        """Forward pass with communication"""
        # Get communication-enhanced state
        comm_state = self.decode_messages(received_messages)

        # Combine local and communication states
        combined_state = torch.cat([
            torch.FloatTensor(local_state),
            comm_state
        ], dim=-1)

        # Get action
        action = self.policy(combined_state)
        return torch.tanh(action)  # Actions in [-1, 1]

class MultiAgentSystem:
    def __init__(self, num_agents, state_dim, action_dim, message_dim=32):
        self.num_agents = num_agents
        self.agents = [
            CommunicationNetwork(i, state_dim, action_dim, message_dim, num_agents)
            for i in range(num_agents)
        ]
        self.optimizer = torch.optim.Adam(
            [p for agent in self.agents for p in agent.parameters()],
            lr=1e-3
        )

    def get_actions(self, states):
        """Get actions from all agents with communication"""
        # First, all agents encode their messages
        messages = []
        for i, agent in enumerate(self.agents):
            msg = agent.encode_message(states[i])
            messages.append(msg)

        # Then, each agent processes messages from others and computes action
        actions = []
        for i, agent in enumerate(self.agents):
            # Collect messages from other agents
            other_messages = [msg for j, msg in enumerate(messages) if j != i]
            action = agent(states[i], other_messages)
            actions.append(action.detach().numpy())

        return actions
```

### 9.2.2 Communication-Constrained Scenarios

```python
class BandwidthLimitedCommunication:
    def __init__(self, max_message_size=16, compression_ratio=0.5):
        self.max_message_size = max_message_size
        self.compression_ratio = compression_ratio

        # Compression network
        self.compressor = nn.Sequential(
            nn.Linear(32, 64),  # Assuming original message is 32-dim
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, int(32 * compression_ratio))
        )

        # Decompression network
        self.decompressor = nn.Sequential(
            nn.Linear(int(32 * compression_ratio), 32),
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 32)
        )

    def compress_message(self, message):
        """Compress message to fit bandwidth constraints"""
        compressed = self.compressor(message)
        return compressed

    def decompress_message(self, compressed_message):
        """Decompress received message"""
        return self.decompressor(compressed_message)

    def communicate(self, messages):
        """Process communication with bandwidth constraints"""
        compressed_messages = [self.compress_message(msg) for msg in messages]
        decompressed_messages = [self.decompress_message(cmsg) for cmsg in compressed_messages]
        return decompressed_messages

class CommunicationStrategyLearner:
    def __init__(self, state_dim, action_dim, message_dim=32):
        # Network to decide whether to communicate
        self.communication_decision = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2),  # 0: no communication, 1: communicate
            nn.Softmax(dim=-1)
        )

        # Message content generator
        self.message_generator = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, message_dim)
        )

    def should_communicate(self, state):
        """Decide whether to communicate based on state"""
        decision_probs = self.communication_decision(torch.FloatTensor(state))
        return torch.argmax(decision_probs).item()

    def generate_message(self, state):
        """Generate message content based on state"""
        return self.message_generator(torch.FloatTensor(state))
```

## 9.3 Cooperative Multi-Agent Learning

### 9.3.1 Centralized Training with Decentralized Execution (CTDE)

```python
class CentralizedCritic(nn.Module):
    def __init__(self, num_agents, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Centralized critic that sees global state and all actions
        self.critic = nn.Sequential(
            nn.Linear(num_agents * state_dim + num_agents * action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, global_state, global_actions):
        """Evaluate joint state-action value"""
        x = torch.cat([global_state, global_actions], dim=-1)
        return self.critic(x)

class MADDPGAgent:
    def __init__(self, agent_id, num_agents, state_dim, action_dim, lr_actor=1e-4, lr_critic=1e-3):
        self.agent_id = agent_id
        self.num_agents = num_agents

        # Local actor (decentralized execution)
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        # Local critic target
        self.actor_target = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        # Copy parameters to target
        self.hard_update(self.actor_target, self.actor)

        # Centralized critic (for training only)
        self.critic = CentralizedCritic(num_agents, state_dim, action_dim)
        self.critic_target = CentralizedCritic(num_agents, state_dim, action_dim)
        self.hard_update(self.critic_target, self.critic)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.tau = 0.01  # Target network update rate

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + param.data * self.tau
            )

    def select_action(self, state, add_noise=True, noise_scale=0.1):
        """Select action using local policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state_tensor).squeeze(0).detach().numpy()

        if add_noise:
            noise = np.random.normal(0, noise_scale, size=action.shape)
            action = np.clip(action + noise, -1, 1)

        return action

    def update(self, states, actions, rewards, next_states, dones, gamma=0.99):
        """Update using centralized critic"""
        # Convert to tensors
        states_tensor = torch.FloatTensor(states)
        actions_tensor = torch.FloatTensor(actions)
        rewards_tensor = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_tensor = torch.FloatTensor(next_states)
        dones_tensor = torch.BoolTensor(dones).unsqueeze(1)

        # Get next actions using target actors
        next_actions = []
        for i in range(self.num_agents):
            next_action = self.actor_target(
                next_states_tensor[:, i * self.state_dim:(i + 1) * self.state_dim]
            )
            next_actions.append(next_action)
        next_actions_tensor = torch.cat(next_actions, dim=1)

        # Compute target Q-values
        next_q_values = self.critic_target(next_states_tensor, next_actions_tensor)
        target_q_values = rewards_tensor + (gamma * next_q_values * ~dones_tensor)

        # Current Q-values
        current_q_values = self.critic(states_tensor, actions_tensor)
        critic_loss = nn.MSELoss()(current_q_values, target_q_values.detach())

        # Actor loss (maximize Q-value for own actions)
        own_actions = self.actor(states_tensor[:, self.agent_id * self.state_dim:(self.agent_id + 1) * self.state_dim])

        # Replace agent's action in the joint action tensor
        temp_actions = actions_tensor.clone()
        temp_actions[:, self.agent_id * self.action_dim:(self.agent_id + 1) * self.action_dim] = own_actions

        actor_loss = -self.critic(states_tensor, temp_actions).mean()

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

class MultiAgentEnvironment:
    def __init__(self, num_agents, state_dim, action_dim):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agents = [
            MADDPGAgent(i, num_agents, state_dim, action_dim)
            for i in range(num_agents)
        ]

    def train_episode(self, env):
        """Train all agents for one episode"""
        states = env.reset()
        total_rewards = [0] * self.num_agents
        done = False

        while not done:
            # Get actions from all agents
            actions = []
            for i, agent in enumerate(self.agents):
                action = agent.select_action(states[i])
                actions.append(action)

            # Execute joint action in environment
            next_states, rewards, done, _ = env.step(actions)

            # Store transition for each agent
            for i, agent in enumerate(self.agents):
                agent.experience(states[i], actions[i], rewards[i],
                               next_states[i], done)

            states = next_states
            total_rewards = [tr + r for tr, r in zip(total_rewards, rewards)]

        # Update all agents
        for i, agent in enumerate(self.agents):
            agent.update_experience_buffer()
            agent.update_networks()

        return total_rewards
```

## 9.4 Competitive Multi-Agent Scenarios

### 9.4.1 Game-Theoretic Approaches

```python
class NashQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, num_agents):
        super().__init__()
        self.num_agents = num_agents
        self.action_dim = action_dim

        # Q-network for each agent
        self.q_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim + (num_agents-1) * action_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim)  # Q-values for each action
            ) for _ in range(num_agents)
        ])

    def forward(self, agent_id, state, other_actions):
        """Get Q-values for specific agent"""
        # Concatenate state and other agents' actions
        if len(other_actions) == 0:
            x = torch.FloatTensor(state)
        else:
            x = torch.cat([torch.FloatTensor(state)] + [torch.FloatTensor(a) for a in other_actions])

        return self.q_networks[agent_id](x)

def compute_nash_equilibrium(q_values_list):
    """Compute Nash equilibrium given Q-values for all agents"""
    # Simple best response dynamics
    num_agents = len(q_values_list)
    num_actions = q_values_list[0].shape[0]

    # Initialize random mixed strategy
    strategies = [torch.ones(num_actions) / num_actions for _ in range(num_agents)]

    # Best response iteration
    for iteration in range(100):
        new_strategies = []
        for i in range(num_agents):
            # Compute expected utility given other agents' strategies
            expected_utilities = torch.zeros(num_actions)
            for a in range(num_actions):
                for other_actions in torch.cartesian_prod(*[torch.arange(num_actions) for _ in range(num_agents-1)]):
                    # Construct joint action
                    joint_action = []
                    other_idx = 0
                    for j in range(num_agents):
                        if j == i:
                            joint_action.append(a)
                        else:
                            joint_action.append(other_actions[other_idx].item())
                            other_idx += 1

                    # Compute probability of this joint action
                    prob = 1.0
                    for j, action in enumerate(joint_action):
                        if j != i:
                            prob *= strategies[j][action]

                    # Add utility contribution
                    expected_utilities[a] += prob * q_values_list[i][a]

            # Best response: pure strategy with highest expected utility
            best_action = torch.argmax(expected_utilities).item()
            new_strategy = torch.zeros(num_actions)
            new_strategy[best_action] = 1.0
            new_strategies.append(new_strategy)

        strategies = new_strategies

    return strategies

class CompetitiveMultiAgent:
    def __init__(self, num_agents, state_dim, action_dim):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.q_network = NashQNetwork(state_dim, action_dim, num_agents)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=1e-3)

    def get_equilibrium_actions(self, state):
        """Get Nash equilibrium actions for current state"""
        # Get Q-values for all agents
        q_values_list = []
        for i in range(self.num_agents):
            # Get other agents' actions (simplified as fixed for this example)
            other_actions = []
            q_values = self.q_network(i, state, other_actions)
            q_values_list.append(q_values)

        # Compute Nash equilibrium
        strategies = compute_nash_equilibrium(q_values_list)

        # Sample actions from equilibrium strategies
        actions = []
        for strategy in strategies:
            action = torch.multinomial(strategy, 1).item()
            actions.append(action)

        return actions
```

## 9.5 Physical Interaction Models

### 9.5.1 Contact and Collision Handling

```python
class PhysicalInteractionModel(nn.Module):
    def __init__(self, num_agents, state_dim, action_dim):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = state_dim

        # Model for predicting interaction outcomes
        self.interaction_predictor = nn.Sequential(
            nn.Linear(num_agents * state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_agents * state_dim)  # Predict next states for all agents
        )

        # Contact detection network
        self.contact_detector = nn.Sequential(
            nn.Linear(num_agents * state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),  # Contact probability
            nn.Sigmoid()
        )

    def detect_contact(self, agent_states):
        """Detect if agents are in contact"""
        flat_states = torch.cat(agent_states, dim=-1)
        contact_prob = self.contact_detector(flat_states)
        return contact_prob > 0.5  # Return contact as boolean

    def predict_interaction(self, agent_states):
        """Predict outcome of physical interactions"""
        flat_states = torch.cat(agent_states, dim=-1)
        predicted_next_states = self.interaction_predictor(flat_states)

        # Reshape to individual agent states
        individual_states = []
        for i in range(self.num_agents):
            start_idx = i * self.state_dim
            end_idx = (i + 1) * self.state_dim
            individual_states.append(
                predicted_next_states[:, start_idx:end_idx]
            )

        return individual_states

class MultiAgentPhysicsEnvironment:
    def __init__(self, num_agents, state_dim, action_dim):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.interaction_model = PhysicalInteractionModel(num_agents, state_dim, action_dim)

        # Store agent positions for contact detection
        self.agent_positions = [None] * num_agents

    def step(self, actions):
        """Step environment with physical interactions"""
        # Apply actions to get preliminary next states
        next_states = []
        rewards = []

        for i in range(self.num_agents):
            # Simplified physics update
            new_state = self.update_agent_state(i, actions[i])
            next_states.append(new_state)

        # Check for interactions
        contact = self.interaction_model.detect_contact(next_states)

        if contact:
            # Apply interaction model
            next_states = self.interaction_model.predict_interaction(next_states)

        # Compute rewards considering interactions
        for i in range(self.num_agents):
            reward = self.compute_agent_reward(i, next_states[i], next_states)
            rewards.append(reward)

        return next_states, rewards, False, {}

    def update_agent_state(self, agent_id, action):
        """Update individual agent state"""
        # Simplified state update (position, velocity, etc.)
        current_pos = self.agent_positions[agent_id] if self.agent_positions[agent_id] is not None else np.zeros(2)
        new_pos = current_pos + action[:2]  # Assuming first 2 dims are position change
        self.agent_positions[agent_id] = new_pos

        # Return state with position and other features
        state = np.concatenate([new_pos, action[2:]])  # Position + other state features
        return state

    def compute_agent_reward(self, agent_id, agent_state, all_states):
        """Compute reward considering multi-agent interactions"""
        # Base reward
        base_reward = 0

        # Contact-based rewards
        for i, other_state in enumerate(all_states):
            if i != agent_id:
                dist = np.linalg.norm(agent_state[:2] - other_state[:2])
                if dist < 0.5:  # Close proximity
                    base_reward -= 0.1  # Penalty for collision risk

        return base_reward
```

## 9.6 Emergent Behaviors and Coordination

### 9.6.1 Learning to Coordinate

```python
class CoordinationLearner:
    def __init__(self, num_agents, state_dim, action_dim):
        self.num_agents = num_agents

        # Individual agent policies
        self.policies = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, action_dim),
                nn.Tanh()
            ) for _ in range(num_agents)
        ])

        # Coordination critic
        self.coordination_critic = nn.Sequential(
            nn.Linear(num_agents * state_dim + num_agents * action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        # Optimizers
        self.policy_optimizers = [
            torch.optim.Adam(policy.parameters(), lr=1e-4)
            for policy in self.policies
        ]
        self.critic_optimizer = torch.optim.Adam(
            self.coordination_critic.parameters(), lr=1e-3
        )

    def compute_coordination_reward(self, states, actions):
        """Compute reward based on coordination quality"""
        # Flatten states and actions
        flat_states = torch.cat(states, dim=-1)
        flat_actions = torch.cat(actions, dim=-1)

        # Coordination score based on critic
        coordination_score = self.coordination_critic(
            torch.cat([flat_states, flat_actions], dim=-1)
        )

        # Distribute coordination reward to all agents
        individual_rewards = [coordination_score / self.num_agents for _ in range(self.num_agents)]

        return individual_rewards

    def update_policies(self, states, actions, coordination_rewards):
        """Update policies based on coordination rewards"""
        for i in range(self.num_agents):
            state_tensor = torch.FloatTensor(states[i]).unsqueeze(0)
            action_tensor = self.policies[i](state_tensor)

            # Coordination reward for this agent
            coord_reward = coordination_rewards[i]

            # Policy loss (negative of coordination reward)
            policy_loss = -coord_reward

            self.policy_optimizers[i].zero_grad()
            policy_loss.backward()
            self.policy_optimizers[i].step()

def multi_agent_training_loop(env, coordination_learner, episodes=5000):
    """Training loop for multi-agent coordination"""
    for episode in range(episodes):
        states = env.reset()
        total_rewards = [0] * env.num_agents
        done = False

        while not done:
            # Get actions from all policies
            actions = []
            state_tensors = []
            for i in range(env.num_agents):
                state_tensor = torch.FloatTensor(states[i]).unsqueeze(0)
                action = coordination_learner.policies[i](state_tensor).squeeze(0).detach().numpy()
                actions.append(action)
                state_tensors.append(state_tensor)

            # Execute in environment
            next_states, base_rewards, done, _ = env.step(actions)

            # Compute coordination rewards
            coordination_rewards = coordination_learner.compute_coordination_reward(
                [torch.FloatTensor(s) for s in states],
                [torch.FloatTensor(a) for a in actions]
            )

            # Update policies with coordination rewards
            coordination_learner.update_policies(
                [torch.FloatTensor(s) for s in states],
                [torch.FloatTensor(a) for a in actions],
                coordination_rewards
            )

            states = next_states
            total_rewards = [tr + br for tr, br in zip(total_rewards, base_rewards)]

        if episode % 500 == 0:
            avg_reward = np.mean(total_rewards)
            print(f"Episode {episode}, Average Reward: {avg_reward:.2f}")
```

## 9.7 Challenges and Solutions

### 9.7.1 Non-Stationarity Problem

In multi-agent systems, each agent's policy changes during training, making the environment non-stationary from any single agent's perspective:

```python
class OpponentModeling:
    def __init__(self, num_agents, state_dim, action_dim):
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.action_dim = action_dim

        # Model for each opponent
        self.opponent_models = nn.ModuleList([
            nn.Sequential(
                nn.Linear(state_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, action_dim)
            ) for _ in range(num_agents - 1)  # Exclude self
        ])

        self.optimizers = [
            torch.optim.Adam(model.parameters(), lr=1e-3)
            for model in self.opponent_models
        ]

    def predict_opponent_actions(self, agent_id, opponent_states):
        """Predict actions of opponents"""
        predicted_actions = []
        model_idx = 0

        for i in range(self.num_agents):
            if i != agent_id:
                state_tensor = torch.FloatTensor(opponent_states[i])
                action = self.opponent_models[model_idx](state_tensor)
                predicted_actions.append(torch.tanh(action))
                model_idx += 1

        return predicted_actions

    def update_opponent_models(self, agent_id, opponent_states, opponent_actions):
        """Update opponent models based on observed behavior"""
        model_idx = 0
        for i in range(self.num_agents):
            if i != agent_id:
                state_tensor = torch.FloatTensor(opponent_states[i])
                true_action_tensor = torch.FloatTensor(opponent_actions[i])

                predicted_action = self.opponent_models[model_idx](state_tensor)
                loss = nn.MSELoss()(predicted_action, true_action_tensor)

                self.optimizers[model_idx].zero_grad()
                loss.backward()
                self.optimizers[model_idx].step()

                model_idx += 1
```

## Key Takeaways

- Multi-agent physical AI requires handling decentralized decision-making with partial observability
- Communication protocols enable coordination but introduce bandwidth and timing constraints
- Centralized training with decentralized execution (CTDE) allows learning complex coordination
- Competitive scenarios require game-theoretic approaches like Nash Q-learning
- Physical interaction models must handle contact, collision, and force transmission
- Non-stationarity is a fundamental challenge where other agents' policies change during training
- Opponent modeling can help address non-stationarity by predicting other agents' behaviors

## Exercises

1. **Coding**: Implement a simple multi-agent environment with 2 agents that must coordinate to reach a common goal and train using MADDPG.

2. **Theoretical**: Prove that in a zero-sum game, the Nash equilibrium strategy is optimal against any opponent strategy.

3. **Coding**: Design a communication protocol for a multi-robot warehouse task and evaluate the impact of communication bandwidth on performance.

4. **Theoretical**: Explain why the non-stationarity problem makes multi-agent RL fundamentally harder than single-agent RL.

5. **Coding**: Implement opponent modeling in a competitive multi-agent environment and measure its impact on learning stability.

## Further Reading

1. Lowe, R., et al. (2017). "Multi-agent actor-critic for mixed cooperative-competitive environments." *NIPS*.

2. Foerster, J., et al. (2018). "Counterfactual multi-agent policy gradients." *AAAI*.

3. Rashid, T., et al. (2018). "QMIX: Monotonic value function factorisation for deep multi-agent reinforcement learning." *ICML*.