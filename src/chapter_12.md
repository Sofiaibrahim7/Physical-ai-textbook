# Chapter 12: Case Studies and Real-World Applications

Real-world applications of Physical AI span diverse domains, from manufacturing and logistics to healthcare and exploration. These case studies illustrate the practical implementation of Physical AI principles and highlight the challenges and solutions encountered in deploying embodied intelligence in real environments.

## 12.1 Industrial Manipulation and Assembly

Industrial robotics represents one of the most successful applications of Physical AI, where robots perform precise manipulation tasks in structured environments. Modern approaches integrate learning and adaptation capabilities to handle variations and uncertainties.

### 12.1.1 Adaptive Assembly Systems

```python
import torch
import torch.nn as nn
import numpy as np

class AdaptiveAssemblyAgent:
    def __init__(self, state_dim, action_dim, num_parts=5):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_parts = num_parts

        # Main policy network
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()
        )

        # Part recognition network
        self.part_recognizer = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_parts),
            nn.Softmax(dim=-1)
        )

        # Assembly sequence predictor
        self.sequence_predictor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_parts)
        )

        self.optimizer = torch.optim.Adam(
            list(self.policy.parameters()) +
            list(self.part_recognizer.parameters()) +
            list(self.sequence_predictor.parameters()),
            lr=1e-4
        )

    def recognize_part(self, state):
        """Recognize which part is currently being handled"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        part_probs = self.part_recognizer(state_tensor)
        return torch.argmax(part_probs).item(), part_probs.squeeze(0).detach().numpy()

    def predict_assembly_sequence(self, state):
        """Predict optimal assembly sequence"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        sequence_logits = self.sequence_predictor(state_tensor)
        return torch.softmax(sequence_logits, dim=-1).squeeze(0).detach().numpy()

    def select_action(self, state, task_context=None):
        """Select action based on current state and task"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        action = self.policy(state_tensor).squeeze(0).numpy()
        return action

class VisionBasedGraspingSystem:
    def __init__(self, camera_resolution=(640, 480)):
        self.camera_resolution = camera_resolution

        # Vision processing network
        self.vision_net = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        )

        # Grasp prediction network
        self.grasp_predictor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # [x, y, angle, force]
        )

    def process_image(self, image):
        """Process image to extract grasp candidates"""
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
        features = self.vision_net(image_tensor)
        grasp_params = self.grasp_predictor(features)

        # Extract grasp parameters
        grasp_x = torch.sigmoid(grasp_params[0, 0]) * self.camera_resolution[0]
        grasp_y = torch.sigmoid(grasp_params[0, 1]) * self.camera_resolution[1]
        grasp_angle = grasp_params[0, 2]  # in radians
        grasp_force = torch.sigmoid(grasp_params[0, 3]) * 100  # max force

        return {
            'position': (grasp_x.item(), grasp_y.item()),
            'angle': grasp_angle.item(),
            'force': grasp_force.item()
        }

class AssemblyLineOptimizer:
    def __init__(self, num_stations, max_queue_length=10):
        self.num_stations = num_stations
        self.max_queue_length = max_queue_length
        self.station_queues = [[] for _ in range(num_stations)]
        self.station_efficiencies = [1.0] * num_stations  # Learning-based efficiency

        # Station coordination network
        self.coordination_network = nn.Sequential(
            nn.Linear(num_stations * 2, 256),  # [queue_length, efficiency] for each station
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, num_stations),  # Priority for each station
            nn.Softmax(dim=-1)
        )

    def get_station_priority(self):
        """Get priority for each station based on current state"""
        state_features = []
        for i in range(self.num_stations):
            queue_length = len(self.station_queues[i]) / self.max_queue_length
            efficiency = self.station_efficiencies[i]
            state_features.extend([queue_length, efficiency])

        state_tensor = torch.FloatTensor(state_features).unsqueeze(0)
        priorities = self.coordination_network(state_tensor).squeeze(0).detach().numpy()
        return priorities

    def coordinate_assembly_flow(self):
        """Coordinate assembly flow between stations"""
        priorities = self.get_station_priority()
        next_station = np.argmax(priorities)
        return next_station
```

## 12.2 Autonomous Mobile Robots

Mobile robots operating in human environments require sophisticated perception, navigation, and interaction capabilities. These systems must handle dynamic obstacles, social norms, and safety requirements.

### 12.2.1 Socially-Aware Navigation

```python
class SocialNavigationSystem:
    def __init__(self, robot_radius=0.3, social_distance=1.0):
        self.robot_radius = robot_radius
        self.social_distance = social_distance

        # Human behavior prediction network
        self.human_predictor = nn.Sequential(
            nn.Linear(8, 128),  # [pos_robot, vel_robot, pos_human, vel_human]
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 4)   # [pos_x, pos_y, vel_x, vel_y] for human
        )

        # Social force model parameters
        self.lambda_social = 2.0
        self.sigma = 0.8

    def social_force(self, robot_pos, human_pos, human_vel):
        """Compute social force from human to robot"""
        # Distance vector
        diff = robot_pos - human_pos
        distance = np.linalg.norm(diff)

        if distance < 0.1:  # Avoid division by zero
            return np.zeros(2)

        # Direction from human to robot
        direction = diff / distance

        # Social force magnitude (exponentially decreases with distance)
        force_magnitude = self.lambda_social * np.exp((self.robot_radius + self.social_distance - distance) / self.sigma)

        return force_magnitude * direction

    def predict_human_trajectory(self, robot_state, human_state):
        """Predict human movement for navigation planning"""
        state_tensor = torch.FloatTensor(
            np.concatenate([robot_state, human_state])
        ).unsqueeze(0)

        prediction = self.human_predictor(state_tensor).squeeze(0).detach().numpy()
        return prediction

class DynamicPathPlanner:
    def __init__(self, grid_resolution=0.1, map_size=(20, 20)):
        self.grid_resolution = grid_resolution
        self.map_size = map_size
        self.grid = np.zeros((int(map_size[0]/grid_resolution),
                             int(map_size[1]/grid_resolution)))

        # Neural network for dynamic obstacle prediction
        self.obstacle_predictor = nn.Sequential(
            nn.Linear(4, 64),  # [current_pos, target_pos]
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 2)   # [dx, dy] for next position
        )

    def plan_path(self, start, goal, dynamic_obstacles):
        """Plan path considering dynamic obstacles"""
        # Convert to grid coordinates
        start_grid = self.to_grid_coords(start)
        goal_grid = self.to_grid_coords(goal)

        # Predict obstacle positions in future
        future_obstacles = self.predict_dynamic_obstacles(dynamic_obstacles)

        # Update grid with obstacles
        self.update_grid_with_obstacles(future_obstacles)

        # A* path planning with dynamic obstacle consideration
        path = self.a_star_with_dynamic_obstacles(start_grid, goal_grid)

        return self.grid_to_world_coords(path)

    def predict_dynamic_obstacles(self, obstacles):
        """Predict future positions of dynamic obstacles"""
        future_obstacles = []
        for obs in obstacles:
            pos = obs['position']
            vel = obs['velocity']
            # Predict 2 seconds ahead
            future_pos = pos + 2.0 * vel
            future_obstacles.append({
                'position': future_pos,
                'radius': obs['radius']
            })
        return future_obstacles

    def a_star_with_dynamic_obstacles(self, start, goal):
        """A* algorithm adapted for dynamic environments"""
        # Implementation of A* with dynamic obstacle consideration
        # For brevity, simplified version
        path = [start]
        current = start

        while np.linalg.norm(current - goal) > 1:  # Grid resolution threshold
            # Find next best grid cell
            neighbors = self.get_neighbors(current)
            best_neighbor = min(neighbors,
                              key=lambda n: self.heuristic(n, goal))
            path.append(best_neighbor)
            current = best_neighbor

            if len(path) > 1000:  # Prevent infinite loops
                break

        return path

    def to_grid_coords(self, world_pos):
        """Convert world coordinates to grid coordinates"""
        grid_x = int((world_pos[0] + self.map_size[0]/2) / self.grid_resolution)
        grid_y = int((world_pos[1] + self.map_size[1]/2) / self.grid_resolution)
        return np.array([grid_x, grid_y])

    def grid_to_world_coords(self, grid_path):
        """Convert grid path to world coordinates"""
        world_path = []
        for grid_pos in grid_path:
            world_x = grid_pos[0] * self.grid_resolution - self.map_size[0]/2
            world_y = grid_pos[1] * self.grid_resolution - self.map_size[1]/2
            world_path.append([world_x, world_y])
        return np.array(world_path)
```

## 12.3 Healthcare and Assistive Robotics

Healthcare applications of Physical AI require special attention to safety, privacy, and human-robot interaction. These systems must operate reliably in sensitive environments while providing meaningful assistance.

### 12.3.1 Rehabilitation Robotics

```python
class RehabilitationRobot:
    def __init__(self, patient_state_dim=10, exercise_types=5):
        self.patient_state_dim = patient_state_dim
        self.exercise_types = exercise_types

        # Patient state estimator
        self.state_estimator = nn.Sequential(
            nn.Linear(20, 128),  # Sensor inputs: position, force, EMG, etc.
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, patient_state_dim)
        )

        # Exercise difficulty adapter
        self.difficulty_adapter = nn.Sequential(
            nn.Linear(patient_state_dim + exercise_types, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3)  # [force_level, speed, range]
        )

        # Safety monitor
        self.safety_monitor = nn.Sequential(
            nn.Linear(patient_state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Safety probability
        )

    def assess_patient_state(self, sensor_data):
        """Assess current patient state from sensor data"""
        sensor_tensor = torch.FloatTensor(sensor_data).unsqueeze(0)
        state = self.state_estimator(sensor_tensor).squeeze(0).detach().numpy()
        return state

    def adapt_exercise(self, patient_state, exercise_type):
        """Adapt exercise parameters based on patient state"""
        exercise_onehot = np.zeros(self.exercise_types)
        exercise_onehot[exercise_type] = 1

        state_tensor = torch.FloatTensor(
            np.concatenate([patient_state, exercise_onehot])
        ).unsqueeze(0)

        adaptation_params = self.difficulty_adapter(state_tensor).squeeze(0).detach().numpy()

        return {
            'force_level': adaptation_params[0],
            'movement_speed': adaptation_params[1],
            'range_of_motion': adaptation_params[2]
        }

    def monitor_safety(self, patient_state):
        """Monitor safety during exercise"""
        state_tensor = torch.FloatTensor(patient_state).unsqueeze(0)
        safety_prob = self.safety_monitor(state_tensor).item()
        return safety_prob > 0.8  # Safe if probability > 80%

class AssistiveFeedingRobot:
    def __init__(self, max_speed=0.1, safety_threshold=0.5):
        self.max_speed = max_speed
        self.safety_threshold = safety_threshold

        # Food recognition and localization
        self.food_detector = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, 4)  # [x, y, z, confidence]
        )

        # Human mouth detection
        self.mouth_detector = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # [x, y, z]
        )

    def locate_food(self, image):
        """Locate food items in the image"""
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
        food_params = self.food_detector(image_tensor).squeeze(0).detach().numpy()

        if food_params[3] > 0.5:  # Confidence threshold
            return food_params[:3]  # [x, y, z]
        else:
            return None

    def locate_mouth(self, face_image):
        """Locate human mouth in face image"""
        image_tensor = torch.FloatTensor(face_image).permute(2, 0, 1).unsqueeze(0) / 255.0
        mouth_pos = self.mouth_detector(image_tensor).squeeze(0).detach().numpy()
        return mouth_pos
```

## 12.4 Exploration and Field Robotics

Exploration robots operate in unstructured, unknown environments where traditional assumptions about the world may not hold. These systems must be capable of autonomous operation, self-repair, and long-term sustainability.

### 12.4.1 Autonomous Exploration

```python
class ExplorationAgent:
    def __init__(self, map_size=(100, 100), resolution=1.0):
        self.map_size = map_size
        self.resolution = resolution
        self.occupancy_map = np.zeros(map_size)  # 0: unknown, 1: free, -1: occupied
        self.visited_map = np.zeros(map_size)
        self.frontier_map = np.zeros(map_size)

        # Exploration policy network
        self.exploration_policy = nn.Sequential(
            nn.Linear(4, 128),  # [robot_x, robot_y, target_x, target_y]
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 2),  # [dx, dy] relative to current position
            nn.Tanh()
        )

        # Frontier detection network
        self.frontier_detector = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def update_map(self, sensor_data):
        """Update occupancy map with new sensor data"""
        # Process LIDAR or other range sensor data
        for beam in sensor_data:
            start_pos = beam['start']
            end_pos = beam['end']
            obstacle = beam['obstacle']

            # Bresenham's line algorithm to update ray
            self.ray_trace(start_pos, end_pos, obstacle)

    def ray_trace(self, start, end, obstacle):
        """Update map along ray from start to end"""
        # Simplified ray tracing
        steps = int(np.linalg.norm(end - start) / self.resolution)
        for i in range(steps):
            t = i / steps
            pos = start + t * (end - start)
            grid_x = int(pos[0] / self.resolution)
            grid_y = int(pos[1] / self.resolution)

            if 0 <= grid_x < self.map_size[0] and 0 <= grid_y < self.map_size[1]:
                if i == steps - 1 and obstacle:  # End point is obstacle
                    self.occupancy_map[grid_x, grid_y] = -1
                else:  # Free space along ray
                    self.occupancy_map[grid_x, grid_y] = 1

    def detect_frontiers(self):
        """Detect frontiers (unknown boundaries) in the map"""
        # Find boundaries between known and unknown areas
        frontiers = []

        for x in range(1, self.map_size[0]-1):
            for y in range(1, self.map_size[1]-1):
                if self.occupancy_map[x, y] == 0:  # Unknown
                    # Check if adjacent to known area
                    adjacent_known = False
                    for dx, dy in [(-1,0), (1,0), (0,-1), (0,1)]:
                        nx, ny = x + dx, y + dy
                        if (0 <= nx < self.map_size[0] and
                            0 <= ny < self.map_size[1] and
                            self.occupancy_map[nx, ny] != 0):
                            adjacent_known = True
                            break

                    if adjacent_known:
                        frontiers.append((x * self.resolution, y * self.resolution))

        return frontiers

    def select_exploration_target(self, robot_pos):
        """Select next exploration target"""
        frontiers = self.detect_frontiers()

        if not frontiers:
            # No frontiers, explore unknown areas
            unknown_areas = np.where(self.occupancy_map == 0)
            if len(unknown_areas[0]) > 0:
                idx = np.random.choice(len(unknown_areas[0]))
                target = (unknown_areas[0][idx] * self.resolution,
                         unknown_areas[1][idx] * self.resolution)
            else:
                # Map is fully explored
                return robot_pos
        else:
            # Select closest frontier
            frontiers = np.array(frontiers)
            distances = np.linalg.norm(frontiers - robot_pos, axis=1)
            closest_idx = np.argmin(distances)
            target = frontiers[closest_idx]

        return target

    def get_exploration_action(self, robot_pos, target_pos):
        """Get action for exploration towards target"""
        state_tensor = torch.FloatTensor(
            np.concatenate([robot_pos, target_pos])
        ).unsqueeze(0)

        action = self.exploration_policy(state_tensor).squeeze(0).detach().numpy()
        # Convert to world coordinates
        action_world = action * self.max_speed  # Scale by max speed
        return action_world

class RobustNavigationSystem:
    def __init__(self, safety_radius=1.0, max_retries=3):
        self.safety_radius = safety_radius
        self.max_retries = max_retries
        self.retry_count = 0

        # Terrain classification network
        self.terrain_classifier = nn.Sequential(
            nn.Conv2d(1, 16, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, 5, padding=2),
            nn.Sigmoid()
        )

        # Traversability predictor
        self.traversability_predictor = nn.Sequential(
            nn.Linear(5, 64),  # [slope, roughness, obstacles, etc.]
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Traversability score
        )

    def assess_traversability(self, terrain_data):
        """Assess terrain traversability"""
        # Process terrain features
        features = np.array([
            terrain_data['slope'],
            terrain_data['roughness'],
            terrain_data['obstacle_density'],
            terrain_data['surface_type'],
            terrain_data['traction_coefficient']
        ])

        features_tensor = torch.FloatTensor(features).unsqueeze(0)
        traversability = self.traversability_predictor(features_tensor).item()
        return traversability

    def plan_robust_path(self, start, goal, terrain_map):
        """Plan path with robustness to terrain variations"""
        # Multi-criteria path planning
        traversable_path = self.find_traversable_path(start, goal, terrain_map)

        if traversable_path is None:
            # Fallback: find safest path even if not optimal
            self.retry_count += 1
            if self.retry_count < self.max_retries:
                return self.plan_robust_path(start, goal, terrain_map)
            else:
                # Return to safe location
                return self.return_to_safe_location(start)

        return traversable_path

    def find_traversable_path(self, start, goal, terrain_map):
        """Find path considering terrain traversability"""
        # Implementation would use D* or similar algorithm
        # Simplified version for demonstration
        path = [start]
        current = np.array(start)
        goal_array = np.array(goal)

        while np.linalg.norm(current - goal_array) > 0.5:
            # Move towards goal while considering terrain
            direction = (goal_array - current) / np.linalg.norm(goal_array - current)
            next_pos = current + 0.5 * direction  # 0.5m steps

            # Check terrain traversability
            terrain_idx = (int(next_pos[0]), int(next_pos[1]))
            if (0 <= terrain_idx[0] < terrain_map.shape[0] and
                0 <= terrain_idx[1] < terrain_map.shape[1]):
                traversability = terrain_map[terrain_idx]
                if traversability > 0.3:  # Acceptable traversability
                    path.append(next_pos.copy())
                    current = next_pos
                else:
                    # Find alternative route
                    current = self.find_alternative_route(current, terrain_map)
                    if current is None:
                        return None
            else:
                return None

        return path

    def find_alternative_route(self, current_pos, terrain_map):
        """Find alternative route around obstacles"""
        # Spiral search for traversable area
        for radius in range(1, 10):
            for angle in np.linspace(0, 2*np.pi, 8):
                dx = radius * np.cos(angle)
                dy = radius * np.sin(angle)
                new_pos = current_pos + np.array([dx, dy])

                terrain_idx = (int(new_pos[0]), int(new_pos[1]))
                if (0 <= terrain_idx[0] < terrain_map.shape[0] and
                    0 <= terrain_idx[1] < terrain_map.shape[1]):
                    traversability = terrain_map[terrain_idx]
                    if traversability > 0.3:
                        return new_pos

        return None
```

## 12.5 Agricultural Robotics

Agricultural applications of Physical AI address challenges in precision farming, crop monitoring, and automated harvesting. These systems must operate in outdoor environments with variable conditions.

### 12.5.1 Precision Agriculture Systems

```python
class AgriculturalMonitoringSystem:
    def __init__(self, field_size=(1000, 1000), crop_types=10):
        self.field_size = field_size
        self.crop_types = crop_types
        self.crop_health_map = np.zeros((*field_size, 3))  # [health, moisture, nutrients]

        # Crop health assessment network
        self.health_assessor = nn.Sequential(
            nn.Conv2d(4, 32, 5, padding=2),  # RGB + NIR
            nn.ReLU(),
            nn.Conv2d(32, 64, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 5, padding=2),
            nn.ReLU(),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        # Yield prediction network
        self.yield_predictor = nn.Sequential(
            nn.Linear(5, 128),  # [health, moisture, nutrients, weather, time]
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.ReLU()
        )

    def assess_crop_health(self, multispectral_image):
        """Assess crop health from multispectral imagery"""
        # Convert to tensor and normalize
        image_tensor = torch.FloatTensor(multispectral_image).permute(2, 0, 1).unsqueeze(0)
        health_map = self.health_assessor(image_tensor).squeeze(0).squeeze(0).detach().numpy()
        return health_map

    def predict_yield(self, field_conditions):
        """Predict crop yield based on field conditions"""
        conditions_tensor = torch.FloatTensor(field_conditions).unsqueeze(0)
        predicted_yield = self.yield_predictor(conditions_tensor).item()
        return predicted_yield

class AutonomousHarvestingRobot:
    def __init__(self, max_speed=1.0, detection_range=5.0):
        self.max_speed = max_speed
        self.detection_range = detection_range

        # Crop detection and classification
        self.crop_detector = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Linear(256, 3)  # [x, y, confidence]
        )

        # Ripeness assessment
        self.ripeness_assessor = nn.Sequential(
            nn.Linear(4, 128),  # [color, size, texture, shape]
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()  # Ripeness score
        )

    def detect_crops(self, image):
        """Detect crops in the field of view"""
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0
        crop_params = self.crop_detector(image_tensor).squeeze(0).detach().numpy()

        if crop_params[2] > 0.7:  # High confidence detection
            return crop_params[:2]  # [x, y] in image coordinates
        else:
            return None

    def assess_ripeness(self, crop_features):
        """Assess ripeness of detected crop"""
        features_tensor = torch.FloatTensor(crop_features).unsqueeze(0)
        ripeness_score = self.ripeness_assessor(features_tensor).item()
        return ripeness_score

    def harvest_decision(self, crop_ripeness, crop_position):
        """Decide whether and how to harvest crop"""
        if crop_ripeness > 0.8:  # Sufficiently ripe
            # Calculate approach trajectory
            approach_vector = crop_position - self.current_position
            approach_distance = np.linalg.norm(approach_vector)

            if approach_distance < self.detection_range:
                return {
                    'harvest': True,
                    'approach_vector': approach_vector / approach_distance,
                    'harvest_method': 'grasp' if crop_ripeness > 0.9 else 'cut'
                }

        return {'harvest': False}
```

## Key Takeaways

- Industrial applications focus on precision, reliability, and efficiency in structured environments
- Mobile robots must handle dynamic environments and social interactions
- Healthcare robotics requires special attention to safety, privacy, and human factors
- Exploration robots need autonomy, robustness, and long-term sustainability
- Agricultural robotics addresses precision farming and environmental challenges
- Real-world deployment requires integration of multiple Physical AI capabilities
- Safety and reliability are paramount across all applications

## Exercises

1. **Coding**: Implement a simplified version of the rehabilitation robot's difficulty adaptation system and test it with simulated patient data.

2. **Theoretical**: Analyze the trade-offs between exploration and exploitation in autonomous exploration systems.

3. **Coding**: Design a socially-aware navigation system for a mobile robot operating in a crowded environment.

4. **Theoretical**: Discuss the challenges of deploying Physical AI systems in unstructured outdoor environments.

5. **Coding**: Implement a crop health monitoring system using the provided architecture and evaluate its performance.

## Further Reading

1. Siciliano, B., & Khatib, O. (2016). "Springer Handbook of Robotics." Springer.

2. Murphy, R. R. (2019). "Introduction to AI Robotics." MIT Press.

3. Srinivasa, S. S., et al. (2019). "The GRiD dataset: A large dataset of real-world robotic manipulation." *IROS*.