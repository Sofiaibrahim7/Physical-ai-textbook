# Chapter 4: Sensors and Perception

Sensors form the interface between the physical world and AI systems, enabling perception and understanding of the environment. This chapter covers the fundamental principles of sensing, sensor fusion, and perception algorithms essential for Physical AI systems.

## 4.1 Introduction to Physical AI Sensors

Sensors in Physical AI systems serve as the primary means of acquiring information about the environment, the system's own state, and interactions with the world. Unlike digital systems that process abstract data, Physical AI systems must interpret signals from the real world, requiring sophisticated approaches to sensing and perception.

Key sensor categories include:

1. **Proprioceptive sensors**: Measure internal state (joint angles, motor currents)
2. **Exteroceptive sensors**: Measure external environment (cameras, LIDAR, touch)
3. **Interoceptive sensors**: Measure system health (temperature, battery level)

![Figure 4.1: Sensor Categories in Physical AI](placeholder)

## 4.2 Proprioceptive Sensing

Proprioceptive sensors provide information about the system's own state, including position, velocity, and internal forces.

### 4.2.1 Encoders and Position Sensing

Encoders are critical for measuring joint positions in robotic systems:

```python
import numpy as np
import matplotlib.pyplot as plt

class Encoder:
    def __init__(self, resolution=4096, gear_ratio=1.0):
        self.resolution = resolution  # Counts per revolution
        self.gear_ratio = gear_ratio  # Motor revolutions to output shaft
        self.raw_count = 0
        self.position = 0.0  # Position in radians
        self.velocity = 0.0  # Velocity in rad/s
        self.prev_position = 0.0
        self.prev_time = 0.0

    def update(self, raw_count, timestamp):
        """Update encoder reading and compute position/velocity"""
        # Calculate position change
        delta_count = raw_count - self.raw_count
        self.raw_count = raw_count

        # Convert to position in radians
        mechanical_revs = delta_count / self.resolution
        output_shaft_revs = mechanical_revs / self.gear_ratio
        delta_position = output_shaft_revs * 2 * np.pi

        self.position += delta_position

        # Compute velocity if we have timing information
        if self.prev_time > 0:
            dt = timestamp - self.prev_time
            if dt > 0:
                self.velocity = delta_position / dt

        self.prev_time = timestamp
        self.prev_position = self.position

        return self.position, self.velocity

class JointStateEstimator:
    def __init__(self, num_joints):
        self.num_joints = num_joints
        self.encoders = [Encoder() for _ in range(num_joints)]
        self.positions = np.zeros(num_joints)
        self.velocities = np.zeros(num_joints)

    def update_sensors(self, encoder_counts, timestamp):
        """Update all joint states from encoder readings"""
        for i, count in enumerate(encoder_counts):
            pos, vel = self.encoders[i].update(count, timestamp)
            self.positions[i] = pos
            self.velocities[i] = vel

        return self.positions.copy(), self.velocities.copy()
```

### 4.2.2 Inertial Measurement Units (IMUs)

IMUs provide orientation, angular velocity, and linear acceleration information:

```python
class IMU:
    def __init__(self, accelerometer_noise=0.01, gyroscope_noise=0.001, magnetometer_noise=0.01):
        self.accelerometer_noise = accelerometer_noise
        self.gyroscope_noise = gyroscope_noise
        self.magnetometer_noise = magnetometer_noise

        # Internal state for orientation estimation
        self.orientation = np.array([1, 0, 0, 0])  # Quaternion [w, x, y, z]
        self.angular_velocity = np.zeros(3)  # [wx, wy, wz]
        self.linear_acceleration = np.zeros(3)  # [ax, ay, az]
        self.magnetic_field = np.array([0.2, 0, 0.4])  # Earth's magnetic field (approx)

    def read_sensors(self, true_angular_velocity, true_linear_acceleration, dt=0.01):
        """Simulate IMU readings with noise"""
        # Add noise to true values
        noisy_angular_velocity = true_angular_velocity + \
                                np.random.normal(0, self.gyroscope_noise, 3)
        noisy_linear_acceleration = true_linear_acceleration + \
                                   np.random.normal(0, self.accelerometer_noise, 3)

        # Update orientation using gyroscope integration
        self.update_orientation(noisy_angular_velocity, dt)

        self.angular_velocity = noisy_angular_velocity
        self.linear_acceleration = noisy_linear_acceleration

        return {
            'angular_velocity': noisy_angular_velocity,
            'linear_acceleration': noisy_linear_acceleration,
            'orientation': self.orientation.copy(),
            'magnetic_field': self.magnetic_field + np.random.normal(0, self.magnetometer_noise, 3)
        }

    def update_orientation(self, angular_velocity, dt):
        """Update orientation quaternion using gyroscope data"""
        # Convert angular velocity to quaternion derivative
        omega_quat = np.array([0, *angular_velocity])  # [0, wx, wy, wz]
        quat_derivative = self.quaternion_multiply(omega_quat, self.orientation) * 0.5

        # Integrate
        new_orientation = self.orientation + quat_derivative * dt

        # Normalize quaternion
        self.orientation = new_orientation / np.linalg.norm(new_orientation)

    def quaternion_multiply(self, q1, q2):
        """Multiply two quaternions"""
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return np.array([w, x, y, z])

    def get_euler_angles(self):
        """Convert quaternion to Euler angles (roll, pitch, yaw)"""
        w, x, y, z = self.orientation

        # Roll (x-axis rotation)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = np.arctan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2 * (w * y - z * x)
        if abs(sinp) >= 1:
            pitch = np.pi / 2 if sinp > 0 else -np.pi / 2
        else:
            pitch = np.arcsin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = np.arctan2(siny_cosp, cosy_cosp)

        return np.array([roll, pitch, yaw])
```

### 4.2.3 Force and Torque Sensing

Force and torque sensors enable precise interaction with the environment:

```python
class ForceTorqueSensor:
    def __init__(self, force_range=1000, torque_range=100, noise_level=0.01):
        self.force_range = force_range
        self.torque_range = torque_range
        self.noise_level = noise_level
        self.bias = np.random.normal(0, noise_level*0.1, 6)  # 3 forces + 3 torques

    def measure(self, true_force, true_torque):
        """Measure force and torque with noise and bias"""
        true_measurement = np.concatenate([true_force, true_torque])

        # Add noise and bias
        noisy_measurement = true_measurement + self.bias + \
                           np.random.normal(0, self.noise_level, 6)

        # Apply sensor limits
        force = np.clip(noisy_measurement[:3], -self.force_range, self.force_range)
        torque = np.clip(noisy_measurement[3:], -self.torque_range, self.torque_range)

        return force, torque

class ContactDetector:
    def __init__(self, force_threshold=5.0, torque_threshold=1.0):
        self.force_threshold = force_threshold
        self.torque_threshold = torque_threshold
        self.contact_history = []

    def detect_contact(self, force, torque):
        """Detect contact based on force/torque thresholds"""
        force_magnitude = np.linalg.norm(force)
        torque_magnitude = np.linalg.norm(torque)

        contact = (force_magnitude > self.force_threshold or
                  torque_magnitude > self.torque_threshold)

        self.contact_history.append(contact)
        return contact

    def estimate_contact_location(self, force, torque, sensor_position):
        """Estimate contact location using force and torque measurements"""
        # For a simple case: if we know the contact force direction,
        # we can estimate the contact location
        if np.linalg.norm(force) > self.force_threshold:
            # Simplified: assume contact point is along force direction
            force_direction = force / np.linalg.norm(force)
            moment_arm = np.cross(force_direction, torque) / np.linalg.norm(force)
            contact_location = sensor_position + moment_arm
            return contact_location
        return None
```

## 4.3 Exteroceptive Sensing

Exteroceptive sensors provide information about the external environment.

### 4.3.1 Camera Systems and Computer Vision

Camera systems are essential for visual perception in Physical AI:

```python
import cv2
import torch
import torch.nn as nn

class CameraSystem:
    def __init__(self, width=640, height=480, fov=60, focal_length=None):
        self.width = width
        self.height = height
        self.fov = np.radians(fov)  # Field of view in radians

        if focal_length is None:
            # Calculate focal length from FOV and image dimensions
            self.focal_length = (self.width / 2) / np.tan(self.fov / 2)
        else:
            self.focal_length = focal_length

        self.intrinsic_matrix = np.array([
            [self.focal_length, 0, self.width / 2],
            [0, self.focal_length, self.height / 2],
            [0, 0, 1]
        ])

    def project_3d_to_2d(self, points_3d):
        """Project 3D points to 2D image coordinates"""
        # Convert to homogeneous coordinates
        points_h = np.column_stack([points_3d, np.ones(len(points_3d))])

        # Apply intrinsic matrix
        points_2d_h = (self.intrinsic_matrix @ points_h.T).T

        # Convert from homogeneous to Cartesian
        points_2d = points_2d_h[:, :2] / points_2d_h[:, 2:3]

        return points_2d

    def undistort_image(self, image, distortion_coeffs):
        """Remove lens distortion from image"""
        h, w = image.shape[:2]
        new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
            self.intrinsic_matrix, distortion_coeffs, (w, h), 1, (w, h)
        )

        undistorted = cv2.undistort(image, self.intrinsic_matrix, distortion_coeffs, None, new_camera_matrix)
        return undistorted

class ObjectDetector:
    def __init__(self):
        # Simple CNN-based object detector
        self.detector = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(576, 512),
            nn.ReLU(),
            nn.Linear(512, 4)  # [x, y, width, height] for bounding box
        )

    def detect_object(self, image):
        """Detect object in image and return bounding box"""
        # Preprocess image
        image_tensor = torch.FloatTensor(image).permute(2, 0, 1).unsqueeze(0) / 255.0

        # Run detection
        bbox = self.detector(image_tensor).squeeze(0).detach().numpy()

        # Convert to proper format [x, y, width, height]
        return bbox

class DepthEstimator:
    def __init__(self, baseline=0.1, focal_length=500):
        self.baseline = baseline  # Distance between stereo cameras
        self.focal_length = focal_length

    def estimate_depth_stereo(self, left_image, right_image):
        """Estimate depth using stereo vision"""
        # Convert to grayscale
        left_gray = cv2.cvtColor(left_image, cv2.COLOR_RGB2GRAY)
        right_gray = cv2.cvtColor(right_image, cv2.COLOR_RGB2GRAY)

        # Compute disparity using OpenCV
        stereo = cv2.StereoSGBM_create(
            minDisparity=0,
            numDisparities=64,
            blockSize=15,
            P1=8 * 3 * 15**2,
            P2=32 * 3 * 15**2
        )

        disparity = stereo.compute(left_gray, right_gray).astype(np.float32) / 16.0

        # Convert disparity to depth
        depth = (self.baseline * self.focal_length) / (disparity + 1e-6)
        depth[disparity == 0] = 0  # Set invalid disparities to 0

        return depth
```

### 4.3.2 Range Sensors (LIDAR, Sonar, IR)

Range sensors provide accurate distance measurements:

```python
class LIDARSensor:
    def __init__(self, num_beams=360, max_range=10.0, fov=360):
        self.num_beams = num_beams
        self.max_range = max_range
        self.fov = np.radians(fov)
        self.angles = np.linspace(0, self.fov, num_beams, endpoint=False)

    def scan(self, environment_map, robot_position, robot_orientation=0):
        """Perform LIDAR scan in environment"""
        ranges = np.full(self.num_beams, self.max_range)

        for i, angle in enumerate(self.angles):
            # Calculate ray direction
            world_angle = angle + robot_orientation
            ray_direction = np.array([np.cos(world_angle), np.sin(world_angle)])

            # Cast ray to find intersection
            for distance in np.arange(0.1, self.max_range, 0.1):
                ray_endpoint = robot_position + distance * ray_direction
                x, y = int(ray_endpoint[0]), int(ray_endpoint[1])

                # Check if ray hits an obstacle (simplified)
                if self.check_collision(environment_map, x, y):
                    ranges[i] = distance
                    break

        return ranges

    def check_collision(self, env_map, x, y):
        """Check if point collides with obstacle in map"""
        # Simplified collision check
        if 0 <= x < env_map.shape[0] and 0 <= y < env_map.shape[1]:
            return env_map[x, y] > 0.5  # Assuming obstacles are marked > 0.5
        return False

    def build_map(self, scan_data, robot_position, resolution=0.1):
        """Build occupancy grid from LIDAR data"""
        # Create local map around robot
        map_size = int(20 / resolution)  # 20m x 20m map
        occupancy_map = np.zeros((map_size, map_size))
        map_center = map_size // 2

        for i, range_reading in enumerate(scan_data):
            if range_reading < self.max_range:
                angle = self.angles[i]
                hit_point = robot_position + range_reading * np.array([np.cos(angle), np.sin(angle)])

                # Convert to map coordinates
                map_x = int((hit_point[0] - robot_position[0]) / resolution) + map_center
                map_y = int((hit_point[1] - robot_position[1]) / resolution) + map_center

                if 0 <= map_x < map_size and 0 <= map_y < map_size:
                    occupancy_map[map_x, map_y] = 1.0  # Mark as occupied

        return occupancy_map

class SonarSensor:
    def __init__(self, max_range=4.0, beam_width=30, noise_level=0.05):
        self.max_range = max_range
        self.beam_width = np.radians(beam_width)
        self.noise_level = noise_level

    def measure(self, environment, sensor_position, target_direction):
        """Measure distance using sonar"""
        # Simulate sonar beam
        for distance in np.arange(0.1, self.max_range, 0.05):
            test_point = sensor_position + distance * target_direction

            # Check for obstacles in beam cone
            if self.detect_obstacle_in_beam(environment, test_point, target_direction):
                # Add noise to measurement
                noisy_distance = distance + np.random.normal(0, self.noise_level)
                return min(noisy_distance, self.max_range)

        # No obstacle detected
        return self.max_range

    def detect_obstacle_in_beam(self, environment, point, direction):
        """Check if there's an obstacle in the sonar beam"""
        # Simplified detection
        x, y = int(point[0]), int(point[1])
        if 0 <= x < environment.shape[0] and 0 <= y < environment.shape[1]:
            return environment[x, y] > 0.5
        return False
```

## 4.4 Sensor Fusion

Sensor fusion combines data from multiple sensors to improve perception accuracy.

### 4.4.1 Kalman Filtering

```python
class KalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim

        # State vector: [x, y, vx, vy] (position and velocity)
        self.state = np.zeros(state_dim)

        # Covariance matrix
        self.P = np.eye(state_dim) * 1000  # Initial uncertainty

        # Process noise
        self.Q = np.eye(state_dim) * 0.1

        # Measurement noise
        self.R = np.eye(measurement_dim) * 1.0

        # State transition model (constant velocity model)
        self.F = np.eye(state_dim)
        self.F[0, 2] = 1.0  # x += vx * dt
        self.F[1, 3] = 1.0  # y += vy * dt

        # Measurement model
        self.H = np.zeros((measurement_dim, state_dim))
        self.H[0, 0] = 1.0  # Measure x position
        self.H[1, 1] = 1.0  # Measure y position

    def predict(self, dt=0.01):
        """Prediction step"""
        # Update state transition matrix for time step
        F = np.eye(self.state_dim)
        F[0, 2] = dt  # x += vx * dt
        F[1, 3] = dt  # y += vy * dt

        # Predict state
        self.state = F @ self.state

        # Predict covariance
        self.P = F @ self.P @ F.T + self.Q

    def update(self, measurement):
        """Update step with measurement"""
        # Innovation
        innovation = measurement - self.H @ self.state

        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R

        # Kalman gain
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ innovation

        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ self.H) @ self.P

class ExtendedKalmanFilter:
    def __init__(self, state_dim, measurement_dim):
        self.state_dim = state_dim
        self.measurement_dim = measurement_dim
        self.state = np.zeros(state_dim)
        self.P = np.eye(state_dim) * 1000
        self.Q = np.eye(state_dim) * 0.1
        self.R = np.eye(measurement_dim) * 1.0

    def predict(self, dt=0.01):
        """Nonlinear prediction step"""
        # Example: constant turn rate model
        x, y, v, psi, omega = self.state  # position, velocity, heading, turn rate

        # Update state (nonlinear model)
        new_state = np.array([
            x + v * np.cos(psi) * dt,
            y + v * np.sin(psi) * dt,
            v,  # constant velocity
            psi + omega * dt,
            omega  # constant turn rate
        ])

        # Linearize around current state
        F = self.jacobian_f(self.state, dt)

        self.state = new_state
        self.P = F @ self.P @ F.T + self.Q

    def jacobian_f(self, state, dt):
        """Jacobian of the motion model"""
        x, y, v, psi, omega = state

        F = np.eye(self.state_dim)
        F[0, 2] = np.cos(psi) * dt  # dx/dv
        F[0, 3] = -v * np.sin(psi) * dt  # dx/dpsi
        F[1, 2] = np.sin(psi) * dt  # dy/dv
        F[1, 3] = v * np.cos(psi) * dt  # dy/dpsi
        F[3, 4] = dt  # dpsi/domega

        return F

    def update(self, measurement):
        """Nonlinear update step"""
        # Measurement function (bearing and range)
        x, y, v, psi, omega = self.state
        meas_x, meas_y = measurement

        # Predicted measurement
        predicted_meas = np.array([
            np.sqrt((meas_x - x)**2 + (meas_y - y)**2),  # Range
            np.arctan2(meas_y - y, meas_x - x)  # Bearing
        ])

        # Innovation
        innovation = measurement - predicted_meas

        # Linearize measurement function
        H = self.jacobian_h(self.state, measurement)

        # Innovation covariance
        S = H @ self.P @ H.T + self.R

        # Kalman gain
        K = self.P @ H.T @ np.linalg.inv(S)

        # Update state
        self.state = self.state + K @ innovation

        # Update covariance
        I = np.eye(self.state_dim)
        self.P = (I - K @ H) @ self.P

    def jacobian_h(self, state, measurement):
        """Jacobian of the measurement function"""
        x, y, v, psi, omega = state
        meas_x, meas_y = measurement

        range_sq = (meas_x - x)**2 + (meas_y - y)**2
        range_val = np.sqrt(range_sq)

        H = np.zeros((self.measurement_dim, self.state_dim))
        H[0, 0] = (x - meas_x) / range_val  # drange/dx
        H[0, 1] = (y - meas_y) / range_val  # drange/dy
        H[1, 0] = (meas_y - y) / range_sq   # dbearing/dx
        H[1, 1] = (meas_x - x) / range_sq   # dbearing/dy

        return H
```

### 4.4.2 Particle Filtering

```python
class ParticleFilter:
    def __init__(self, state_dim, num_particles=1000):
        self.state_dim = state_dim
        self.num_particles = num_particles
        self.particles = np.random.randn(num_particles, state_dim)
        self.weights = np.ones(num_particles) / num_particles

    def predict(self, control_input, process_noise=0.1):
        """Predict step - propagate particles forward"""
        # Apply motion model with noise
        for i in range(self.num_particles):
            # Simple motion model: add control + noise
            self.particles[i] += control_input + np.random.normal(0, process_noise, self.state_dim)

            # Apply constraints if needed (e.g., keep particles in valid region)
            self.particles[i] = np.clip(self.particles[i], -10, 10)  # Example bounds

    def update(self, measurement, measurement_function, measurement_noise=0.1):
        """Update step - reweight particles based on measurement"""
        for i in range(self.num_particles):
            # Predict what measurement this particle would generate
            predicted_measurement = measurement_function(self.particles[i])

            # Calculate likelihood of actual measurement given particle
            diff = measurement - predicted_measurement
            likelihood = np.exp(-0.5 * np.sum((diff)**2) / (measurement_noise**2))

            # Update weight
            self.weights[i] *= likelihood

        # Normalize weights
        self.weights = self.weights / np.sum(self.weights)

    def resample(self):
        """Resample particles based on weights"""
        # Systematic resampling
        indices = []
        cumulative_sum = np.cumsum(self.weights)
        U = np.random.uniform(0, 1/self.num_particles)

        i = 0
        for j in range(self.num_particles):
            while U > cumulative_sum[i]:
                i += 1
            indices.append(i)
            U += 1/self.num_particles

        # Resample particles
        self.particles = self.particles[indices]
        self.weights = np.ones(self.num_particles) / self.num_particles

    def estimate_state(self):
        """Estimate state as weighted average of particles"""
        return np.average(self.particles, axis=0, weights=self.weights)

    def get_state_distribution(self):
        """Return mean and covariance of particle distribution"""
        mean = self.estimate_state()
        # Weighted covariance
        diff = self.particles - mean
        cov = np.zeros((self.state_dim, self.state_dim))
        for i in range(self.num_particles):
            cov += self.weights[i] * np.outer(diff[i], diff[i])
        return mean, cov
```

## 4.5 Perception Algorithms

### 4.5.1 SLAM (Simultaneous Localization and Mapping)

```python
class SimpleSLAM:
    def __init__(self, initial_pose=np.zeros(3)):
        self.robot_pose = initial_pose  # [x, y, theta]
        self.landmarks = {}  # Dictionary of landmark positions
        self.landmark_observations = []  # History of observations

        # Covariance matrix for pose uncertainty
        self.pose_covariance = np.eye(3) * 0.1

    def motion_update(self, control, motion_noise=np.diag([0.1, 0.1, 0.05])):
        """Update robot pose based on motion"""
        # Extract control input [v, omega, dt] or [dx, dy, dtheta]
        if len(control) == 3:
            # Differential drive model
            v, omega, dt = control
            if abs(omega) < 1e-6:  # Straight line
                dx = v * dt * np.cos(self.robot_pose[2])
                dy = v * dt * np.sin(self.robot_pose[2])
                dtheta = 0
            else:  # Circular motion
                radius = v / omega
                dx = radius * (np.sin(self.robot_pose[2] + omega*dt) - np.sin(self.robot_pose[2]))
                dy = radius * (np.cos(self.robot_pose[2]) - np.cos(self.robot_pose[2] + omega*dt))
                dtheta = omega * dt
        else:
            # Direct displacement
            dx, dy, dtheta = control

        # Update pose
        self.robot_pose[0] += dx
        self.robot_pose[1] += dy
        self.robot_pose[2] += dtheta
        self.robot_pose[2] = np.arctan2(np.sin(self.robot_pose[2]), np.cos(self.robot_pose[2]))  # Normalize angle

        # Update covariance (simplified)
        G = np.eye(3)
        G[0, 2] = -dx * np.sin(self.robot_pose[2]) - dy * np.cos(self.robot_pose[2])
        G[1, 2] = dx * np.cos(self.robot_pose[2]) - dy * np.sin(self.robot_pose[2])

        self.pose_covariance = G @ self.pose_covariance @ G.T + motion_noise

    def measurement_update(self, landmark_id, measurement, sensor_noise=np.diag([0.1, 0.05])):
        """Update based on landmark measurement [range, bearing]"""
        if landmark_id not in self.landmarks:
            # Initialize landmark position
            range_meas, bearing_meas = measurement
            x_robot, y_robot, theta_robot = self.robot_pose

            # Calculate landmark position in global frame
            x_landmark = x_robot + range_meas * np.cos(theta_robot + bearing_meas)
            y_landmark = y_robot + range_meas * np.sin(theta_robot + bearing_meas)

            self.landmarks[landmark_id] = np.array([x_landmark, y_landmark])
        else:
            # Update existing landmark
            x_robot, y_robot, theta_robot = self.robot_pose
            x_landmark, y_landmark = self.landmarks[landmark_id]

            # Expected measurement
            dx = x_landmark - x_robot
            dy = y_landmark - y_robot
            expected_range = np.sqrt(dx**2 + dy**2)
            expected_bearing = np.arctan2(dy, dx) - theta_robot
            expected_bearing = np.arctan2(np.sin(expected_bearing), np.cos(expected_bearing))  # Normalize

            expected_measurement = np.array([expected_range, expected_bearing])
            actual_measurement = np.array(measurement)

            # Innovation
            innovation = actual_measurement - expected_measurement

            # Jacobian of measurement function
            H = np.zeros((2, 5))  # 2 measurements, 5 state variables [x_r, y_r, theta_r, x_l, y_l]
            if expected_range > 1e-6:
                H[0, 0] = -dx / expected_range  # drange/dx_robot
                H[0, 1] = -dy / expected_range  # drange/dy_robot
                H[1, 0] = dy / expected_range**2  # dbearing/dx_robot
                H[1, 1] = -dx / expected_range**2  # dbearing/dy_robot
                H[1, 2] = -1  # dbearing/dtheta_robot
                H[0, 3] = -H[0, 0]  # drange/dx_landmark
                H[0, 4] = -H[0, 1]  # drange/dy_landmark
                H[1, 3] = -H[1, 0]  # dbearing/dx_landmark
                H[1, 4] = -H[1, 1]  # dbearing/dy_landmark

            # Kalman gain calculation (simplified)
            S = H @ np.block([[self.pose_covariance, np.zeros((3, 2))],
                              [np.zeros((2, 3)), 0.1*np.eye(2)]]) @ H.T + sensor_noise
            K = np.block([[self.pose_covariance, np.zeros((3, 2))],
                          [np.zeros((2, 3)), 0.1*np.eye(2)]]) @ H.T @ np.linalg.inv(S)

            # Update state
            state_update = K @ innovation
            self.robot_pose[:3] += state_update[:3]
            self.landmarks[landmark_id] += state_update[3:]

    def get_map(self):
        """Return the current map of landmarks"""
        return self.landmarks.copy()
```

### 4.5.2 Feature Detection and Matching

```python
class FeatureDetector:
    def __init__(self, detector_type='SIFT'):
        self.detector_type = detector_type
        if detector_type == 'SIFT':
            self.detector = cv2.SIFT_create()
        elif detector_type == 'ORB':
            self.detector = cv2.ORB_create()
        else:
            raise ValueError("Unsupported detector type")

    def detect_features(self, image):
        """Detect features in image"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors

    def match_features(self, desc1, desc2, ratio_threshold=0.7):
        """Match features between two images using FLANN matcher"""
        if desc1 is None or desc2 is None or len(desc1) == 0 or len(desc2) == 0:
            return []

        # FLANN matcher
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)

        matches = flann.knnMatch(desc1, desc2, k=2)

        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches

class VisualOdometry:
    def __init__(self):
        self.prev_image = None
        self.prev_features = None
        self.position = np.array([0.0, 0.0, 0.0])  # x, y, theta
        self.feature_detector = FeatureDetector('ORB')

    def process_frame(self, current_image):
        """Process a new frame and estimate motion"""
        if self.prev_image is None:
            # Initialize with first frame
            self.prev_image = current_image
            self.prev_features = self.feature_detector.detect_features(current_image)
            return self.position.copy()

        # Detect features in current frame
        curr_features, curr_descriptors = self.feature_detector.detect_features(current_image)

        # Match features between frames
        matches = self.feature_detector.match_features(
            self.prev_features[1], curr_descriptors
        )

        if len(matches) >= 10:  # Need enough matches for robust estimation
            # Extract matched points
            prev_pts = np.float32([self.prev_features[0][m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            curr_pts = np.float32([curr_features[0][m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            # Estimate motion using optical flow
            E, mask = cv2.findEssentialMat(
                curr_pts, prev_pts, focal=500, pp=(320, 240), method=cv2.RANSAC, threshold=1.0
            )

            if E is not None:
                # Recover pose
                _, R, t, _ = cv2.recoverPose(E, curr_pts, prev_pts, focal=500, pp=(320, 240))

                # Convert rotation matrix to angle (simplified for planar motion)
                angle = np.arctan2(R[1, 0], R[0, 0])

                # Update position (simplified)
                self.position[0] += t[0, 0]
                self.position[1] += t[1, 0]
                self.position[2] += angle

        # Update for next iteration
        self.prev_image = current_image
        self.prev_features = (curr_features[0], curr_descriptors)

        return self.position.copy()
```

## 4.6 Sensor-Based Control

### 4.6.1 Feedback Control with Sensor Data

```python
class SensorBasedController:
    def __init__(self, sensor_weights=None):
        if sensor_weights is None:
            self.sensor_weights = {'position': 1.0, 'force': 0.5, 'vision': 0.3}
        else:
            self.sensor_weights = sensor_weights

        self.integral_error = 0
        self.previous_error = 0
        self.max_integral = 10.0

    def compute_control(self, target, current_state, sensors_data, dt=0.01):
        """Compute control action based on sensor fusion"""
        # Compute error from different sensors
        position_error = self.get_position_error(target, current_state)
        force_error = self.get_force_error(sensors_data.get('force', np.zeros(3)))
        vision_error = self.get_vision_error(sensors_data.get('vision', np.zeros(2)))

        # Weighted combination of sensor errors
        total_error = (
            self.sensor_weights['position'] * position_error +
            self.sensor_weights['force'] * force_error +
            self.sensor_weights['vision'] * vision_error
        )

        # PID control
        self.integral_error += total_error * dt
        self.integral_error = np.clip(self.integral_error, -self.max_integral, self.max_integral)

        derivative_error = (total_error - self.previous_error) / dt if dt > 0 else 0

        # PID parameters
        kp, ki, kd = 1.0, 0.1, 0.05

        control_output = kp * total_error + ki * self.integral_error + kd * derivative_error

        self.previous_error = total_error

        return control_output

    def get_position_error(self, target, current):
        """Calculate position error"""
        return target - current

    def get_force_error(self, force_vector):
        """Calculate force error (deviation from desired force)"""
        # For example, if we want zero force
        desired_force = np.zeros_like(force_vector)
        return desired_force - force_vector

    def get_vision_error(self, vision_vector):
        """Calculate vision-based error"""
        # For example, deviation from center of image
        center = np.array([320, 240])  # Assuming 640x480 image
        return center - vision_vector
```

## Key Takeaways

- Sensors provide the crucial link between the physical world and AI systems
- Proprioceptive sensors measure internal state (position, velocity, forces)
- Exteroceptive sensors measure external environment (cameras, LIDAR, etc.)
- Sensor fusion combines multiple sensor readings to improve accuracy
- Kalman filters and particle filters are essential for state estimation
- SLAM enables simultaneous localization and mapping
- Feature detection and matching enable visual perception
- Sensor-based control uses sensor feedback for precise manipulation

## Exercises

1. **Implementation**: Implement a simple Kalman filter for tracking a moving object using camera measurements.

2. **Analysis**: Compare the performance of different sensor fusion techniques for robot localization.

3. **Design**: Design a sensor suite for a mobile manipulator and justify your choices.

4. **Experiment**: Implement visual odometry using feature matching and evaluate its accuracy.

5. **Research**: Investigate how event-based cameras can improve perception in Physical AI systems.

## Further Reading

1. Thrun, S., Burgard, W., & Fox, D. (2005). "Probabilistic Robotics." MIT Press.

2. Szeliski, R. (2022). "Computer Vision: Algorithms and Applications." Springer.

3. Siciliano, B., & Khatib, O. (2016). "Springer Handbook of Robotics." Springer.