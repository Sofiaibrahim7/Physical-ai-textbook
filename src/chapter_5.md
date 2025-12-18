# Chapter 5: Motor Control

Motor control is the bridge between decision-making and physical action in Physical AI systems. This chapter covers the fundamental principles of controlling actuators, from basic position control to sophisticated force and impedance control strategies that enable safe and effective interaction with the physical world.

## 5.1 Introduction to Motor Control

Motor control encompasses the algorithms and techniques used to command actuators to achieve desired movements, forces, and interactions. Unlike traditional control systems that might operate in simulation or on abstract data, motor control in Physical AI must account for real physical constraints, dynamics, and safety considerations.

Key aspects of motor control include:

1. **Position control**: Moving to desired positions
2. **Velocity control**: Achieving desired velocities
3. **Force control**: Applying desired forces to the environment
4. **Impedance control**: Controlling the mechanical impedance of the system
5. **Safety**: Ensuring safe operation during control

![Figure 5.1: Motor Control Hierarchy in Physical AI](placeholder)

## 5.2 Basic Control Concepts

### 5.2.1 Proportional-Integral-Derivative (PID) Control

PID control forms the foundation of most motor control systems:

```python
import numpy as np
import matplotlib.pyplot as plt

class PIDController:
    def __init__(self, kp=1.0, ki=0.0, kd=0.0, output_limits=(-10, 10)):
        self.kp = kp  # Proportional gain
        self.ki = ki  # Integral gain
        self.kd = kd  # Derivative gain
        self.output_limits = output_limits  # Min and max output values

        self.reset()

    def reset(self):
        """Reset the PID controller"""
        self.previous_error = 0.0
        self.integral = 0.0
        self.previous_time = None

    def compute(self, setpoint, measurement, dt=None):
        """Compute control output using PID algorithm"""
        current_time = np.time.time() if dt is None else None

        if self.previous_time is not None and dt is None:
            dt = current_time - self.previous_time

        if dt is None or dt <= 0:
            dt = 0.01  # Default time step

        # Calculate error
        error = setpoint - measurement

        # Proportional term
        p_term = self.kp * error

        # Integral term
        self.integral += error * dt
        i_term = self.ki * self.integral

        # Derivative term
        if dt > 0:
            derivative = (error - self.previous_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative

        # Calculate output
        output = p_term + i_term + d_term

        # Apply output limits
        output = np.clip(output, self.output_limits[0], self.output_limits[1])

        # Update previous values
        self.previous_error = error
        self.previous_time = current_time

        return output

class JointController:
    def __init__(self, joint_limits=(-np.pi, np.pi), kp=10.0, ki=1.0, kd=0.5):
        self.position_controller = PIDController(kp=kp, ki=ki, kd=kd)
        self.velocity_controller = PIDController(kp=kp/10, ki=ki/10, kd=kd/10)
        self.joint_limits = joint_limits
        self.current_position = 0.0
        self.current_velocity = 0.0

    def control_position(self, target_position, current_position, dt=0.01):
        """Control joint to reach target position"""
        # Apply joint limits
        target_position = np.clip(target_position, self.joint_limits[0], self.joint_limits[1])

        # Compute position error and control
        control_effort = self.position_controller.compute(target_position, current_position, dt)

        # Convert to torque (simplified)
        torque = control_effort

        return torque

    def control_velocity(self, target_velocity, current_velocity, dt=0.01):
        """Control joint to reach target velocity"""
        control_effort = self.velocity_controller.compute(target_velocity, current_velocity, dt)
        torque = control_effort
        return torque

    def control_with_feedforward(self, target_position, current_position, target_velocity, current_velocity, dt=0.01):
        """Control with feedforward compensation"""
        # Feedback control
        feedback_torque = self.control_position(target_position, current_position, dt)

        # Feedforward compensation (simplified)
        # In practice, this would include gravity, Coriolis, and other dynamic terms
        feedforward_torque = 0.1 * target_velocity  # Simplified velocity feedforward

        total_torque = feedback_torque + feedforward_torque

        # Apply limits
        max_torque = 100.0  # Nm
        total_torque = np.clip(total_torque, -max_torque, max_torque)

        return total_torque
```

### 5.2.2 System Identification

Before effective control can be achieved, system dynamics must be understood:

```python
class SystemIdentifier:
    def __init__(self, order=2):
        self.order = order
        self.input_history = []
        self.output_history = []
        self.parameters = None

    def collect_data(self, input_signal, output_signal):
        """Collect input-output data for system identification"""
        self.input_history.append(input_signal)
        self.output_history.append(output_signal)

    def identify_system(self):
        """Identify system parameters using least squares"""
        if len(self.input_history) < self.order * 2:
            return None

        # Formulate as Ax = b for least squares
        # For a second-order system: y(k) = a1*y(k-1) + a2*y(k-2) + b0*u(k-1) + b1*u(k-2)
        n = len(self.input_history)
        A = []
        b = []

        for k in range(self.order, n):
            row = []
            # Past outputs
            for i in range(1, self.order + 1):
                row.append(self.output_history[k - i])
            # Past inputs
            for i in range(1, self.order + 1):
                row.append(self.input_history[k - i])

            A.append(row)
            b.append(self.output_history[k])

        A = np.array(A)
        b = np.array(b)

        # Solve for parameters using least squares
        try:
            params = np.linalg.lstsq(A, b, rcond=None)[0]
            self.parameters = params
            return params
        except np.linalg.LinAlgError:
            return None

    def predict_output(self, input_sequence):
        """Predict system output using identified model"""
        if self.parameters is None:
            return None

        # Extract parameters (for second-order system)
        a1, a2 = self.parameters[:2]
        b0, b1 = self.parameters[2:4]

        outputs = []
        for i, u in enumerate(input_sequence):
            if i < 2:
                outputs.append(0)  # Initial conditions
            else:
                y_pred = a1 * outputs[i-1] + a2 * outputs[i-2] + b0 * input_sequence[i-1] + b1 * input_sequence[i-2]
                outputs.append(y_pred)

        return outputs
```

## 5.3 Advanced Control Strategies

### 5.3.1 Impedance Control

Impedance control allows specification of mechanical impedance (stiffness, damping) rather than just position:

```python
class ImpedanceController:
    def __init__(self, stiffness=1000, damping=100, mass=10):
        self.stiffness = stiffness  # N/m or Nm/rad
        self.damping = damping      # Ns/m or Nms/rad
        self.mass = mass           # kg or kg*m^2
        self.desired_position = 0.0
        self.desired_velocity = 0.0
        self.desired_acceleration = 0.0

    def compute_impedance_force(self, current_pos, current_vel, dt=0.01):
        """Compute force based on impedance model"""
        # Calculate position, velocity, and acceleration errors
        pos_error = self.desired_position - current_pos
        vel_error = self.desired_velocity - current_vel

        # Desired acceleration based on impedance model
        desired_acc = (self.stiffness * pos_error - self.damping * vel_error) / self.mass

        # Calculate required force
        force = self.mass * desired_acc

        return force

    def update_trajectory(self, target_pos, target_vel, dt=0.01):
        """Update desired trajectory"""
        self.desired_position = target_pos
        self.desired_velocity = target_vel

    def control_with_environment(self, current_pos, current_vel, external_force, dt=0.01):
        """Impedance control with external force compensation"""
        # Desired impedance behavior
        impedance_force = self.compute_impedance_force(current_pos, current_vel, dt)

        # Add external force (for interaction control)
        total_force = impedance_force + external_force

        return total_force

class AdmittanceController:
    def __init__(self, stiffness=100, damping=10, mass=1):
        self.stiffness = stiffness
        self.damping = damping
        self.mass = mass
        self.position = 0.0
        self.velocity = 0.0
        self.acceleration = 0.0

    def update_with_force(self, applied_force, dt=0.01):
        """Update position based on applied force (admittance control)"""
        # Calculate acceleration from force
        self.acceleration = applied_force / self.mass

        # Update velocity and position using numerical integration
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

        # Add damping and spring forces to simulate admittance
        spring_force = -self.stiffness * self.position
        damping_force = -self.damping * self.velocity

        return self.position, self.velocity, self.acceleration
```

### 5.3.2 Force Control

Force control is essential for safe interaction with the environment:

```python
class ForceController:
    def __init__(self, kp=100, ki=10, kd=5, max_force=50):
        self.force_controller = PIDController(kp=kp, ki=ki, kd=kd, output_limits=(-max_force, max_force))
        self.position_controller = PIDController(kp=50, ki=5, kd=2, output_limits=(-10, 10))
        self.desired_force = 0.0
        self.max_force = max_force

    def control_force(self, desired_force, measured_force, current_position, dt=0.01):
        """Control applied force while maintaining position"""
        self.desired_force = desired_force

        # Compute force error and required position adjustment
        force_error = desired_force - measured_force

        # Adjust position to achieve desired force
        position_adjustment = self.force_controller.compute(0, force_error, dt)

        # Calculate new target position
        target_position = current_position + position_adjustment

        # Apply position control to reach target
        control_effort = self.position_controller.compute(target_position, current_position, dt)

        return control_effort

    def hybrid_force_position_control(self, position_dof, force_dof,
                                    desired_pos, desired_force,
                                    current_pos, measured_force, dt=0.01):
        """Hybrid force/position control in different DOFs"""
        control_outputs = []

        for i in range(len(position_dof)):
            if position_dof[i]:  # Position control
                control_out = self.position_controller.compute(
                    desired_pos[i], current_pos[i], dt
                )
            elif force_dof[i]:  # Force control
                control_out = self.control_force(
                    desired_force[i], measured_force[i], current_pos[i], dt
                )
            else:  # Free motion
                control_out = 0

            control_outputs.append(control_out)

        return np.array(control_outputs)
```

## 5.4 Robot Control Architectures

### 5.4.1 Hierarchical Control

```python
class HierarchicalController:
    def __init__(self):
        # High-level planner
        self.trajectory_planner = TrajectoryPlanner()

        # Mid-level controller
        self.impedance_controller = ImpedanceController()

        # Low-level controller
        self.joint_controller = JointController()

    def control_step(self, task_command, sensor_data, dt=0.01):
        """Execute control step through hierarchy"""
        # High-level: Plan trajectory
        desired_trajectory = self.trajectory_planner.plan(task_command, sensor_data)

        # Mid-level: Impedance control
        desired_impedance = self.impedance_controller.compute_impedance_force(
            sensor_data['position'],
            sensor_data['velocity'],
            dt
        )

        # Low-level: Joint control
        joint_commands = []
        for i, (pos, vel) in enumerate(zip(sensor_data['position'], sensor_data['velocity'])):
            torque = self.joint_controller.control_with_feedforward(
                desired_trajectory['positions'][i],
                pos,
                desired_trajectory['velocities'][i],
                vel,
                dt
            )
            joint_commands.append(torque)

        return np.array(joint_commands)

class TrajectoryPlanner:
    def __init__(self):
        self.current_waypoint = 0

    def plan(self, task_command, sensor_data):
        """Plan trajectory based on task command"""
        if task_command['type'] == 'point_to_point':
            return self.plan_point_to_point(
                sensor_data['position'],
                task_command['target_position']
            )
        elif task_command['type'] == 'follow_path':
            return self.plan_path_following(task_command['path'])
        else:
            # Default: hold current position
            return {
                'positions': sensor_data['position'],
                'velocities': np.zeros_like(sensor_data['position']),
                'accelerations': np.zeros_like(sensor_data['position'])
            }

    def plan_point_to_point(self, start_pos, end_pos, duration=2.0, dt=0.01):
        """Plan smooth trajectory between two points"""
        # Use minimum jerk trajectory
        steps = int(duration / dt)
        t = np.linspace(0, 1, steps)

        # Minimum jerk polynomial: s(t) = 10*t^3 - 15*t^4 + 6*t^5
        s = 10*t**3 - 15*t**4 + 6*t**5

        # Interpolate between start and end
        positions = []
        velocities = []
        accelerations = []

        for i in range(len(start_pos)):
            pos_profile = start_pos[i] + s * (end_pos[i] - start_pos[i])

            # Velocity (derivative of position)
            ds_dt = (30*t**2 - 60*t**3 + 30*t**4) * (end_pos[i] - start_pos[i]) / duration
            vel_profile = ds_dt

            # Acceleration (derivative of velocity)
            d2s_dt2 = (60*t - 180*t**2 + 120*t**3) * (end_pos[i] - start_pos[i]) / (duration**2)
            acc_profile = d2s_dt2

            positions.append(pos_profile)
            velocities.append(vel_profile)
            accelerations.append(acc_profile)

        # Return current values (first step)
        current_positions = np.array([pos[0] for pos in positions])
        current_velocities = np.array([vel[0] for vel in velocities])
        current_accelerations = np.array([acc[0] for acc in accelerations])

        return {
            'positions': current_positions,
            'velocities': current_velocities,
            'accelerations': current_accelerations
        }
```

### 5.4.2 Operational Space Control

```python
class OperationalSpaceController:
    def __init__(self, robot_model):
        self.robot_model = robot_model
        self.jacobian_cache = {}
        self.mass_matrix_cache = {}

    def control_cartesian(self, target_pos, target_vel, target_acc,
                         current_joints, current_velocities, dt=0.01):
        """Control end-effector in Cartesian space"""
        # Compute Jacobian
        J = self.robot_model.jacobian(current_joints)

        # Compute task-space mass matrix
        M_inv = np.linalg.inv(self.robot_model.mass_matrix(current_joints))
        Lambda = np.linalg.inv(J @ M_inv @ J.T)

        # Compute Coriolis and gravity compensation
        h = self.robot_model.coriolis_gravity(current_joints, current_velocities)

        # Compute desired task-space force
        pos_error = target_pos - self.robot_model.forward_kinematics(current_joints)[:3, 3]
        vel_error = target_vel - J @ current_velocities

        Kp = 100 * np.eye(3)
        Kd = 20 * np.eye(3)

        F_task = Lambda @ (target_acc + Kp @ pos_error + Kd @ vel_error)

        # Convert to joint torques
        torques = J.T @ F_task + h

        return torques

    def impedance_control_cartesian(self, target_pose, stiffness, damping,
                                  current_joints, current_velocities, dt=0.01):
        """Impedance control in Cartesian space"""
        # Get current Cartesian pose
        current_pose = self.robot_model.forward_kinematics(current_joints)

        # Compute pose error
        pos_error = target_pose[:3, 3] - current_pose[:3, 3]
        rot_error = self.rotation_error(target_pose[:3, :3], current_pose[:3, :3])

        pose_error = np.concatenate([pos_error, rot_error])

        # Compute desired acceleration
        M_task = self.compute_task_inertia(current_joints)
        K = np.diag(stiffness)  # Stiffness matrix
        D = np.diag(damping)    # Damping matrix

        J = self.robot_model.jacobian(current_joints)
        current_task_vel = J @ current_velocities

        desired_acc = np.linalg.inv(M_task) @ (
            -K @ pose_error - D @ current_task_vel
        )

        # Convert to joint space
        torques = self.compute_cartesian_impedance_torques(
            desired_acc, current_joints, current_velocities, J
        )

        return torques

    def compute_task_inertia(self, joints):
        """Compute task-space inertia matrix"""
        M = self.robot_model.mass_matrix(joints)
        J = self.robot_model.jacobian(joints)
        M_inv = np.linalg.inv(M)
        return np.linalg.inv(J @ M_inv @ J.T)

    def rotation_error(self, R_desired, R_current):
        """Compute rotation error as angle-axis representation"""
        R_error = R_desired @ R_current.T
        angle = np.arccos(np.clip((np.trace(R_error) - 1) / 2, -1, 1))

        if np.sin(angle) != 0:
            axis = np.array([
                R_error[2, 1] - R_error[1, 2],
                R_error[0, 2] - R_error[2, 0],
                R_error[1, 0] - R_error[0, 1]
            ]) / (2 * np.sin(angle))
        else:
            axis = np.array([0, 0, 0])

        return angle * axis

    def compute_cartesian_impedance_torques(self, x_ddot, joints, velocities, jacobian):
        """Compute joint torques for Cartesian impedance control"""
        M = self.robot_model.mass_matrix(joints)
        J = jacobian
        h = self.robot_model.coriolis_gravity(joints, velocities)

        # Task space inertia
        Lambda = self.compute_task_inertia(joints)

        # Joint space inertia
        M_inv = np.linalg.inv(M)
        J_pinv = M_inv @ J.T @ np.linalg.inv(J @ M_inv @ J.T)

        # Compute torques
        tau = J.T @ Lambda @ x_ddot + h

        return tau
```

## 5.5 Safety and Robustness

### 5.5.1 Safety Controllers

```python
class SafetyController:
    def __init__(self, max_velocity=1.0, max_acceleration=5.0, max_force=100.0):
        self.max_velocity = max_velocity
        self.max_acceleration = max_acceleration
        self.max_force = max_force
        self.safety_margin = 0.1

    def check_safety(self, state, sensor_data):
        """Check if current state is safe"""
        # Check velocity limits
        if np.any(np.abs(state['velocity']) > self.max_velocity * (1 - self.safety_margin)):
            return False, "Velocity limit exceeded"

        # Check force limits
        if 'force' in sensor_data:
            if np.any(np.abs(sensor_data['force']) > self.max_force * (1 - self.safety_margin)):
                return False, "Force limit exceeded"

        # Check position limits
        if 'position' in sensor_data:
            if np.any(np.abs(sensor_data['position']) > 10.0):  # Example limit
                return False, "Position limit exceeded"

        return True, "Safe"

    def enforce_safety(self, control_command, state, sensor_data):
        """Enforce safety constraints on control command"""
        # Check if we need to limit the command
        is_safe, reason = self.check_safety(state, sensor_data)

        if not is_safe:
            # Emergency stop or reduce command
            if "Velocity" in reason:
                # Limit acceleration
                current_vel = state['velocity']
                max_change = self.max_acceleration * 0.01  # dt = 0.01s
                limited_vel = np.clip(
                    control_command,
                    current_vel - max_change,
                    current_vel + max_change
                )
                return limited_vel
            elif "Force" in reason:
                # Reduce force command
                return control_command * 0.5  # Reduce by half
            else:
                # Emergency stop
                return np.zeros_like(control_command)

        return control_command

class BackupController:
    def __init__(self):
        self.active = False
        self.backup_mode = "stop"  # "stop", "home", "safe_position"

    def activate_backup(self, reason="Emergency"):
        """Activate backup controller"""
        print(f"Activating backup controller: {reason}")
        self.active = True

    def deactivate_backup(self):
        """Deactivate backup controller"""
        self.active = False

    def get_backup_command(self, current_state):
        """Get command from backup controller"""
        if not self.active:
            return None

        if self.backup_mode == "stop":
            # Zero velocity command
            return np.zeros_like(current_state['velocity'])
        elif self.backup_mode == "home":
            # Move to home position
            home_pos = np.zeros_like(current_state['position'])
            error = home_pos - current_state['position']
            return 10 * error  # Simple proportional control to home
        elif self.backup_mode == "safe_position":
            # Move to predefined safe position
            safe_pos = np.array([0.5, 0.0, -0.5])  # Example safe position
            if len(current_state['position']) >= 3:
                error = safe_pos - current_state['position'][:3]
                return np.concatenate([10 * error, np.zeros(len(current_state['position']) - 3)])
            else:
                return np.zeros_like(current_state['position'])
```

### 5.5.2 Adaptive Control

```python
class AdaptiveController:
    def __init__(self, initial_params=None):
        if initial_params is None:
            self.params = np.array([10.0, 1.0, 0.5])  # [kp, ki, kd]
        else:
            self.params = np.array(initial_params)

        self.param_adaptation_rate = 0.001
        self.error_history = []
        self.max_history = 100

    def update_params(self, tracking_error, dt=0.01):
        """Update controller parameters based on tracking error"""
        # Store error for adaptation
        self.error_history.append(tracking_error)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)

        # Compute performance metric (e.g., integral of squared error)
        if len(self.error_history) > 10:  # Need enough data
            ise = np.sum([e**2 for e in self.error_history[-10:]])

            # Gradient-based parameter update (simplified)
            # In practice, this would use more sophisticated adaptation laws
            param_gradients = np.array([
                -2 * tracking_error * tracking_error,  # Gradient w.r.t. kp
                -2 * np.sum(self.error_history[-10:]) * tracking_error,  # Gradient w.r.t. ki
                -2 * (tracking_error - self.error_history[-2]) / dt if len(self.error_history) > 1 else 0  # Gradient w.r.t. kd
            ])

            # Update parameters
            self.params += self.param_adaptation_rate * param_gradients

            # Apply parameter limits
            self.params = np.clip(self.params, [1e-6, 1e-6, 1e-6], [1000, 100, 100])

    def compute_control(self, setpoint, measurement, dt=0.01):
        """Compute control with adaptive parameters"""
        # Create PID controller with current parameters
        pid = PIDController(kp=self.params[0], ki=self.params[1], kd=self.params[2])
        control_output = pid.compute(setpoint, measurement, dt)

        # Update parameters based on error
        tracking_error = setpoint - measurement
        self.update_params(tracking_error, dt)

        return control_output

class RobustController:
    def __init__(self, nominal_params, uncertainty_bounds):
        self.nominal_params = nominal_params
        self.uncertainty_bounds = uncertainty_bounds
        self.robust_margin = 1.5  # Extra gain for robustness

    def compute_robust_control(self, setpoint, measurement, dt=0.01):
        """Compute control with robustness to parameter uncertainty"""
        # Calculate nominal control
        nominal_pid = PIDController(
            kp=self.nominal_params[0],
            ki=self.nominal_params[1],
            kd=self.nominal_params[2]
        )
        nominal_control = nominal_pid.compute(setpoint, measurement, dt)

        # Add robustness term to handle uncertainties
        error = setpoint - measurement
        error_derivative = (error - getattr(self, 'prev_error', error)) / dt if dt > 0 else 0
        self.prev_error = error

        # Robust term based on uncertainty bounds
        robust_term = 0
        for bound in self.uncertainty_bounds:
            robust_term += bound * (abs(error) + abs(error_derivative))

        # Total control with robustness
        robust_control = nominal_control + self.robust_margin * robust_term * np.sign(error)

        return robust_control
```

## 5.6 Real-Time Control Implementation

### 5.6.1 Real-Time Control Loop

```python
import time
import threading
from collections import deque

class RealTimeController:
    def __init__(self, control_frequency=1000):  # 1kHz
        self.control_frequency = control_frequency
        self.control_period = 1.0 / control_frequency
        self.running = False
        self.control_thread = None

        # Control components
        self.controllers = {}
        self.sensors = {}
        self.actuators = {}

        # Timing statistics
        self.loop_times = deque(maxlen=1000)
        self.deadline_misses = 0

    def add_controller(self, name, controller):
        """Add a controller to the system"""
        self.controllers[name] = controller

    def add_sensor(self, name, sensor):
        """Add a sensor to the system"""
        self.sensors[name] = sensor

    def add_actuator(self, name, actuator):
        """Add an actuator to the system"""
        self.actuators[name] = actuator

    def control_loop(self):
        """Main control loop running at specified frequency"""
        while self.running:
            start_time = time.time()

            try:
                # Read sensors
                sensor_data = {}
                for name, sensor in self.sensors.items():
                    sensor_data[name] = sensor.read()

                # Execute control algorithms
                control_commands = {}
                for name, controller in self.controllers.items():
                    if name in sensor_data:
                        command = controller.compute_control(sensor_data[name])
                        control_commands[name] = command

                # Send commands to actuators
                for name, command in control_commands.items():
                    if name in self.actuators:
                        self.actuators[name].send_command(command)

            except Exception as e:
                print(f"Control loop error: {e}")
                self.emergency_stop()

            # Calculate loop time
            loop_time = time.time() - start_time
            self.loop_times.append(loop_time)

            # Calculate sleep time to maintain frequency
            sleep_time = self.control_period - loop_time
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                self.deadline_misses += 1
                # Log deadline miss but continue

    def start(self):
        """Start the real-time control loop"""
        if not self.running:
            self.running = True
            self.control_thread = threading.Thread(target=self.control_loop)
            self.control_thread.start()

    def stop(self):
        """Stop the real-time control loop"""
        self.running = False
        if self.control_thread:
            self.control_thread.join()

    def emergency_stop(self):
        """Emergency stop all actuators"""
        for name, actuator in self.actuators.items():
            actuator.emergency_stop()

    def get_performance_stats(self):
        """Get control loop performance statistics"""
        if len(self.loop_times) == 0:
            return None

        avg_loop_time = np.mean(self.loop_times)
        max_loop_time = np.max(self.loop_times)
        min_loop_time = np.min(self.loop_times)
        deadline_miss_rate = self.deadline_misses / len(self.loop_times) if len(self.loop_times) > 0 else 0

        return {
            'avg_loop_time': avg_loop_time,
            'max_loop_time': max_loop_time,
            'min_loop_time': min_loop_time,
            'deadline_miss_rate': deadline_miss_rate,
            'utilization': avg_loop_time / self.control_period
        }
```

### 5.6.2 Control System Integration

```python
class IntegratedMotorController:
    def __init__(self, robot_config):
        self.robot_config = robot_config

        # Initialize controllers for each joint
        self.joint_controllers = {}
        for joint_name in robot_config['joints']:
            self.joint_controllers[joint_name] = JointController(
                joint_limits=robot_config['joints'][joint_name]['limits'],
                kp=robot_config['joints'][joint_name].get('kp', 10.0),
                ki=robot_config['joints'][joint_name].get('ki', 1.0),
                kd=robot_config['joints'][joint_name].get('kd', 0.5)
            )

        # High-level controllers
        self.trajectory_planner = TrajectoryPlanner()
        self.operational_controller = OperationalSpaceController(robot_config['model'])
        self.safety_controller = SafetyController()

        # State tracking
        self.current_positions = np.zeros(len(robot_config['joints']))
        self.current_velocities = np.zeros(len(robot_config['joints']))
        self.current_efforts = np.zeros(len(robot_config['joints']))

    def update_state(self, joint_positions, joint_velocities, joint_efforts):
        """Update internal state with current measurements"""
        self.current_positions = np.array(joint_positions)
        self.current_velocities = np.array(joint_velocities)
        self.current_efforts = np.array(joint_efforts)

    def compute_commands(self, task_command, sensor_data, dt=0.01):
        """Compute motor commands based on task and sensor data"""
        # Safety check first
        is_safe, safety_msg = self.safety_controller.check_safety(
            {'position': self.current_positions, 'velocity': self.current_velocities},
            sensor_data
        )

        if not is_safe:
            print(f"Safety violation: {safety_msg}")
            return self.emergency_commands()

        # Process task command
        if task_command['type'] == 'joint_position':
            commands = self.joint_position_control(task_command['positions'], dt)
        elif task_command['type'] == 'cartesian_pose':
            commands = self.cartesian_control(task_command['pose'], dt)
        elif task_command['type'] == 'trajectory':
            commands = self.follow_trajectory(task_command['trajectory'], dt)
        elif task_command['type'] == 'force_control':
            commands = self.force_control(task_command['forces'], sensor_data, dt)
        else:
            # Default: hold current position
            commands = self.hold_position(self.current_positions, dt)

        # Apply safety limits
        commands = self.safety_controller.enforce_safety(
            commands,
            {'position': self.current_positions, 'velocity': self.current_velocities},
            sensor_data
        )

        return commands

    def joint_position_control(self, target_positions, dt=0.01):
        """Control joints to reach target positions"""
        commands = []
        for i, (target_pos, current_pos) in enumerate(zip(target_positions, self.current_positions)):
            joint_name = list(self.joint_controllers.keys())[i]
            controller = self.joint_controllers[joint_name]

            command = controller.control_position(target_pos, current_pos, dt)
            commands.append(command)

        return np.array(commands)

    def cartesian_control(self, target_pose, dt=0.01):
        """Control end-effector to reach target Cartesian pose"""
        try:
            commands = self.operational_controller.control_cartesian(
                target_pose[:3, 3],  # Position
                np.zeros(3),        # Desired velocity
                np.zeros(3),        # Desired acceleration
                self.current_positions,
                self.current_velocities,
                dt
            )
            return commands
        except Exception as e:
            print(f"Cartesian control failed: {e}")
            # Fall back to joint control
            return self.hold_position(self.current_positions, dt)

    def follow_trajectory(self, trajectory, dt=0.01):
        """Follow a predefined trajectory"""
        # This would typically use trajectory interpolation
        # For simplicity, just use the trajectory planner
        planned = self.trajectory_planner.plan_point_to_point(
            self.current_positions,
            trajectory['target'],
            duration=trajectory.get('duration', 2.0),
            dt=dt
        )

        target_positions = planned['positions']
        return self.joint_position_control(target_positions, dt)

    def force_control(self, target_forces, sensor_data, dt=0.01):
        """Control forces applied by the robot"""
        if 'force_torque' not in sensor_data:
            return self.hold_position(self.current_positions, dt)

        force_controller = ForceController()
        commands = []

        for i, (target_force, measured_force) in enumerate(
            zip(target_forces, sensor_data['force_torque'])
        ):
            command = force_controller.control_force(
                target_force, measured_force, self.current_positions[i], dt
            )
            commands.append(command)

        return np.array(commands)

    def hold_position(self, target_positions, dt=0.01):
        """Hold current or target position"""
        return self.joint_position_control(target_positions, dt)

    def emergency_commands(self):
        """Return emergency stop commands"""
        return np.zeros_like(self.current_positions)
```

## Key Takeaways

- PID control provides the foundation for most motor control systems
- Impedance control enables safe interaction with the environment
- Force control is essential for tasks requiring precise force application
- Hierarchical control structures organize complex control tasks
- Operational space control allows control in task coordinates
- Safety systems are critical for physical AI systems
- Real-time implementation requires careful timing considerations
- Adaptive control handles parameter uncertainties
- System identification is necessary for effective control

## Exercises

1. **Implementation**: Implement a PID controller for a single joint and tune the parameters for optimal performance.

2. **Analysis**: Compare position control vs. impedance control for a manipulation task with environmental contact.

3. **Design**: Design a hybrid force/position controller for a peg-in-hole assembly task.

4. **Simulation**: Simulate a robotic arm with different control strategies and analyze their performance.

5. **Research**: Investigate how machine learning techniques can improve traditional motor control approaches.

## Further Reading

1. Spong, M. W., Hutchinson, S., & Vidyasagar, M. (2006). "Robot Modeling and Control." Wiley.

2. Sciavicco, L., & Siciliano, B. (2000). "Modelling and Control of Robot Manipulators." Springer.

3. Slotine, J. J. E., & Li, W. (1991). "Applied Nonlinear Control." Prentice Hall.