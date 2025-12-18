# Chapter 3: Physics for AI

Understanding the physical world is fundamental to Physical AI. This chapter covers the essential physics concepts that AI systems must understand to operate effectively in the real world, from basic mechanics to complex multi-body dynamics.

## 3.1 Introduction to Physics for AI

Physical AI systems must understand and reason about the physical world to achieve their goals. This requires knowledge of physics principles ranging from basic Newtonian mechanics to complex phenomena like fluid dynamics and material properties.

The physics for AI encompasses:

1. **Classical mechanics**: Motion, forces, and energy
2. **Dynamics**: How systems evolve over time
3. **Contact mechanics**: Interactions between objects
4. **Material properties**: How different materials behave
5. **Multi-body systems**: Interactions in complex mechanical systems

![Figure 3.1: Physics Understanding in Physical AI](placeholder)

## 3.2 Classical Mechanics Fundamentals

### 3.2.1 Newton's Laws of Motion

Newton's three laws form the foundation of classical mechanics:

1. **First Law (Inertia)**: An object at rest stays at rest, and an object in motion stays in motion at constant velocity unless acted upon by a net external force.
2. **Second Law (F = ma)**: The acceleration of an object is directly proportional to the net force acting on it and inversely proportional to its mass.
3. **Third Law (Action-Reaction)**: For every action, there is an equal and opposite reaction.

```python
import numpy as np

class NewtonianPhysics:
    def __init__(self, mass=1.0):
        self.mass = mass
        self.position = np.zeros(3)
        self.velocity = np.zeros(3)
        self.acceleration = np.zeros(3)

    def apply_force(self, force, dt=0.01):
        """Apply force and update motion using Newton's second law"""
        # F = ma => a = F/m
        self.acceleration = force / self.mass

        # Update velocity and position using Euler integration
        self.velocity += self.acceleration * dt
        self.position += self.velocity * dt

        return self.position.copy(), self.velocity.copy()

# Example: Simple projectile motion
def simulate_projectile(initial_velocity, angle, gravity=9.81, duration=10.0, dt=0.01):
    """Simulate projectile motion under gravity"""
    physics = NewtonianPhysics(mass=1.0)

    # Set initial conditions
    v0_x = initial_velocity * np.cos(np.radians(angle))
    v0_y = initial_velocity * np.sin(np.radians(angle))
    physics.velocity = np.array([v0_x, v0_y, 0.0])

    # Gravity force
    gravity_force = np.array([0.0, -gravity, 0.0])

    trajectory = []
    for t in np.arange(0, duration, dt):
        pos, vel = physics.apply_force(gravity_force * physics.mass, dt)
        trajectory.append(pos.copy())

        # Stop if projectile hits ground
        if pos[1] < 0:
            break

    return np.array(trajectory)
```

### 3.2.2 Conservation Laws

Conservation laws provide powerful constraints for understanding physical systems:

- **Conservation of Energy**: Energy cannot be created or destroyed, only transformed from one form to another.
- **Conservation of Momentum**: The total momentum of a closed system remains constant.
- **Conservation of Angular Momentum**: The angular momentum of a closed system remains constant.

```python
class ConservationLaws:
    def __init__(self):
        pass

    def kinetic_energy(self, mass, velocity):
        """Calculate kinetic energy: KE = 0.5 * m * v^2"""
        speed = np.linalg.norm(velocity)
        return 0.5 * mass * speed**2

    def potential_energy_gravitational(self, mass, height, gravity=9.81):
        """Calculate gravitational potential energy: PE = mgh"""
        return mass * gravity * height

    def momentum(self, mass, velocity):
        """Calculate momentum: p = mv"""
        return mass * velocity

    def angular_momentum(self, position, momentum):
        """Calculate angular momentum: L = r × p"""
        return np.cross(position, momentum)

    def simulate_elastic_collision(self, m1, v1, m2, v2):
        """Simulate 1D elastic collision using conservation of momentum and energy"""
        # For 1D elastic collision:
        # v1_final = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
        # v2_final = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)

        v1_final = ((m1 - m2) * v1 + 2 * m2 * v2) / (m1 + m2)
        v2_final = ((m2 - m1) * v2 + 2 * m1 * v1) / (m1 + m2)

        return v1_final, v2_final
```

## 3.3 Rigid Body Dynamics

Rigid body dynamics describes the motion of objects that maintain their shape under applied forces.

### 3.3.1 Rotational Motion

Rotational motion involves angular displacement, velocity, and acceleration:

```python
class RigidBody:
    def __init__(self, mass=1.0, inertia_tensor=None):
        self.mass = mass
        # 3x3 inertia tensor (simplified as diagonal for symmetric objects)
        if inertia_tensor is None:
            self.inertia_tensor = np.eye(3) * 0.4 * mass  # Approximate for sphere
        else:
            self.inertia_tensor = np.array(inertia_tensor)

        self.position = np.zeros(3)
        self.orientation = np.eye(3)  # Rotation matrix
        self.linear_velocity = np.zeros(3)
        self.angular_velocity = np.zeros(3)

    def update_rotation(self, torque, dt=0.01):
        """Update orientation using torque and angular velocity"""
        # Angular acceleration: alpha = I^(-1) * torque
        angular_acceleration = np.linalg.inv(self.inertia_tensor) @ torque

        # Update angular velocity
        self.angular_velocity += angular_acceleration * dt

        # Update orientation (simplified using small angle approximation)
        # In practice, quaternions are often used for better numerical stability
        omega_skew = np.array([
            [0, -self.angular_velocity[2], self.angular_velocity[1]],
            [self.angular_velocity[2], 0, -self.angular_velocity[0]],
            [-self.angular_velocity[1], self.angular_velocity[0], 0]
        ])

        self.orientation += omega_skew @ self.orientation * dt
        # Re-orthogonalize rotation matrix
        U, _, Vt = np.linalg.svd(self.orientation)
        self.orientation = U @ Vt

        return self.orientation.copy()

    def kinetic_energy_rotational(self):
        """Calculate rotational kinetic energy: KE = 0.5 * ω^T * I * ω"""
        return 0.5 * self.angular_velocity.T @ self.inertia_tensor @ self.angular_velocity
```

### 3.3.2 Euler-Lagrange Dynamics

The Euler-Lagrange formulation provides a systematic way to derive equations of motion:

```python
class LagrangianDynamics:
    def __init__(self, generalized_coordinates):
        self.q = generalized_coordinates  # Generalized coordinates
        self.q_dot = np.zeros_like(generalized_coordinates)  # Generalized velocities

    def kinetic_energy(self, q, q_dot):
        """Calculate kinetic energy as function of generalized coordinates and velocities"""
        # This would be specific to the system
        # For a simple system: T = 0.5 * q_dot^T * M(q) * q_dot
        # where M(q) is the mass matrix
        raise NotImplementedError

    def potential_energy(self, q):
        """Calculate potential energy as function of generalized coordinates"""
        raise NotImplementedError

    def lagrangian(self, q, q_dot):
        """Calculate Lagrangian: L = T - V"""
        return self.kinetic_energy(q, q_dot) - self.potential_energy(q)

    def euler_lagrange_equation(self, q, q_dot, t):
        """Calculate d/dt(∂L/∂q_dot) - ∂L/∂q = Q (generalized forces)"""
        # This would require computing partial derivatives of the Lagrangian
        raise NotImplementedError

class DoublePendulum(LagrangianDynamics):
    def __init__(self, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
        super().__init__(np.array([np.pi/4, np.pi/4]))  # Initial angles
        self.m1, self.m2 = m1, m2
        self.l1, self.l2 = l1, l2
        self.g = g

    def kinetic_energy(self, q, q_dot):
        """Kinetic energy of double pendulum"""
        theta1, theta2 = q
        theta1_dot, theta2_dot = q_dot

        # Complex kinetic energy expression for double pendulum
        t1 = 0.5 * self.m1 * (self.l1 * theta1_dot)**2
        t2 = 0.5 * self.m2 * (
            (self.l1 * theta1_dot)**2 +
            (self.l2 * theta2_dot)**2 +
            2 * self.l1 * self.l2 * theta1_dot * theta2_dot * np.cos(theta1 - theta2)
        )
        return t1 + t2

    def potential_energy(self, q):
        """Potential energy of double pendulum"""
        theta1, theta2 = q

        v1 = -self.m1 * self.g * self.l1 * np.cos(theta1)
        v2 = -self.m2 * self.g * (
            self.l1 * np.cos(theta1) + self.l2 * np.cos(theta2)
        )
        return v1 + v2

    def equations_of_motion(self, state, t):
        """Return derivatives of state for numerical integration"""
        theta1, theta2, theta1_dot, theta2_dot = state

        # Calculate the derivatives using Lagrange's equations
        # (This is a simplified version - full derivation is quite complex)

        # Mass matrix components
        M11 = (self.m1 + self.m2) * self.l1**2 + self.m2 * self.l2**2 + \
              2 * self.m2 * self.l1 * self.l2 * np.cos(theta1 - theta2)
        M12 = self.m2 * self.l2**2 + self.m2 * self.l1 * self.l2 * np.cos(theta1 - theta2)
        M21 = M12
        M22 = self.m2 * self.l2**2

        # Force vector components
        F1 = -self.m2 * self.l1 * self.l2 * theta2_dot**2 * np.sin(theta1 - theta2) - \
             (self.m1 + self.m2) * self.g * self.l1 * np.sin(theta1)
        F2 = self.m2 * self.l1 * self.l2 * theta1_dot**2 * np.sin(theta1 - theta2) - \
             self.m2 * self.g * self.l2 * np.sin(theta2)

        # Solve M * [thetadd1, thetadd2] = [F1, F2]
        M = np.array([[M11, M12], [M21, M22]])
        F = np.array([F1, F2])

        accelerations = np.linalg.solve(M, F)
        theta1_ddot, theta2_ddot = accelerations

        return np.array([theta1_dot, theta2_dot, theta1_ddot, theta2_ddot])
```

## 3.4 Contact Mechanics

Contact mechanics deals with the interaction between objects when they come into contact.

### 3.4.1 Collision Detection and Response

```python
class ContactModel:
    def __init__(self, restitution=0.8, friction=0.1):
        self.restitution = restitution  # Coefficient of restitution (bounciness)
        self.friction = friction        # Coefficient of friction

    def sphere_sphere_collision(self, pos1, vel1, radius1, mass1,
                               pos2, vel2, radius2, mass2):
        """Handle collision between two spheres"""
        # Calculate collision normal
        collision_vector = pos2 - pos1
        distance = np.linalg.norm(collision_vector)

        if distance > radius1 + radius2:
            return vel1, vel2  # No collision

        normal = collision_vector / distance

        # Relative velocity in normal direction
        rel_vel = vel1 - vel2
        vel_along_normal = np.dot(rel_vel, normal)

        # Do not resolve if velocities are separating
        if vel_along_normal > 0:
            return vel1, vel2

        # Calculate impulse magnitude
        impulse_magnitude = -(1 + self.restitution) * vel_along_normal
        impulse_magnitude /= (1/mass1 + 1/mass2)

        # Apply impulse
        impulse = impulse_magnitude * normal
        vel1_after = vel1 + impulse / mass1
        vel2_after = vel2 - impulse / mass2

        # Apply friction (simplified)
        rel_vel_after = vel1_after - vel2_after
        tangent_vel = rel_vel_after - np.dot(rel_vel_after, normal) * normal
        tangent_speed = np.linalg.norm(tangent_vel)

        if tangent_speed > 0:
            tangent = tangent_vel / tangent_speed
            friction_impulse = min(
                self.friction * impulse_magnitude,
                tangent_speed / (1/mass1 + 1/mass2)
            )
            friction_vector = friction_impulse * tangent

            vel1_after -= friction_vector / mass1
            vel2_after += friction_vector / mass2

        return vel1_after, vel2_after

    def plane_sphere_collision(self, sphere_pos, sphere_vel, sphere_radius, sphere_mass,
                              plane_point, plane_normal):
        """Handle collision between sphere and plane"""
        plane_normal = plane_normal / np.linalg.norm(plane_normal)

        # Calculate distance from sphere center to plane
        distance = np.dot(sphere_pos - plane_point, plane_normal)

        if distance > sphere_radius:
            return sphere_vel  # No collision

        # Calculate velocity in normal direction
        vel_normal = np.dot(sphere_vel, plane_normal)

        if vel_normal > 0:
            return sphere_vel  # Moving away from plane

        # Reflect velocity with restitution
        normal_impulse = -(1 + self.restitution) * vel_normal
        reflected_vel = sphere_vel + normal_impulse * plane_normal

        # Apply friction
        tangent_vel = sphere_vel - vel_normal * plane_normal
        tangent_speed = np.linalg.norm(tangent_vel)

        if tangent_speed > 0:
            tangent = tangent_vel / tangent_speed
            friction_impulse = min(
                self.friction * abs(normal_impulse) * sphere_mass,
                tangent_speed
            )
            friction_deceleration = friction_impulse * tangent / sphere_mass
            reflected_vel -= friction_deceleration

        return reflected_vel
```

### 3.4.2 Soft Body Dynamics

For deformable objects, more complex models are needed:

```python
class MassSpringSystem:
    def __init__(self, masses, spring_constants, rest_lengths, connections):
        """
        masses: list of masses
        spring_constants: list of spring constants
        rest_lengths: list of rest lengths
        connections: list of (node1, node2) tuples
        """
        self.masses = np.array(masses)
        self.spring_constants = np.array(spring_constants)
        self.rest_lengths = np.array(rest_lengths)
        self.connections = connections  # List of (i, j) tuples

        # Initialize positions (random for now)
        self.positions = np.random.rand(len(masses), 3) * 2 - 1
        self.velocities = np.zeros((len(masses), 3))

    def compute_spring_forces(self):
        """Compute forces from all springs"""
        forces = np.zeros_like(self.positions)

        for idx, (i, j) in enumerate(self.connections):
            # Vector from i to j
            delta_pos = self.positions[j] - self.positions[i]
            distance = np.linalg.norm(delta_pos)

            if distance > 0:
                # Unit vector
                direction = delta_pos / distance

                # Spring force (Hooke's law)
                spring_force = self.spring_constants[idx] * (distance - self.rest_lengths[idx])
                force_vector = spring_force * direction

                # Apply equal and opposite forces
                forces[i] += force_vector
                forces[j] -= force_vector

        return forces

    def step(self, dt=0.01, external_forces=None):
        """Update the mass-spring system"""
        if external_forces is None:
            external_forces = np.zeros_like(self.positions)

        # Compute internal forces
        internal_forces = self.compute_spring_forces()

        # Total forces
        total_forces = internal_forces + external_forces

        # Update velocities and positions using F=ma
        accelerations = total_forces / self.masses.reshape(-1, 1)
        self.velocities += accelerations * dt
        self.positions += self.velocities * dt

        # Apply gravity
        gravity = np.array([0, -9.81, 0])
        self.velocities += gravity * dt
```

## 3.5 Multi-Body Dynamics

Complex physical systems often involve multiple interconnected bodies with constraints.

### 3.5.1 Articulated Rigid Bodies

```python
class ArticulatedBody:
    def __init__(self, links, joints):
        """
        links: list of link properties (mass, inertia, etc.)
        joints: list of joint definitions
        """
        self.links = links
        self.joints = joints
        self.joint_angles = np.zeros(len(joints))
        self.joint_velocities = np.zeros(len(joints))

    def forward_kinematics(self, joint_angles):
        """Calculate end-effector position from joint angles"""
        # Simplified for a planar 2-link manipulator
        if len(joint_angles) >= 2:
            l1, l2 = self.links[0]['length'], self.links[1]['length']
            theta1, theta2 = joint_angles[0], joint_angles[1]

            x = l1 * np.cos(theta1) + l2 * np.cos(theta1 + theta2)
            y = l1 * np.sin(theta1) + l2 * np.sin(theta1 + theta2)

            return np.array([x, y, 0])
        return np.zeros(3)

    def jacobian(self, joint_angles):
        """Calculate Jacobian matrix for velocity kinematics"""
        if len(joint_angles) >= 2:
            l1, l2 = self.links[0]['length'], self.links[1]['length']
            theta1, theta2 = joint_angles[0], joint_angles[1]

            # Jacobian for 2-link planar manipulator
            J = np.array([
                [-l1*np.sin(theta1) - l2*np.sin(theta1+theta2), -l2*np.sin(theta1+theta2)],
                [l1*np.cos(theta1) + l2*np.cos(theta1+theta2), l2*np.cos(theta1+theta2)],
                [0, 0]  # No rotation in planar case
            ])

            return J
        return np.zeros((3, len(joint_angles)))

    def inverse_kinematics(self, target_pos, max_iterations=100, tolerance=1e-4):
        """Solve inverse kinematics using Jacobian transpose method"""
        current_angles = self.joint_angles.copy()

        for i in range(max_iterations):
            current_pos = self.forward_kinematics(current_angles)
            error = target_pos - current_pos

            if np.linalg.norm(error) < tolerance:
                break

            J = self.jacobian(current_angles)
            # Use pseudoinverse for better numerical stability
            delta_angles = np.linalg.pinv(J) @ error[:2]  # Only x,y for planar
            current_angles[:2] += delta_angles * 0.1  # Small step size

        return current_angles

class PhysicsSimulator:
    def __init__(self, gravity=np.array([0, -9.81, 0])):
        self.gravity = gravity
        self.objects = []
        self.contact_model = ContactModel()

    def add_rigid_body(self, body):
        """Add a rigid body to the simulation"""
        self.objects.append(body)

    def step_simulation(self, dt=0.01):
        """Step the physics simulation forward in time"""
        # Apply forces (gravity, etc.)
        for obj in self.objects:
            if hasattr(obj, 'apply_force'):
                # Apply gravity
                gravity_force = obj.mass * self.gravity
                obj.apply_force(gravity_force, dt)

        # Handle collisions
        for i in range(len(self.objects)):
            for j in range(i + 1, len(self.objects)):
                # Simple sphere-sphere collision detection
                if hasattr(self.objects[i], 'position') and hasattr(self.objects[j], 'position'):
                    pos1 = self.objects[i].position
                    pos2 = self.objects[j].position
                    dist = np.linalg.norm(pos2 - pos1)

                    if hasattr(self.objects[i], 'radius') and hasattr(self.objects[j], 'radius'):
                        if dist < self.objects[i].radius + self.objects[j].radius:
                            # Handle collision
                            vel1, vel2 = self.contact_model.sphere_sphere_collision(
                                pos1, self.objects[i].velocity, self.objects[i].radius, self.objects[i].mass,
                                pos2, self.objects[j].velocity, self.objects[j].radius, self.objects[j].mass
                            )
                            self.objects[i].velocity = vel1
                            self.objects[j].velocity = vel2