---
title: URDF and XACRO Modeling
sidebar_position: 5
---

# URDF and XACRO Modeling

## Introduction: What is URDF and Why is it Essential for Humanoid Robots?

URDF (Unified Robot Description Format) is an XML-based format used to describe robot models in ROS. For Physical AI applications, particularly humanoid robotics, URDF serves as the foundational language for defining robot kinematics, dynamics, and visual properties. It allows developers to specify how different parts of a robot are connected, how they move relative to each other, and what they look like in simulation and visualization tools.

In humanoid robotics, URDF is essential because it provides a standardized way to represent the complex structure of human-like robots. Humanoid robots have multiple degrees of freedom, complex joint configurations, and detailed physical properties that must be accurately represented for effective simulation and control. URDF enables the precise definition of these properties, allowing for realistic simulation and accurate control algorithms.

The format supports the specification of:
- Links (rigid bodies) with visual, collision, and inertial properties
- Joints that connect links with specific degrees of freedom
- Transmission elements for actuator modeling
- Gazebo-specific plugins for simulation
- Sensors and their mounting positions

## Basic URDF Structure: Links, Joints, Transmissions, and Gazebo Plugins

### Link Elements

A link represents a rigid body in the robot. Each link can have multiple properties:

- **Visual**: How the link appears in visualization tools
- **Collision**: How the link interacts with the environment in simulation
- **Inertial**: Physical properties like mass and moment of inertia

### Joint Elements

Joints define the relationship between two links. Common joint types include:
- **Revolute**: Rotational joint with limited range
- **Continuous**: Rotational joint without limits
- **Prismatic**: Linear sliding joint
- **Fixed**: No movement between links
- **Floating**: 6 degrees of freedom

### Transmission Elements

Transmissions define how actuators connect to joints, specifying the mechanical relationship between motor commands and joint movement.

### Gazebo Plugins

Gazebo-specific plugins extend URDF functionality for simulation, including controllers, sensors, and custom behaviors.

## Step-by-Step Example: Simple Humanoid Arm

Let's build a simple humanoid arm model step by step:

```xml
<?xml version="1.0"?>
<robot name="simple_arm" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Base Link -->
  <link name="base_link">
    <visual>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
      <material name="blue">
        <color rgba="0 0 1 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.05" length="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Shoulder Joint -->
  <joint name="shoulder_joint" type="revolute">
    <parent link="base_link"/>
    <child link="upper_arm"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-1.57" upper="1.57" effort="100" velocity="1.0"/>
  </joint>

  <!-- Upper Arm Link -->
  <link name="upper_arm">
    <visual>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
      <material name="red">
        <color rgba="1 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.04" length="0.3"/>
      </geometry>
      <origin xyz="0 0 0.15" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="0.5"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Elbow Joint -->
  <joint name="elbow_joint" type="revolute">
    <parent link="upper_arm"/>
    <child link="forearm"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="0" upper="3.14" effort="100" velocity="1.0"/>
  </joint>

  <!-- Forearm Link -->
  <link name="forearm">
    <visual>
      <geometry>
        <cylinder radius="0.03" length="0.25"/>
      </geometry>
      <origin xyz="0 0 0.125" rpy="0 0 0"/>
      <material name="green">
        <color rgba="0 1 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.03" length="0.25"/>
      </geometry>
      <origin xyz="0 0 0.125" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="0.3"/>
      <inertia ixx="0.005" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
    </inertial>
  </link>
</robot>
```

## Introduction to XACRO: Macros, Parameters, and Reusability

XACRO (XML Macros) is an XML macro language that extends URDF with powerful features:

- **Macros**: Reusable components that can be instantiated multiple times
- **Parameters**: Variables that can be defined and reused throughout the model
- **Expressions**: Mathematical expressions that can be evaluated
- **File inclusion**: Ability to include other XACRO files

XACRO is essential for humanoid robot modeling because it allows for the creation of complex, reusable components that would be extremely difficult to manage in pure URDF.

## Converting URDF to XACRO for Complex Humanoids

For humanoid robots with multiple identical limbs, sensors, or other components, XACRO provides significant advantages over pure URDF by enabling parameterized, reusable definitions.

## Code Example 1: Basic Box Robot URDF

```xml
<?xml version="1.0"?>
<robot name="box_robot">
  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
      <material name="grey">
        <color rgba="0.5 0.5 0.5 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.5 0.3 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="1.0" ixy="0" ixz="0" iyy="1.0" iyz="0" izz="1.0"/>
    </inertial>
  </link>

  <!-- Simple wheel -->
  <link name="wheel">
    <visual>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.570796 0 0"/>
      <material name="black">
        <color rgba="0 0 0 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <cylinder radius="0.1" length="0.05"/>
      </geometry>
      <origin xyz="0 0 0" rpy="1.570796 0 0"/>
    </collision>
    <inertial>
      <mass value="1.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>

  <!-- Joint connecting wheel to base -->
  <joint name="wheel_joint" type="continuous">
    <parent link="base_link"/>
    <child link="wheel"/>
    <origin xyz="0.2 0 -0.1" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
  </joint>

  <!-- Gazebo plugin for differential drive -->
  <gazebo>
    <plugin name="differential_drive" filename="libgazebo_ros_diff_drive.so">
      <left_joint>wheel_joint</left_joint>
      <right_joint>wheel_joint</right_joint>
      <wheel_separation>0.3</wheel_separation>
      <wheel_diameter>0.2</wheel_diameter>
      <command_topic>cmd_vel</command_topic>
      <odometry_topic>odom</odometry_topic>
      <odometry_frame>odom</odometry_frame>
      <robot_base_frame>base_link</robot_base_frame>
    </plugin>
  </gazebo>
</robot>
```

## Code Example 2: Humanoid Torso with Joints

```xml
<?xml version="1.0"?>
<robot name="humanoid_torso">
  <!-- Base link (pelvis) -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.2 0.25 0.15"/>
      </geometry>
      <material name="body_color">
        <color rgba="0.8 0.8 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.2 0.25 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="5.0"/>
      <inertia ixx="0.1" ixy="0" ixz="0" iyy="0.1" iyz="0" izz="0.1"/>
    </inertial>
  </link>

  <!-- Spine joint -->
  <joint name="spine_joint" type="revolute">
    <parent link="base_link"/>
    <child link="torso"/>
    <origin xyz="0 0 0.075" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.5" upper="0.5" effort="100" velocity="1.0"/>
  </joint>

  <!-- Torso link -->
  <link name="torso">
    <visual>
      <geometry>
        <box size="0.3 0.3 0.4"/>
      </geometry>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
      <material name="body_color">
        <color rgba="0.8 0.8 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.3 0.4"/>
      </geometry>
      <origin xyz="0 0 0.2" rpy="0 0 0"/>
    </collision>
    <inertial>
      <mass value="8.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Neck joint -->
  <joint name="neck_joint" type="revolute">
    <parent link="torso"/>
    <child link="head"/>
    <origin xyz="0 0 0.4" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-0.5" upper="0.5" effort="50" velocity="1.0"/>
  </joint>

  <!-- Head link -->
  <link name="head">
    <visual>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
      <material name="head_color">
        <color rgba="0.9 0.9 0.9 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <sphere radius="0.1"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="2.0"/>
      <inertia ixx="0.01" ixy="0" ixz="0" iyy="0.01" iyz="0" izz="0.01"/>
    </inertial>
  </link>
</robot>
```

## Code Example 3: XACRO Macro for Legs

```xml
<?xml version="1.0"?>
<robot name="leg_macro_example" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <!-- Define parameters -->
  <xacro:property name="M_PI" value="3.1415926535897931" />
  <xacro:property name="leg_length" value="0.4" />
  <xacro:property name="leg_radius" value="0.05" />

  <!-- Leg macro definition -->
  <xacro:macro name="leg" params="name parent xyz rpy">
    <!-- Thigh link -->
    <link name="${name}_thigh">
      <visual>
        <geometry>
          <cylinder radius="${leg_radius}" length="${leg_length}"/>
        </geometry>
        <origin xyz="0 0 ${leg_length/2}" rpy="0 0 0"/>
        <material name="leg_color">
          <color rgba="0.7 0.7 0.7 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${leg_radius}" length="${leg_length}"/>
        </geometry>
        <origin xyz="0 0 ${leg_length/2}" rpy="0 0 0"/>
      </collision>
      <inertial>
        <mass value="2.0"/>
        <inertia ixx="0.05" ixy="0" ixz="0" iyy="0.05" iyz="0" izz="0.01"/>
      </inertial>
    </link>

    <!-- Hip joint -->
    <joint name="${name}_hip_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${name}_thigh"/>
      <origin xyz="${xyz}" rpy="${rpy}"/>
      <axis xyz="0 0 1"/>
      <limit lower="-1.57" upper="1.57" effort="200" velocity="1.0"/>
    </joint>

    <!-- Knee joint -->
    <joint name="${name}_knee_joint" type="revolute">
      <parent link="${name}_thigh"/>
      <child link="${name}_shin"/>
      <origin xyz="0 0 ${leg_length}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="0" upper="2.35" effort="200" velocity="1.0"/>
    </joint>

    <!-- Shin link -->
    <link name="${name}_shin">
      <visual>
        <geometry>
          <cylinder radius="${leg_radius}" length="${leg_length}"/>
        </geometry>
        <origin xyz="0 0 ${leg_length/2}" rpy="0 0 0"/>
        <material name="leg_color">
          <color rgba="0.7 0.7 0.7 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${leg_radius}" length="${leg_length}"/>
        </geometry>
        <origin xyz="0 0 ${leg_length/2}" rpy="0 0 0"/>
      </collision>
      <inertial>
        <mass value="1.5"/>
        <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.01"/>
      </inertial>
    </link>

    <!-- Ankle joint -->
    <joint name="${name}_ankle_joint" type="revolute">
      <parent link="${name}_shin"/>
      <child link="${name}_foot"/>
      <origin xyz="0 0 ${leg_length}" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="-0.785" upper="0.785" effort="100" velocity="1.0"/>
    </joint>

    <!-- Foot link -->
    <link name="${name}_foot">
      <visual>
        <geometry>
          <box size="0.2 0.1 0.05"/>
        </geometry>
        <material name="foot_color">
          <color rgba="0.3 0.3 0.3 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <box size="0.2 0.1 0.05"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.5"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Base link -->
  <link name="base_link">
    <visual>
      <geometry>
        <box size="0.3 0.4 0.2"/>
      </geometry>
      <material name="body_color">
        <color rgba="0.8 0.8 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.3 0.4 0.2"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="10.0"/>
      <inertia ixx="0.2" ixy="0" ixz="0" iyy="0.2" iyz="0" izz="0.2"/>
    </inertial>
  </link>

  <!-- Instantiate left leg using macro -->
  <xacro:leg name="left" parent="base_link" xyz="-0.1 -0.1 0" rpy="0 0 0"/>

  <!-- Instantiate right leg using macro -->
  <xacro:leg name="right" parent="base_link" xyz="-0.1 0.1 0" rpy="0 0 0"/>
</robot>
```

## Code Example 4: Full Humanoid Lower Body XACRO

```xml
<?xml version="1.0"?>
<robot name="humanoid_lower_body" xmlns:xacro="http://www.ros.org/wiki/xacro">
  <xacro:property name="M_PI" value="3.1415926535897931" />

  <!-- Define common properties -->
  <xacro:property name="body_density" value="1000" />
  <xacro:property name="joint_effort" value="200" />
  <xacro:property name="joint_velocity" value="1.0" />

  <!-- Macro for leg with parameters -->
  <xacro:macro name="humanoid_leg" params="side parent xyz">
    <xacro:property name="thigh_length" value="0.4" />
    <xacro:property name="thigh_radius" value="0.06" />
    <xacro:property name="shin_length" value="0.45" />
    <xacro:property name="shin_radius" value="0.05" />

    <!-- Hip joint (3 DOF) -->
    <joint name="${side}_hip_yaw_joint" type="revolute">
      <parent link="${parent}"/>
      <child link="${side}_hip_yaw_link"/>
      <origin xyz="${xyz}" rpy="0 0 0"/>
      <axis xyz="0 0 1"/>
      <limit lower="-0.785" upper="0.785" effort="${joint_effort}" velocity="${joint_velocity}"/>
    </joint>

    <link name="${side}_hip_yaw_link">
      <inertial>
        <mass value="0.1"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${side}_hip_roll_joint" type="revolute">
      <parent link="${side}_hip_yaw_link"/>
      <child link="${side}_hip_roll_link"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="-0.524" upper="0.524" effort="${joint_effort}" velocity="${joint_velocity}"/>
    </joint>

    <link name="${side}_hip_roll_link">
      <inertial>
        <mass value="0.1"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${side}_hip_pitch_joint" type="revolute">
      <parent link="${side}_hip_roll_link"/>
      <child link="${side}_thigh"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-2.094" upper="0.785" effort="${joint_effort}" velocity="${joint_velocity}"/>
    </joint>

    <!-- Thigh link -->
    <link name="${side}_thigh">
      <visual>
        <geometry>
          <cylinder radius="${thigh_radius}" length="${thigh_length}"/>
        </geometry>
        <origin xyz="0 0 ${thigh_length/2}" rpy="0 0 0"/>
        <material name="leg_color">
          <color rgba="0.7 0.7 0.7 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${thigh_radius}" length="${thigh_length}"/>
        </geometry>
        <origin xyz="0 0 ${thigh_length/2}" rpy="0 0 0"/>
      </collision>
      <inertial>
        <mass value="3.0"/>
        <inertia ixx="0.06" ixy="0" ixz="0" iyy="0.06" iyz="0" izz="0.01"/>
      </inertial>
    </link>

    <!-- Knee joint -->
    <joint name="${side}_knee_joint" type="revolute">
      <parent link="${side}_thigh"/>
      <child link="${side}_shin"/>
      <origin xyz="0 0 ${thigh_length}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="0" upper="2.356" effort="${joint_effort}" velocity="${joint_velocity}"/>
    </joint>

    <!-- Shin link -->
    <link name="${side}_shin">
      <visual>
        <geometry>
          <cylinder radius="${shin_radius}" length="${shin_length}"/>
        </geometry>
        <origin xyz="0 0 ${shin_length/2}" rpy="0 0 0"/>
        <material name="leg_color">
          <color rgba="0.7 0.7 0.7 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <cylinder radius="${shin_radius}" length="${shin_length}"/>
        </geometry>
        <origin xyz="0 0 ${shin_length/2}" rpy="0 0 0"/>
      </collision>
      <inertial>
        <mass value="2.0"/>
        <inertia ixx="0.04" ixy="0" ixz="0" iyy="0.04" iyz="0" izz="0.01"/>
      </inertial>
    </link>

    <!-- Ankle joints -->
    <joint name="${side}_ankle_pitch_joint" type="revolute">
      <parent link="${side}_shin"/>
      <child link="${side}_ankle"/>
      <origin xyz="0 0 ${shin_length}" rpy="0 0 0"/>
      <axis xyz="0 1 0"/>
      <limit lower="-0.524" upper="0.524" effort="${joint_effort}" velocity="${joint_velocity}"/>
    </joint>

    <link name="${side}_ankle">
      <inertial>
        <mass value="0.1"/>
        <inertia ixx="0.001" ixy="0" ixz="0" iyy="0.001" iyz="0" izz="0.001"/>
      </inertial>
    </link>

    <joint name="${side}_ankle_roll_joint" type="revolute">
      <parent link="${side}_ankle"/>
      <child link="${side}_foot"/>
      <origin xyz="0 0 0" rpy="0 0 0"/>
      <axis xyz="1 0 0"/>
      <limit lower="-0.349" upper="0.349" effort="${joint_effort}" velocity="${joint_velocity}"/>
    </joint>

    <!-- Foot link -->
    <link name="${side}_foot">
      <visual>
        <geometry>
          <box size="0.25 0.12 0.06"/>
        </geometry>
        <material name="foot_color">
          <color rgba="0.3 0.3 0.3 1"/>
        </material>
      </visual>
      <collision>
        <geometry>
          <box size="0.25 0.12 0.06"/>
        </geometry>
      </collision>
      <inertial>
        <mass value="0.8"/>
        <inertia ixx="0.002" ixy="0" ixz="0" iyy="0.005" iyz="0" izz="0.005"/>
      </inertial>
    </link>
  </xacro:macro>

  <!-- Pelvis link -->
  <link name="pelvis">
    <visual>
      <geometry>
        <box size="0.25 0.3 0.15"/>
      </geometry>
      <material name="body_color">
        <color rgba="0.8 0.8 0.8 1"/>
      </material>
    </visual>
    <collision>
      <geometry>
        <box size="0.25 0.3 0.15"/>
      </geometry>
    </collision>
    <inertial>
      <mass value="8.0"/>
      <inertia ixx="0.15" ixy="0" ixz="0" iyy="0.15" iyz="0" izz="0.15"/>
    </inertial>
  </link>

  <!-- Instantiate both legs -->
  <xacro:humanoid_leg side="left" parent="pelvis" xyz="0 -0.12 0"/>
  <xacro:humanoid_leg side="right" parent="pelvis" xyz="0 0.12 0"/>
</robot>
```

## Visualization with RViz and Gazebo

### Using RViz for URDF Visualization

To visualize your URDF model in RViz:

1. Start the robot state publisher:
```bash
ros2 run robot_state_publisher robot_state_publisher --ros-args -p robot_description:='$(cat your_model.urdf)'
```

2. Launch RViz:
```bash
ros2 run rviz2 rviz2
```

3. Add a RobotModel display and set the robot description topic

### Using Gazebo for Simulation

To spawn your robot in Gazebo:
```bash
ros2 run gazebo_ros spawn_entity.py -file your_model.urdf -entity robot_name
```

## Common Mistakes and Best Practices

### Common Mistakes

1. **Incorrect mass properties**: Setting masses too low or too high can cause simulation instability
2. **Poor inertial properties**: Using simplified inertial tensors can lead to unrealistic physics behavior
3. **Invalid joint limits**: Setting joint limits that are too restrictive or too permissive
4. **Collision mesh issues**: Using visual meshes for collision instead of simplified collision meshes
5. **Missing or incorrect origins**: Incorrect joint origins can cause kinematic errors

### Best Practices

1. **Use consistent units**: Always use SI units (meters, kilograms, seconds)
2. **Parameterize models**: Use XACRO properties to make models configurable
3. **Validate with tools**: Use `check_urdf` to validate your URDF files
4. **Separate visual and collision**: Use different meshes for visual and collision properties
5. **Realistic inertial properties**: Calculate or estimate realistic mass and inertial properties
6. **Modular design**: Break complex robots into reusable components using XACRO macros

## Exercises for Readers

### Exercise 1: Build a Simple Arm
Create a 3-DOF robotic arm using URDF. The arm should have a base, shoulder, elbow, and simple gripper. Test the model in RViz to ensure the kinematic chain is correct.

### Exercise 2: Convert URDF to XACRO
Take the simple arm model from Exercise 1 and convert it to XACRO format. Add parameters for link lengths and masses, and create a macro that allows for easy duplication of similar arms.

### Exercise 3: Create a Humanoid Head
Design a humanoid head with neck joints (pitch and yaw) and attach a simple camera sensor. Include proper visual and collision properties.

### Exercise 4: Parameterized Humanoid Model
Create a parameterized humanoid model using XACRO that allows for adjusting the overall size of the robot while maintaining proper proportions and joint limits.

## Visual Aids

<!-- Diagram 1: URDF Tree Structure -->
<!-- Caption: Hierarchical representation of robot links and joints showing parent-child relationships -->

<!-- Diagram 2: Joint Types -->
<!-- Caption: Visualization of different joint types (revolute, prismatic, continuous, fixed) and their degrees of freedom -->

<!-- Diagram 3: XACRO Inheritance -->
<!-- Caption: Example of how XACRO macros can be used to create reusable robot components -->

## Conclusion

URDF and XACRO form the foundation of robot modeling in ROS, providing the essential tools for creating accurate, reusable robot descriptions. For Physical AI applications, particularly humanoid robotics, mastering these formats is crucial for successful simulation, control, and deployment of robotic systems.

The combination of URDF's descriptive power and XACRO's macro capabilities enables the creation of complex, parameterized robot models that can be efficiently managed and extended. As you progress to more advanced topics in Physical AI, a solid understanding of these modeling tools will prove invaluable for creating realistic and functional robotic systems.

In the next chapters, we'll explore how to connect these models to controllers and integrate them with perception systems to create fully functional Physical AI applications.