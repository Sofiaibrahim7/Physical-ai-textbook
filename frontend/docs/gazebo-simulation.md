---
title: Gazebo Simulation
sidebar_position: 4
---

# Gazebo Simulation

## Introduction: Why Gazebo is Crucial for Physical AI Development

Gazebo has become the cornerstone of robotics simulation, providing a realistic physics engine and sensor models essential for developing and testing Physical AI systems. For humanoid robots and other complex robotic platforms, Gazebo offers the ability to test algorithms, validate control strategies, and train AI systems in a safe, controlled environment before deployment on real hardware.

The transition from traditional Gazebo Classic to the modern Ignition Gazebo (now called Gazebo Harmonic as of 2025) represents a significant evolution in simulation technology. This new architecture provides better performance, modularity, and maintainability, making it ideal for the demanding requirements of Physical AI research and development.

Simulation is particularly important for Physical AI because it allows researchers and developers to:
- Test complex behaviors without risk of hardware damage
- Create diverse environmental conditions for robustness testing
- Generate large amounts of training data for machine learning algorithms
- Validate multi-robot systems and coordination strategies
- Debug control algorithms in a controlled setting

## Gazebo Classic vs. Ignition Gazebo (Harmonic)

### Gazebo Classic
Gazebo Classic was the traditional version of the simulator that served the robotics community for many years. While still functional, it had limitations in terms of modularity and maintainability. It used a monolithic architecture that made it difficult to extend and maintain.

### Ignition Gazebo (Modern Gazebo)
The modern Ignition Gazebo, now known as Gazebo Harmonic, features a modular architecture with separate components for physics, rendering, sensors, and transport. This design allows for:
- Better performance through parallel processing
- More flexible plugin architecture
- Improved rendering capabilities
- Better integration with ROS 2

The transition to Gazebo Harmonic brings significant improvements in physics accuracy, rendering quality, and overall simulation fidelity, making it the preferred choice for Physical AI applications.

## Installation: ROS 2 Humble with Gazebo Harmonic

### Ubuntu 22.04 with ROS 2 Humble Installation

First, ensure you have ROS 2 Humble installed (as covered in the previous chapter). Then install Gazebo Harmonic:

```bash
# Update package lists
sudo apt update

# Install Gazebo Harmonic
sudo apt install ros-humble-gazebo-ros ros-humble-gazebo-plugins ros-humble-gazebo-dev

# Install additional packages for simulation
sudo apt install ros-humble-ros-gz ros-humble-ros-gz-bridge ros-humble-gz-ros-pkgs

# Verify installation
gz --version
```

### Docker Installation (Alternative)

For a consistent environment, you can also use Docker with pre-built Gazebo Harmonic images:

```bash
# Pull the latest Gazebo Harmonic image
docker pull gzsim/harmonic

# Run Gazebo with GUI support
xhost +local:docker
docker run -it --rm -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix gzsim/harmonic gz sim
```

## Creating a Simple World

### Basic Empty World SDF

SDF (Simulation Description Format) is the XML-based format used by Gazebo to describe simulation worlds. Here's a basic empty world file:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="empty_world">
    <!-- Include the default sun -->
    <include>
      <uri>model://sun</uri>
    </include>

    <!-- Include the default ground plane -->
    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Physics engine configuration -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
    </physics>

    <!-- GUI configuration -->
    <gui fullscreen="0">
      <camera name="user_camera">
        <pose>5 -5 2 0 0.4 1.5707</pose>
        <view_controller>orbit</view_controller>
      </camera>
    </gui>
  </world>
</sdf>
```

Save this as `empty_world.sdf` in your simulation directory.

### Advanced World with Objects

Here's a more complex world file with additional objects and lighting:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <world name="physical_ai_world">
    <!-- Include default models -->
    <include>
      <uri>model://sun</uri>
    </include>

    <include>
      <uri>model://ground_plane</uri>
    </include>

    <!-- Custom objects -->
    <model name="box_obstacle">
      <pose>2 0 0.5 0 0 0</pose>
      <link name="box_link">
        <visual name="visual">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
          <material>
            <ambient>1 0 0 1</ambient>
            <diffuse>1 0 0 1</diffuse>
          </material>
        </visual>
        <collision name="collision">
          <geometry>
            <box>
              <size>1 1 1</size>
            </box>
          </geometry>
        </collision>
        <inertial>
          <mass>1.0</mass>
          <inertia>
            <ixx>0.166667</ixx>
            <ixy>0</ixy>
            <ixz>0</ixz>
            <iyy>0.166667</iyy>
            <iyz>0</iyz>
            <izz>0.166667</izz>
          </inertia>
        </inertial>
      </link>
    </model>

    <!-- Physics configuration -->
    <physics name="1ms" type="ode">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
      <real_time_update_rate>1000.0</real_time_update_rate>
      <ode>
        <solver>
          <type>quick</type>
          <iters>10</iters>
          <sor>1.3</sor>
        </solver>
        <constraints>
          <cfm>0.0</cfm>
          <erp>0.2</erp>
          <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
          <contact_surface_layer>0.001</contact_surface_layer>
        </constraints>
      </ode>
    </physics>

    <!-- GUI configuration -->
    <gui fullscreen="0">
      <plugin filename="GzScene3D" name="3D View">
        <engine>ogre</engine>
        <scene>scene</scene>
        <ambient_light>0.4 0.4 0.4</ambient_light>
        <background_color>0.8 0.8 0.8</background_color>
        <camera_pose>6 -8 3 0 0.5 1.57</camera_pose>
      </plugin>
    </gui>
  </world>
</sdf>
```

## Spawning a Robot in Simulation

### Simple Robot Model SDF

Here's an example of a simple humanoid robot model that can be spawned in Gazebo:

```xml
<?xml version="1.0" ?>
<sdf version="1.7">
  <model name="simple_humanoid">
    <!-- Main body -->
    <link name="base_link">
      <pose>0 0 0.5 0 0 0</pose>
      <inertial>
        <mass>10.0</mass>
        <inertia>
          <ixx>0.4</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.4</iyy>
          <iyz>0</iyz>
          <izz>0.4</izz>
        </inertia>
      </inertial>

      <visual name="visual">
        <geometry>
          <box>
            <size>0.5 0.3 0.8</size>
          </box>
        </geometry>
        <material>
          <ambient>0.8 0.8 0.8 1</ambient>
          <diffuse>0.8 0.8 0.8 1</diffuse>
        </material>
      </visual>

      <collision name="collision">
        <geometry>
          <box>
            <size>0.5 0.3 0.8</size>
          </box>
        </geometry>
      </collision>
    </link>

    <!-- Head -->
    <link name="head">
      <pose>0 0 0.9 0 0 0</pose>
      <inertial>
        <mass>2.0</mass>
        <inertia>
          <ixx>0.01</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.01</iyy>
          <iyz>0</iyz>
          <izz>0.01</izz>
        </inertia>
      </inertial>

      <visual name="head_visual">
        <geometry>
          <sphere>
            <radius>0.15</radius>
          </sphere>
        </geometry>
        <material>
          <ambient>0.7 0.7 0.7 1</ambient>
          <diffuse>0.7 0.7 0.7 1</diffuse>
        </material>
      </visual>

      <collision name="head_collision">
        <geometry>
          <sphere>
            <radius>0.15</radius>
          </sphere>
        </geometry>
      </collision>
    </link>

    <!-- Joint connecting head to body -->
    <joint name="neck_joint" type="revolute">
      <parent>base_link</parent>
      <child>head</child>
      <axis>
        <xyz>0 0 1</xyz>
        <limit>
          <lower>-0.5</lower>
          <upper>0.5</upper>
          <effort>100</effort>
          <velocity>1.0</velocity>
        </limit>
      </axis>
      <pose>0 0 0.4 0 0 0</pose>
    </joint>

    <!-- Legs -->
    <link name="left_leg">
      <pose>-0.15 0 0.1 0 0 0</pose>
      <inertial>
        <mass>3.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.1</iyy>
          <iyz>0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>

      <visual name="left_leg_visual">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.6</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.5 0.5 0.5 1</diffuse>
        </material>
      </visual>

      <collision name="left_leg_collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.6</length>
          </cylinder>
        </geometry>
      </collision>
    </link>

    <joint name="left_hip" type="revolute">
      <parent>base_link</parent>
      <child>left_leg</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>100</effort>
          <velocity>1.0</velocity>
        </limit>
      </axis>
      <pose>-0.15 0 -0.3 0 0 0</pose>
    </joint>

    <!-- Right leg -->
    <link name="right_leg">
      <pose>0.15 0 0.1 0 0 0</pose>
      <inertial>
        <mass>3.0</mass>
        <inertia>
          <ixx>0.1</ixx>
          <ixy>0</ixy>
          <ixz>0</ixz>
          <iyy>0.1</iyy>
          <iyz>0</iyz>
          <izz>0.1</izz>
        </inertia>
      </inertial>

      <visual name="right_leg_visual">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.6</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.5 0.5 0.5 1</ambient>
          <diffuse>0.5 0.5 0.5 1</diffuse>
        </material>
      </visual>

      <collision name="right_leg_collision">
        <geometry>
          <cylinder>
            <radius>0.05</radius>
            <length>0.6</length>
          </cylinder>
        </geometry>
      </collision>
    </link>

    <joint name="right_hip" type="revolute">
      <parent>base_link</parent>
      <child>right_leg</child>
      <axis>
        <xyz>0 1 0</xyz>
        <limit>
          <lower>-1.57</lower>
          <upper>1.57</upper>
          <effort>100</effort>
          <velocity>1.0</velocity>
        </limit>
      </axis>
      <pose>0.15 0 -0.3 0 0 0</pose>
    </joint>
  </model>
</sdf>
```

## Adding Sensors to Your Robot

### Camera Sensor Configuration

Here's how to add a camera sensor to your robot model:

```xml
<link name="camera_link">
  <pose>0 0 0.7 0 0 0</pose>
  <inertial>
    <mass>0.1</mass>
    <inertia>
      <ixx>0.001</ixx>
      <ixy>0</ixy>
      <ixz>0</ixz>
      <iyy>0.001</iyy>
      <iyz>0</iyz>
      <izz>0.001</izz>
    </inertia>
  </inertial>

  <visual name="camera_visual">
    <geometry>
      <box>
        <size>0.05 0.05 0.05</size>
      </box>
    </geometry>
    <material>
      <ambient>0 0 1 1</ambient>
      <diffuse>0 0 1 1</diffuse>
    </material>
  </visual>

  <collision name="camera_collision">
    <geometry>
      <box>
        <size>0.05 0.05 0.05</size>
      </box>
    </geometry>
  </collision>
</link>

<sensor name="camera" type="camera">
  <pose>0 0 0 0 0 0</pose>
  <camera>
    <horizontal_fov>1.047</horizontal_fov>
    <image>
      <width>640</width>
      <height>480</height>
      <format>R8G8B8</format>
    </image>
    <clip>
      <near>0.1</near>
      <far>100</far>
    </clip>
  </camera>
  <always_on>1</always_on>
  <update_rate>30</update_rate>
  <visualize>true</visualize>
</sensor>
```

### LIDAR Sensor Configuration

Here's an example of adding a LIDAR sensor:

```xml
<sensor name="lidar" type="gpu_lidar">
  <pose>0 0 0.8 0 0 0</pose>
  <topic>scan</topic>
  <ray>
    <scan>
      <horizontal>
        <samples>360</samples>
        <resolution>1</resolution>
        <min_angle>-3.14159</min_angle>
        <max_angle>3.14159</max_angle>
      </horizontal>
    </scan>
    <range>
      <min>0.1</min>
      <max>30.0</max>
      <resolution>0.01</resolution>
    </range>
  </ray>
  <always_on>1</always_on>
  <update_rate>10</update_rate>
  <visualize>true</visualize>
</sensor>
```

### IMU Sensor Configuration

Here's how to add an IMU sensor:

```xml
<sensor name="imu" type="imu">
  <pose>0 0 0.3 0 0 0</pose>
  <topic>imu</topic>
  <always_on>1</always_on>
  <update_rate>100</update_rate>
  <imu>
    <angular_velocity>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>2e-4</stddev>
        </noise>
      </z>
    </angular_velocity>
    <linear_acceleration>
      <x>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </x>
      <y>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </y>
      <z>
        <noise type="gaussian">
          <mean>0.0</mean>
          <stddev>1.7e-2</stddev>
        </noise>
      </z>
    </linear_acceleration>
  </imu>
</sensor>
```

## Physics Settings and Realism for Humanoid Robots

### Physics Configuration for Humanoid Simulation

For realistic humanoid robot simulation, you need to carefully configure the physics parameters:

```xml
<physics name="realistic_humanoid_physics" type="ode">
  <max_step_size>0.001</max_step_size>
  <real_time_factor>1.0</real_time_factor>
  <real_time_update_rate>1000.0</real_time_update_rate>
  <ode>
    <solver>
      <type>quick</type>
      <iters>100</iters>
      <sor>1.3</sor>
    </solver>
    <constraints>
      <cfm>0.000001</cfm>
      <erp>0.2</erp>
      <contact_max_correcting_vel>100.0</contact_max_correcting_vel>
      <contact_surface_layer>0.001</contact_surface_layer>
    </constraints>
  </ode>
</physics>
```

### Friction and Contact Properties

For realistic humanoid walking, proper friction settings are crucial:

```xml
<surface>
  <friction>
    <ode>
      <mu>1.0</mu>
      <mu2>1.0</mu2>
      <fdir1>0 0 0</fdir1>
      <slip1>0</slip1>
      <slip2>0</slip2>
    </ode>
    <torsional>
      <coefficient>1.0</coefficient>
      <use_patch_radius>1</use_patch_radius>
      <surface_radius>0.01</surface_radius>
      <ode>
        <slip>0.0</slip>
      </ode>
    </torsional>
  </friction>
  <contact>
    <ode>
      <soft_cfm>0</soft_cfm>
      <soft_erp>0.2</soft_erp>
      <kp>1e+13</kp>
      <kd>1</kd>
      <max_vel>100.0</max_vel>
      <min_depth>0.001</min_depth>
    </ode>
  </contact>
</surface>
```

## ROS 2 Integration Launch File

### Complete Launch File for Gazebo + ROS 2

Here's a launch file that integrates Gazebo with ROS 2:

```python
import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Launch configuration variables
    world = LaunchConfiguration('world')
    use_sim_time = LaunchConfiguration('use_sim_time', default='true')
    headless = LaunchConfiguration('headless', default='false')

    # Declare launch arguments
    world_arg = DeclareLaunchArgument(
        'world',
        default_value='empty',
        description='Choose one of the world files from `/gazebo_ros_pkgs/gazebo_ros/worlds`'
    )

    # Include Gazebo launch
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([
                FindPackageShare('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            ])
        ]),
        launch_arguments={
            'world': world,
            'headless': headless,
            'verbose': 'false',
        }.items()
    )

    # Spawn robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', 'simple_humanoid',
            '-x', '0',
            '-y', '0',
            '-z', '1.0'
        ],
        output='screen'
    )

    # Robot state publisher
    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        name='robot_state_publisher',
        output='screen',
        parameters=[{
            'use_sim_time': use_sim_time,
        }]
    )

    # Joint state publisher
    joint_state_publisher = Node(
        package='joint_state_publisher',
        executable='joint_state_publisher',
        name='joint_state_publisher',
        parameters=[{
            'use_sim_time': use_sim_time,
        }]
    )

    return LaunchDescription([
        world_arg,
        gazebo,
        robot_state_publisher,
        joint_state_publisher,
        spawn_entity,
    ])
```

## Exercises for Readers

### Exercise 1: Spawn a Humanoid Model
Create a more detailed humanoid model with at least 6 degrees of freedom (legs, arms, head) and spawn it in a Gazebo world. Add basic controllers to move the joints and verify that the model behaves realistically in the simulation environment.

### Exercise 2: Add Noise to Sensors
Modify the camera and LIDAR sensors to include realistic noise models that simulate real-world sensor imperfections. Compare the performance of your algorithms with and without sensor noise to understand the impact on perception and control systems.

### Exercise 3: Create a Walking Environment
Design a Gazebo world with obstacles, uneven terrain, and different surface materials. Test your humanoid robot's ability to navigate this environment and implement basic path planning and obstacle avoidance behaviors.

### Exercise 4: Performance Optimization
Experiment with different physics settings and rendering options to optimize simulation performance. Measure the real-time factor and identify bottlenecks in your simulation setup. Implement techniques to improve performance while maintaining simulation accuracy.

## Troubleshooting Common Issues

### Plugin Loading Errors
If you encounter plugin loading errors:
1. Verify that all required packages are installed: `sudo apt install ros-humble-gazebo-*`
2. Check that your Gazebo plugins path is set correctly: `echo $GAZEBO_PLUGIN_PATH`
3. Ensure ROS 2 packages are properly sourced: `source /opt/ros/humble/setup.bash`

### Performance Issues on Low-End Hardware
For systems with limited resources:
1. Reduce physics update rate in your world file
2. Disable unnecessary visualizations
3. Use simpler collision meshes
4. Consider using CPU-based rendering instead of GPU rendering

### Sensor Data Issues
If sensor data appears incorrect:
1. Check sensor topics: `ros2 topic list | grep sensor`
2. Verify sensor configuration in your robot model
3. Ensure proper coordinate frame transformations
4. Test sensor output with simple visualization tools

## Visual Aids

<!-- Diagram 1: Gazebo GUI Overview -->
<!-- Caption: Layout and interface elements of the Gazebo simulation environment -->

<!-- Diagram 2: Simulation Pipeline -->
<!-- Caption: Data flow from robot model to physics simulation to sensor output -->

<!-- Diagram 3: Sensor Data Flow -->
<!-- Caption: How sensor data flows from Gazebo through ROS 2 to your algorithms -->

## Preview of Next Topics

In the upcoming chapters, we'll explore how to create detailed robot models:

- **URDF/XACRO**: Learn to create sophisticated robot descriptions with proper kinematics and dynamics
- **Controller Integration**: Understand how to connect your robot controllers with Gazebo simulation
- **Simulation Testing**: Master the art of validating your algorithms in simulated environments
- **Hardware-in-the-Loop**: Bridge the gap between simulation and real hardware

Gazebo provides the essential simulation infrastructure that allows you to test and validate your Physical AI systems before deploying them on real robots. Mastering these simulation techniques will significantly accelerate your development process and improve the robustness of your robotic systems.