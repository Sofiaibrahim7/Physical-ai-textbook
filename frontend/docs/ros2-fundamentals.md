---
title: ROS 2 Fundamentals
sidebar_position: 3
---

# ROS 2 Fundamentals

## Introduction: Why ROS 2 for Physical AI and Humanoid Robots

Robot Operating System 2 (ROS 2) has become the de facto standard for developing complex robotic systems, particularly in the realm of Physical AI and humanoid robotics. Unlike its predecessor ROS 1, ROS 2 is built on modern middleware technologies that make it suitable for real-world deployment, safety-critical applications, and commercial robots.

For Physical AI applications, ROS 2 provides the essential infrastructure for connecting perception, planning, and control systems. Humanoid robots require sophisticated coordination between multiple subsystems: sensors, actuators, perception modules, motion planning, and high-level decision-making. ROS 2's distributed architecture enables these components to communicate efficiently while maintaining modularity and scalability.

ROS 2's real-time capabilities, improved security features, and support for multiple operating systems make it ideal for humanoid robotics applications where reliability and performance are paramount. The middleware's Quality of Service (QoS) policies allow developers to fine-tune communication behavior for different types of data, ensuring critical control messages are delivered with appropriate priority and reliability.

## Core Concepts: Nodes, Topics, Services, Actions, and Parameters

### Nodes: The Building Blocks of ROS 2

A ROS 2 node is a process that performs computation. Nodes are the fundamental building blocks of a ROS 2 system, each typically responsible for a specific task. In a humanoid robot, you might have nodes for sensor processing, motion control, perception, planning, and user interaction.

Nodes are organized into packages, which contain source code, launch files, and other resources. This modular approach promotes code reuse and makes complex robotic systems more manageable.

### Topics: Publish/Subscribe Communication

Topics enable asynchronous communication through a publish/subscribe model. Publishers send messages to topics, and subscribers receive messages from topics. This decoupled communication pattern allows for flexible system architectures where components can be added or removed without affecting others.

In Physical AI, topics are commonly used for:
- Sensor data (LIDAR scans, camera images, IMU readings)
- Robot state information (joint positions, odometry)
- Control commands (velocity commands, joint trajectories)

### Services: Request/Response Communication

Services provide synchronous request/response communication. A client sends a request to a service and waits for a response. This pattern is ideal for operations that require immediate results, such as changing robot parameters or requesting specific computations.

### Actions: Goal-Based Communication

Actions extend the service model to support long-running operations with feedback. An action client sends a goal to an action server, which provides feedback during execution and returns a result upon completion. This is perfect for complex robot behaviors like navigation, manipulation, or motion planning.

### Parameters: Configuration Management

Parameters allow nodes to be configured at runtime. They provide a way to adjust node behavior without recompiling code, making systems more flexible and easier to tune for different environments or applications.

## Understanding DDS and Why ROS 2 is Superior to ROS 1

ROS 2 is built on Data Distribution Service (DDS), a middleware standard designed for real-time, distributed systems. DDS provides several advantages over the custom middleware used in ROS 1:

- **Real-time performance**: DDS is designed for time-critical applications with deterministic behavior
- **Scalability**: Better handling of large numbers of nodes and messages
- **Security**: Built-in security features for protecting robotic systems
- **Cross-platform compatibility**: Robust support across different operating systems
- **Quality of Service (QoS)**: Configurable policies for reliability, durability, and performance

These improvements make ROS 2 suitable for production robotics, where ROS 1 was primarily a research tool. For humanoid robots operating in human environments, these features are essential for safety and reliability.

## Installation Instructions for ROS 2

### Installing ROS 2 Humble Hawksbill (Recommended for 2025)

ROS 2 Humble Hawksbill is the current LTS (Long Term Support) version, making it ideal for production systems and long-term projects.

#### Ubuntu 22.04 Installation:

```bash
# Set locale
locale  # check for UTF-8
sudo locale-gen en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8

# Add ROS 2 apt repository
sudo apt update && sudo apt install -y curl gnupg lsb-release
sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg

echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(source /etc/os-release && echo $UBUNTU_CODENAME) main" | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null

# Install ROS 2 packages
sudo apt update
sudo apt install ros-humble-desktop

# Install additional packages for robotics development
sudo apt install python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool build-essential

# Source ROS 2 environment
echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc
source ~/.bashrc
```

#### Windows Installation:

1. Install Visual Studio Community 2019 or 2022
2. Install Python 3.8 or later
3. Download and run the ROS 2 installer from the official website
4. Follow the installation wizard

#### macOS Installation:

1. Install Homebrew if not already installed
2. Install ROS 2 using Homebrew:
```bash
brew install ros/humble/ros-desktop
```

## Code Example 1: Simple Publisher Node

```python
#!/usr/bin/env python3

"""
Simple publisher node that sends messages to a topic.
This example demonstrates the basic structure of a ROS 2 publisher.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import time

class MinimalPublisher(Node):
    def __init__(self):
        super().__init__('minimal_publisher')

        # Create a publisher for the 'topic' topic
        self.publisher_ = self.create_publisher(String, 'topic', 10)

        # Create a timer to publish messages at 10 Hz
        timer_period = 0.1  # seconds
        self.timer = self.create_timer(timer_period, self.timer_callback)

        # Counter for message numbering
        self.i = 0

        self.get_logger().info('Publisher node initialized')

    def timer_callback(self):
        """Callback function that publishes messages at regular intervals"""
        msg = String()
        msg.data = f'Hello World: {self.i}'

        # Publish the message
        self.publisher_.publish(msg)

        # Log the published message
        self.get_logger().info(f'Publishing: "{msg.data}"')

        self.i += 1


def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create the publisher node
    minimal_publisher = MinimalPublisher()

    try:
        # Spin the node to keep it alive
        rclpy.spin(minimal_publisher)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        minimal_publisher.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Code Example 2: Simple Subscriber Node

```python
#!/usr/bin/env python3

"""
Simple subscriber node that receives messages from a topic.
This example demonstrates the basic structure of a ROS 2 subscriber.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import String

class MinimalSubscriber(Node):
    def __init__(self):
        super().__init__('minimal_subscriber')

        # Create a subscription to the 'topic' topic
        self.subscription = self.create_subscription(
            String,
            'topic',
            self.listener_callback,
            10)

        # Make sure the subscription is properly created
        self.subscription  # prevent unused variable warning

        self.get_logger().info('Subscriber node initialized')

    def listener_callback(self, msg):
        """Callback function that processes incoming messages"""
        # Log the received message
        self.get_logger().info(f'I heard: "{msg.data}"')


def main(args=None):
    # Initialize ROS 2
    rclpy.init(args=args)

    # Create the subscriber node
    minimal_subscriber = MinimalSubscriber()

    try:
        # Spin the node to keep it alive and process callbacks
        rclpy.spin(minimal_subscriber)
    except KeyboardInterrupt:
        pass
    finally:
        # Clean up
        minimal_subscriber.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
```

## Code Example 3: Service Server and Client

```python
#!/usr/bin/env python3

"""
Service server and client example.
This demonstrates how to create and use ROS 2 services.
"""

import rclpy
from rclpy.node import Node
from example_interfaces.srv import AddTwoInts


class MinimalService(Node):
    def __init__(self):
        super().__init__('minimal_service')

        # Create a service that responds to AddTwoInts requests
        self.srv = self.create_service(
            AddTwoInts,
            'add_two_ints',
            self.add_two_ints_callback)

        self.get_logger().info('Service server initialized')

    def add_two_ints_callback(self, request, response):
        """Callback function that processes service requests"""
        # Perform the computation
        response.sum = request.a + request.b

        # Log the request and response
        self.get_logger().info(f'Incoming request\na: {request.a}, b: {request.b}')
        self.get_logger().info(f'Sending response: {response.sum}')

        return response


class MinimalClientAsync(Node):
    def __init__(self):
        super().__init__('minimal_client_async')

        # Create a client for the AddTwoInts service
        self.cli = self.create_client(AddTwoInts, 'add_two_ints')

        # Wait for the service to be available
        while not self.cli.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('Service not available, waiting again...')

        self.req = AddTwoInts.Request()

        self.get_logger().info('Service client initialized')

    def send_request(self, a, b):
        """Send a request to the service"""
        self.req.a = a
        self.req.b = b

        # Call the service asynchronously
        self.future = self.cli.call_async(self.req)
        rclpy.spin_until_future_complete(self, self.future)

        return self.future.result()


def main_server(args=None):
    """Main function for the service server"""
    rclpy.init(args=args)

    minimal_service = MinimalService()

    try:
        rclpy.spin(minimal_service)
    except KeyboardInterrupt:
        pass
    finally:
        minimal_service.destroy_node()
        rclpy.shutdown()


def main_client(args=None):
    """Main function for the service client"""
    rclpy.init(args=args)

    minimal_client = MinimalClientAsync()

    # Send a request
    response = minimal_client.send_request(41, 1)

    # Print the result
    minimal_client.get_logger().info(f'Result of add_two_ints: {response.sum}')

    # Clean up
    minimal_client.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        main_server()
    elif len(sys.argv) > 1 and sys.argv[1] == 'client':
        main_client()
    else:
        print("Usage: python script.py [server|client]")
```

## Code Example 4: Action Server and Client

```python
#!/usr/bin/env python3

"""
Action server and client example.
This demonstrates how to create and use ROS 2 actions for long-running operations.
"""

import rclpy
from rclpy.action import ActionServer, ActionClient
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup

from example_interfaces.action import Fibonacci


class FibonacciActionServer(Node):
    def __init__(self):
        super().__init__('fibonacci_action_server')

        # Create an action server
        self._action_server = ActionServer(
            self,
            Fibonacci,
            'fibonacci',
            self.execute_callback)

        self.get_logger().info('Action server initialized')

    def execute_callback(self, goal_handle):
        """Execute callback for the action server"""
        self.get_logger().info('Executing goal...')

        # Create the result message
        feedback_msg = Fibonacci.Feedback()
        feedback_msg.sequence = [0, 1]

        # Simulate the long-running operation
        for i in range(1, goal_handle.request.order):
            # Check if the goal has been canceled
            if goal_handle.is_cancel_requested:
                goal_handle.canceled()
                self.get_logger().info('Goal canceled')
                result = Fibonacci.Result()
                result.sequence = feedback_msg.sequence
                return result

            # Update the sequence
            feedback_msg.sequence.append(
                feedback_msg.sequence[i] + feedback_msg.sequence[i-1])

            # Publish feedback
            goal_handle.publish_feedback(feedback_msg)
            self.get_logger().info(f'Publishing feedback: {feedback_msg.sequence}')

        # Complete the goal
        goal_handle.succeed()
        result = Fibonacci.Result()
        result.sequence = feedback_msg.sequence

        self.get_logger().info(f'Returning result: {result.sequence}')
        return result


class FibonacciActionClient(Node):
    def __init__(self):
        super().__init__('fibonacci_action_client')

        # Create an action client
        self._action_client = ActionClient(
            self,
            Fibonacci,
            'fibonacci')

        self.get_logger().info('Action client initialized')

    def send_goal(self, order):
        """Send a goal to the action server"""
        # Wait for the action server to be available
        self._action_client.wait_for_server()

        # Create the goal message
        goal_msg = Fibonacci.Goal()
        goal_msg.order = order

        # Send the goal asynchronously
        self._send_goal_future = self._action_client.send_goal_async(
            goal_msg,
            feedback_callback=self.feedback_callback)

        # Add a callback for when the goal is accepted
        self._send_goal_future.add_done_callback(self.goal_response_callback)

    def goal_response_callback(self, future):
        """Callback for when the goal response is received"""
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().info('Goal rejected')
            return

        self.get_logger().info('Goal accepted')

        # Get the result asynchronously
        self._get_result_future = goal_handle.get_result_async()
        self._get_result_future.add_done_callback(self.get_result_callback)

    def get_result_callback(self, future):
        """Callback for when the result is received"""
        result = future.result().result
        self.get_logger().info(f'Result: {result.sequence}')

        # Shutdown after receiving the result
        rclpy.shutdown()

    def feedback_callback(self, feedback_msg):
        """Callback for receiving feedback"""
        feedback = feedback_msg.feedback
        self.get_logger().info(f'Received feedback: {feedback.sequence}')


def main_server(args=None):
    """Main function for the action server"""
    rclpy.init(args=args)

    fibonacci_action_server = FibonacciActionServer()

    try:
        rclpy.spin(fibonacci_action_server)
    except KeyboardInterrupt:
        pass
    finally:
        fibonacci_action_server.destroy_node()
        rclpy.shutdown()


def main_client(args=None):
    """Main function for the action client"""
    rclpy.init(args=args)

    fibonacci_action_client = FibonacciActionClient()

    # Send a goal
    fibonacci_action_client.send_goal(10)

    try:
        rclpy.spin(fibonacci_action_client)
    except KeyboardInterrupt:
        pass
    finally:
        fibonacci_action_client.destroy_node()


if __name__ == '__main__':
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == 'server':
        main_server()
    elif len(sys.argv) > 1 and sys.argv[1] == 'client':
        main_client()
    else:
        print("Usage: python script.py [server|client]")
```

## Troubleshooting Tips for Common Issues

### 1. Node Discovery Issues
- Ensure all nodes are on the same ROS domain (use `ROS_DOMAIN_ID` environment variable)
- Check network configuration for multi-machine setups
- Verify that `localhost` is properly configured in `/etc/hosts`

### 2. Permission and Environment Issues
- Source the ROS 2 setup file in each new terminal: `source /opt/ros/humble/setup.bash`
- Use `rosdep` to install missing dependencies: `rosdep install --from-paths src --ignore-src -r -y`
- Check that Python packages are properly installed in your environment

### 3. Build Issues
- Use `colcon build` in your workspace root directory
- Clean builds if needed: `rm -rf build/ install/ log/`
- Check that CMakeLists.txt and package.xml are properly configured

### 4. Message/Service Type Issues
- Ensure custom message packages are built before using them
- Check that message/service definitions are properly exported
- Verify that the correct message types are being used in publishers/subscribers

## Exercises for Readers

### Exercise 1: Create a Talker-Listener Pair
Create a publisher node that publishes temperature readings and a subscriber node that receives and processes these readings. The subscriber should log warnings when temperatures exceed a threshold (e.g., 30°C). Test the nodes by running them in separate terminals.

### Exercise 2: Build a Simple Service
Create a service that converts temperatures between Celsius and Fahrenheit. Implement both the server and client nodes. The server should handle requests to convert temperatures in both directions. Test the service with various temperature values.

### Exercise 3: Implement an Action for Robot Movement
Create an action that moves a simulated robot to a specified position. The action should provide feedback on the robot's progress and return the final position. Implement both the action server and client nodes.

### Exercise 4: Parameter Server Integration
Modify one of your nodes to use parameters instead of hardcoded values. Create a launch file that sets these parameters at startup. Experiment with changing parameters at runtime using the `ros2 param` command.

## Visual Aids

<!-- Diagram 1: ROS 2 Computation Graph -->
<!-- Caption: Visualization of nodes, topics, services, and actions in a ROS 2 system -->

<!-- Diagram 2: Publish/Subscribe Model -->
<!-- Caption: Illustration of the publish/subscribe communication pattern with publishers, topics, and subscribers -->

<!-- Diagram 3: Service Call Flow -->
<!-- Caption: Step-by-step visualization of service request and response communication -->

<!-- Diagram 4: Action Feedback Loop -->
<!-- Caption: Detailed view of action communication including goals, feedback, and results -->

## Preview of Next Topics

In the upcoming chapters, we'll explore how ROS 2 integrates with other essential tools for Physical AI development:

- **URDF/XACRO**: Learn to model robot structures and define kinematic chains for humanoid robots
- **Gazebo Simulation**: Master simulation environments for testing and developing robotic behaviors
- **Navigation and Motion Planning**: Understand how to plan and execute robot movements in complex environments
- **Perception Systems**: Build vision and sensor processing pipelines using ROS 2

ROS 2 provides the essential communication infrastructure that connects all these components, making it the backbone of modern robotic systems. Mastering these fundamentals will enable you to build sophisticated Physical AI applications that can operate effectively in real-world environments.