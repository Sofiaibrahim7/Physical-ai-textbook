---
title: Synthetic Data Generation
sidebar_position: 7
---

# Synthetic Data Generation

## Introduction: Why Synthetic Data is Vital for Physical AI

Synthetic data generation has become a cornerstone of modern Physical AI development, addressing critical challenges that traditional data collection methods cannot solve. In the realm of robotics and embodied intelligence, real-world data collection is often expensive, time-consuming, and sometimes impossible to scale to the volumes required for training robust AI systems.

The scarcity of real-world data is particularly acute in robotics because:
- Physical robots are expensive to deploy at scale
- Real-world environments are difficult to control and reproduce
- Annotation of robot data requires significant manual effort
- Dangerous or rare scenarios cannot be safely replicated
- Privacy and safety concerns limit data collection in human environments

Synthetic data generation addresses these challenges by creating photorealistic, fully-annotated datasets in controlled virtual environments. This approach enables:
- Rapid generation of large, diverse datasets
- Complete control over environmental conditions
- Automatic annotation of complex scene properties
- Reproducible experiments with known ground truth
- Safe testing of edge cases and failure scenarios

## Tools for Synthetic Data Generation

### NVIDIA Isaac Sim Replicator
Isaac Sim Replicator is the most advanced tool for synthetic data generation in robotics, offering:
- Photorealistic rendering with RTX acceleration
- Advanced domain randomization capabilities
- Multi-sensor data capture (RGB, depth, segmentation, point clouds)
- Automatic annotation tools
- Integration with popular ML frameworks

### Gazebo Simulation
Gazebo provides synthetic data capabilities with:
- Realistic physics simulation
- Plugin-based sensor systems
- ROS integration for robotics workflows
- Lower computational requirements
- Open-source accessibility

### Unity ML-Agents
Unity's ML-Agents toolkit offers:
- Game engine-level rendering quality
- Flexible scene composition
- Reinforcement learning integration
- Cross-platform compatibility
- Extensive asset library

## Step-by-Step Workflows

### Domain Randomization Workflow

Domain randomization is a powerful technique that introduces controlled variations in simulation parameters to improve model robustness:

```python
import omni
from omni.isaac.core import World
from omni.isaac.core.utils.stage import add_reference_to_stage
import numpy as np
import random

def setup_domain_randomization():
    """
    Set up domain randomization parameters for synthetic data generation
    """
    # Define randomization ranges
    randomization_params = {
        "lighting": {
            "intensity_range": (300, 1500),
            "color_range": [(0.8, 0.8, 1.0), (1.0, 0.9, 0.8), (0.9, 1.0, 0.9)],
            "position_range": [(-5, -5, 5), (5, 5, 10)]
        },
        "materials": {
            "roughness_range": (0.1, 0.9),
            "metallic_range": (0.0, 0.5),
            "albedo_range": [(0.2, 0.2, 0.2), (0.8, 0.8, 0.8)]
        },
        "objects": {
            "position_range": [(-2, -2, 0), (2, 2, 2)],
            "rotation_range": (0, 360),
            "scale_range": (0.8, 1.2)
        }
    }

    return randomization_params

def apply_domain_randomization(randomization_params):
    """
    Apply domain randomization to the scene
    """
    # Randomize lighting
    light_intensity = random.uniform(
        randomization_params["lighting"]["intensity_range"][0],
        randomization_params["lighting"]["intensity_range"][1]
    )

    light_color = random.choice(randomization_params["lighting"]["color_range"])

    # Randomize materials
    roughness = random.uniform(
        randomization_params["materials"]["roughness_range"][0],
        randomization_params["materials"]["roughness_range"][1]
    )

    metallic = random.uniform(
        randomization_params["materials"]["metallic_range"][0],
        randomization_params["materials"]["metallic_range"][1]
    )

    # Apply randomizations to scene objects
    # Implementation details would depend on specific scene setup
    print(f"Applied domain randomization: intensity={light_intensity}, roughness={roughness}")
```

### Parametric Variation Workflow

Parametric variation systematically changes model parameters to generate diverse data:

```python
def generate_parametric_variations():
    """
    Generate systematic variations in robot poses and environments
    """
    # Define parameter spaces
    robot_poses = []
    for joint1 in np.linspace(-0.5, 0.5, 5):
        for joint2 in np.linspace(-0.3, 0.3, 5):
            for joint3 in np.linspace(-0.4, 0.4, 5):
                robot_poses.append([joint1, joint2, joint3])

    # Environment variations
    env_configs = []
    for floor_texture in ["wood", "tile", "carpet"]:
        for lighting_condition in ["bright", "dim", "overcast"]:
            for background_objects in [1, 2, 3]:
                env_configs.append({
                    "floor_texture": floor_texture,
                    "lighting": lighting_condition,
                    "objects": background_objects
                })

    return robot_poses, env_configs

def capture_data_with_variations(robot_poses, env_configs):
    """
    Capture synthetic data with parametric variations
    """
    for i, pose in enumerate(robot_poses):
        for j, env_config in enumerate(env_configs):
            # Apply robot pose
            set_robot_pose(pose)

            # Apply environment configuration
            configure_environment(env_config)

            # Capture data
            data = capture_synthetic_frame()

            # Save with metadata
            save_data_with_metadata(data, pose, env_config, f"data_{i}_{j}")
```

### Scene Composition Workflow

Scene composition involves creating complex environments with multiple objects:

```python
def compose_random_scenes(num_scenes=100):
    """
    Compose random scenes with multiple objects for synthetic data
    """
    # Define object library
    object_library = [
        {"name": "chair", "path": "/Objects/Chair.usd", "count_range": (1, 3)},
        {"name": "table", "path": "/Objects/Table.usd", "count_range": (1, 2)},
        {"name": "plant", "path": "/Objects/Plant.usd", "count_range": (0, 2)},
        {"name": "box", "path": "/Objects/Box.usd", "count_range": (1, 4)}
    ]

    for scene_idx in range(num_scenes):
        # Clear previous scene
        clear_scene()

        # Add objects randomly
        for obj_info in object_library:
            count = random.randint(
                obj_info["count_range"][0],
                obj_info["count_range"][1]
            )

            for obj_idx in range(count):
                # Random position
                x = random.uniform(-3, 3)
                y = random.uniform(-3, 3)
                z = random.uniform(0, 2)

                # Random rotation
                rotation = random.uniform(0, 360)

                # Add object to scene
                add_object_to_scene(
                    obj_info["path"],
                    position=(x, y, z),
                    rotation=rotation
                )

        # Add humanoid robot
        add_humanoid_robot()

        # Capture scene data
        capture_scene_data(f"scene_{scene_idx}")
```

## Generating Different Data Types

### RGB Images and Depth Maps

```python
from omni.isaac.sensor import Camera
import numpy as np
import cv2

def setup_rgb_depth_capture():
    """
    Set up RGB and depth capture in Isaac Sim
    """
    # Create camera
    camera = Camera(
        prim_path="/World/rgb_depth_camera",
        frequency=30,
        resolution=(640, 480)
    )

    # Set camera pose
    camera.set_world_pose(
        position=np.array([2, 0, 1.5]),
        orientation=np.array([0.707, 0, 0, 0.707])  # Looking at origin
    )

    return camera

def capture_rgb_depth_pair(camera, frame_idx):
    """
    Capture synchronized RGB and depth data
    """
    # Get RGB data
    rgb_data = camera.get_rgb()
    rgb_image = rgb_data["data"]

    # Get depth data
    depth_data = camera.get_depth()
    depth_image = depth_data["data"]

    # Save images
    cv2.imwrite(f"rgb_{frame_idx:06d}.png", cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))
    cv2.imwrite(f"depth_{frame_idx:06d}.png", depth_image)

    return rgb_image, depth_image
```

### Segmentation Masks

```python
def setup_segmentation_capture():
    """
    Set up semantic segmentation capture
    """
    from omni.isaac.synthetic_utils import SyntheticDataCapture

    sd_capture = SyntheticDataCapture(
        viewport_name="Viewport",
        semantic=True,
        instance=True
    )

    return sd_capture

def capture_segmentation_data(sd_capture, frame_idx):
    """
    Capture semantic and instance segmentation data
    """
    # Capture data
    data = sd_capture.capture()

    # Get semantic segmentation
    semantic_data = data["semantic"]

    # Get instance segmentation
    instance_data = data["instance"]

    # Save segmentation masks
    save_segmentation_mask(semantic_data, f"semantic_{frame_idx:06d}.png")
    save_segmentation_mask(instance_data, f"instance_{frame_idx:06d}.png")

    return semantic_data, instance_data
```

### Point Clouds and Poses

```python
def generate_point_cloud_from_depth(depth_image, camera_intrinsics):
    """
    Generate point cloud from depth image
    """
    height, width = depth_image.shape
    cx, cy = camera_intrinsics[0, 2], camera_intrinsics[1, 2]
    fx, fy = camera_intrinsics[0, 0], camera_intrinsics[1, 1]

    # Generate coordinate grids
    x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))

    # Convert to 3D coordinates
    x_3d = (x_grid - cx) * depth_image / fx
    y_3d = (y_grid - cy) * depth_image / fy
    z_3d = depth_image

    # Stack to form point cloud
    point_cloud = np.stack([x_3d, y_3d, z_3d], axis=-1)

    # Reshape to (N, 3)
    point_cloud = point_cloud.reshape(-1, 3)

    # Remove invalid points (depth = 0 or inf)
    valid_mask = (point_cloud[:, 2] > 0) & (np.isfinite(point_cloud).all(axis=1))
    point_cloud = point_cloud[valid_mask]

    return point_cloud

def capture_robot_poses():
    """
    Capture robot joint poses and end-effector positions
    """
    from omni.isaac.core.robots import Robot

    robot = Robot(prim_path="/World/humanoid")

    # Get joint positions
    joint_positions = robot.get_joint_positions()

    # Get end-effector pose (if applicable)
    ee_pose = robot.get_end_effector_pose()

    # Get robot base pose
    base_pose = robot.get_world_pose()

    return {
        "joint_positions": joint_positions,
        "end_effector_pose": ee_pose,
        "base_pose": base_pose
    }
```

## Integration with ML Pipelines

### PyTorch Dataset Integration

```python
import torch
from torch.utils.data import Dataset
import cv2
import numpy as np
import os
import json

class SyntheticRobotDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # Load metadata
        with open(os.path.join(data_dir, "metadata.json"), "r") as f:
            self.metadata = json.load(f)

        self.data_files = [f for f in os.listdir(data_dir) if f.endswith(".png")]

    def __len__(self):
        return len(self.data_files) // 2  # RGB + depth pairs

    def __getitem__(self, idx):
        # Load RGB image
        rgb_path = os.path.join(self.data_dir, f"rgb_{idx:06d}.png")
        rgb_image = cv2.imread(rgb_path)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)

        # Load depth image
        depth_path = os.path.join(self.data_dir, f"depth_{idx:06d}.png")
        depth_image = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH)

        # Load segmentation
        seg_path = os.path.join(self.data_dir, f"semantic_{idx:06d}.png")
        seg_image = cv2.imread(seg_path, cv2.IMREAD_GRAYSCALE)

        # Apply transforms
        if self.transform:
            rgb_image = self.transform(rgb_image)
            depth_image = self.transform(depth_image.astype(np.float32))

        # Get annotations from metadata
        annotations = self.metadata.get(f"frame_{idx:06d}", {})

        sample = {
            "rgb": torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0,
            "depth": torch.from_numpy(depth_image).unsqueeze(0).float(),
            "segmentation": torch.from_numpy(seg_image).long(),
            "annotations": annotations
        }

        return sample

def create_data_loader(data_dir, batch_size=32, shuffle=True):
    """
    Create PyTorch data loader for synthetic robot data
    """
    dataset = SyntheticRobotDataset(data_dir)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=4,
        pin_memory=True
    )

    return data_loader
```

## Practical Examples

### Example 1: Simple Replicator Script for Randomizing Lights/Objects

```python
import omni
from omni.replicator.core import Replicator
import omni.replicator.core.random as rand
import numpy as np

def setup_replicator_randomization():
    """
    Set up Isaac Sim Replicator for randomizing lights and objects
    """
    # Initialize replicator
    replicator = Replicator()

    # Define randomization functions
    def randomize_lighting():
        # Randomize light intensity
        intensity = rand.uniform(300, 1500)

        # Randomize light color
        color = rand.uniform([0.7, 0.7, 0.7], [1.0, 1.0, 1.0])

        # Apply to lights in scene
        # Implementation depends on light setup

    def randomize_objects():
        # Randomize object positions
        for obj_path in ["/World/Box", "/World/Cylinder", "/World/Sphere"]:
            pos_x = rand.uniform(-2.0, 2.0)
            pos_y = rand.uniform(-2.0, 2.0)
            pos_z = rand.uniform(0.1, 1.0)

            # Apply random position
            replicator.get_node(obj_path).set_local_pos([pos_x, pos_y, pos_z])

    # Register randomization functions
    replicator.randomizer.add_randomization(
        randomize_lighting,
        frequency="frame"
    )

    replicator.randomizer.add_randomization(
        randomize_objects,
        frequency="frame"
    )

    return replicator

def generate_randomized_data(replicator, num_frames=1000):
    """
    Generate data with replicator randomization
    """
    # Start replication
    replicator.setup_camera("/World/Camera")

    for frame in range(num_frames):
        # Step simulation
        omni.timeline.get_timeline_interface().step(1)

        # Capture data
        rgb_data = replicator.get_rgb_data()
        depth_data = replicator.get_depth_data()

        # Save data
        save_frame_data(rgb_data, depth_data, frame)

        print(f"Generated frame {frame+1}/{num_frames}")
```

### Example 2: Generating Annotated Images in Isaac Sim

```python
from omni.isaac.synthetic_utils import SyntheticDataCapture
import numpy as np

def generate_annotated_dataset(num_images=1000):
    """
    Generate fully annotated dataset using Isaac Sim
    """
    # Initialize synthetic data capture
    sd_capture = SyntheticDataCapture(
        viewport_name="Viewport",
        rgb=True,
        depth=True,
        semantic=True,
        instance=True,
        bbox_2d_tight=True,
        bbox_2d_loose=True
    )

    # Set up scene with humanoid robot
    setup_scene_with_humanoid()

    for i in range(num_images):
        # Randomize scene
        randomize_scene()

        # Capture all data types
        data = sd_capture.capture()

        # Save RGB image
        save_image(data["rgb"], f"images/rgb_{i:06d}.png")

        # Save depth image
        save_image(data["depth"], f"images/depth_{i:06d}.png")

        # Save semantic segmentation
        save_image(data["semantic"], f"labels/semantic_{i:06d}.png")

        # Save instance segmentation
        save_image(data["instance"], f"labels/instance_{i:06d}.png")

        # Save bounding boxes
        bbox_data = data["bbox_2d_tight"]
        save_bounding_boxes(bbox_data, f"labels/bbox_{i:06d}.json")

        print(f"Generated annotated image {i+1}/{num_images}")
```

### Example 3: Exporting Data to HDF5/CSV

```python
import h5py
import pandas as pd
import numpy as np

def export_data_to_hdf5(data_dir, output_file, num_samples):
    """
    Export synthetic data to HDF5 format
    """
    with h5py.File(output_file, 'w') as f:
        # Create datasets
        rgb_dataset = f.create_dataset(
            'rgb_images',
            (num_samples, 480, 640, 3),
            dtype='uint8',
            compression='gzip'
        )

        depth_dataset = f.create_dataset(
            'depth_maps',
            (num_samples, 480, 640),
            dtype='float32',
            compression='gzip'
        )

        poses_dataset = f.create_dataset(
            'robot_poses',
            (num_samples, 7),  # Position (3) + Orientation (4)
            dtype='float32'
        )

        # Fill datasets
        for i in range(num_samples):
            # Load and store RGB image
            rgb_img = load_image(f"{data_dir}/rgb_{i:06d}.png")
            rgb_dataset[i] = rgb_img

            # Load and store depth map
            depth_map = load_depth_map(f"{data_dir}/depth_{i:06d}.png")
            depth_dataset[i] = depth_map

            # Load and store robot pose
            pose = load_robot_pose(f"{data_dir}/poses_{i:06d}.json")
            poses_dataset[i] = pose

def export_metadata_to_csv(data_dir, output_file):
    """
    Export metadata to CSV format
    """
    metadata_list = []

    for i in range(1000):  # Assuming 1000 samples
        # Load metadata for this sample
        metadata = load_sample_metadata(f"{data_dir}/metadata_{i:06d}.json")
        metadata['sample_id'] = i
        metadata['rgb_path'] = f"rgb_{i:06d}.png"
        metadata['depth_path'] = f"depth_{i:06d}.png"

        metadata_list.append(metadata)

    # Create DataFrame and save to CSV
    df = pd.DataFrame(metadata_list)
    df.to_csv(output_file, index=False)
```

### Example 4: Gazebo Plugin for Data Capture

```cpp
#include <gazebo/gazebo.hh>
#include <gazebo/sensors/sensors.hh>
#include <gazebo/physics/physics.hh>
#include <opencv2/opencv.hpp>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>

namespace gazebo
{
  class SyntheticDataCapturePlugin : public SensorPlugin
  {
    public: void Load(sensors::SensorPtr _sensor, sdf::ElementPtr _sdf)
    {
      // Get the camera sensor
      this->cameraSensor =
        std::dynamic_pointer_cast<sensors::CameraSensor>(_sensor);

      if (!this->cameraSensor)
      {
        gzerr << "Not a camera sensor\n";
        return;
      }

      // Connect to sensor update event
      this->updateConnection = this->cameraSensor->ConnectUpdated(
          std::bind(&SyntheticDataCapturePlugin::OnUpdate, this));

      // Initialize OpenCV image
      this->img = cv::Mat(this->cameraSensor->ImageHeight(),
                         this->cameraSensor->ImageWidth(),
                         CV_8UC3);
    }

    private: void OnUpdate()
    {
      // Get image data
      const unsigned char *data = this->cameraSensor->ImageData();

      // Copy to OpenCV image
      memcpy(this->img.data, data,
             this->cameraSensor->ImageHeight() *
             this->cameraSensor->ImageWidth() * 3);

      // Apply randomization effects (if needed)
      this->applyRandomization();

      // Save image with timestamp
      std::string filename = "synthetic_" +
                            std::to_string(this->frameCount) + ".png";
      cv::imwrite(filename, this->img);

      this->frameCount++;
    }

    private: void applyRandomization()
    {
      // Add noise, blur, or other effects
      cv::Mat noisy_img;
      cv::randn(this->img, noisy_img, cv::Scalar(0), cv::Scalar(10));
      this->img += noisy_img;
    }

    private: sensors::CameraSensorPtr cameraSensor;
    private: event::ConnectionPtr updateConnection;
    private: cv::Mat img;
    private: int frameCount = 0;
  };

  // Register plugin
  GZ_REGISTER_SENSOR_PLUGIN(SyntheticDataCapturePlugin)
}
```

### Example 5: Training a Simple Vision Model on Synthetic Data

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision.models import resnet18

class SimpleVisionModel(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleVisionModel, self).__init__()
        self.backbone = resnet18(pretrained=False)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

def train_on_synthetic_data(data_dir, num_epochs=10):
    """
    Train a simple vision model on synthetic data
    """
    # Define transforms
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # Create dataset and dataloader
    dataset = SyntheticRobotDataset(data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    # Initialize model
    model = SimpleVisionModel(num_classes=10)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_idx, batch in enumerate(dataloader):
            # Get data
            images = batch['rgb'].to(device)
            labels = batch['annotations']['class_labels'].to(device)

            # Zero gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if batch_idx % 100 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}')

        print(f'Epoch {epoch} completed, Average Loss: {running_loss/len(dataloader):.4f}')

    # Save model
    torch.save(model.state_dict(), 'synthetic_vision_model.pth')
    return model
```

## Ethical Considerations

### Bias Mitigation
Synthetic data generation can inadvertently introduce biases that affect model performance:
- **Environmental bias**: Over-representation of certain lighting conditions or textures
- **Pose bias**: Limited range of robot poses or movements
- **Object bias**: Uneven distribution of objects or scenarios

To mitigate these biases:
- Ensure diverse domain randomization parameters
- Monitor data distribution during generation
- Validate model performance across different conditions
- Combine synthetic and real data when possible

### Sim-to-Real Gap
The sim-to-real gap refers to performance differences between simulation and reality:
- **Visual differences**: Lighting, textures, and rendering differences
- **Physics differences**: Simulation inaccuracies compared to real physics
- **Sensor differences**: Simulated vs. real sensor characteristics

Strategies to bridge the gap:
- Use photo-realistic rendering
- Implement sensor noise models
- Apply domain adaptation techniques
- Fine-tune on real data when available

## Exercises for Readers

### Exercise 1: Generate 500 Varied Humanoid Poses
Create a script that generates 500 synthetic images of a humanoid robot in various poses. Use domain randomization to vary the environment, lighting, and camera positions. Ensure the poses cover the full range of motion for the robot's joints.

### Exercise 2: Fine-tune on Mixed Real/Synthetic Data
Train a perception model on synthetic data, then fine-tune it on a small set of real robot data. Compare the performance of models trained only on synthetic data, only on real data, and the mixed approach. Analyze the trade-offs in terms of data requirements and performance.

### Exercise 3: Synthetic Data Quality Assessment
Develop metrics to assess the quality of synthetic data for your specific application. Create a validation pipeline that measures how well synthetic data mimics real-world conditions and how effectively models trained on synthetic data transfer to real scenarios.

### Exercise 4: Large-Scale Dataset Generation
Design and implement a pipeline for generating a large-scale synthetic dataset (10,000+ images) with proper organization, metadata, and quality control. Include error handling and progress tracking for long-running generation processes.

## Visual Aids

<!-- Diagram 1: Data Generation Pipeline -->
<!-- Caption: End-to-end pipeline from 3D scene setup to ML-ready datasets -->

<!-- Diagram 2: Domain Randomization Examples -->
<!-- Caption: Examples of how domain randomization changes scene appearance and properties -->

<!-- Diagram 3: Sim-to-Real Transfer -->
<!-- Caption: Process of transferring models trained on synthetic data to real-world applications -->

## Conclusion

Synthetic data generation has become an indispensable tool for Physical AI development, enabling the creation of large-scale, diverse, and fully-annotated datasets that would be impossible to collect in the real world. As we advance in 2025, the quality and realism of synthetic data continue to improve, making it increasingly valuable for training robust robotic systems.

The combination of advanced simulation platforms like Isaac Sim, sophisticated domain randomization techniques, and efficient ML pipeline integration provides researchers and developers with powerful tools for accelerating Physical AI development. However, careful attention must be paid to bias mitigation and the sim-to-real gap to ensure that models trained on synthetic data perform effectively in real-world applications.

In the next chapters, we'll explore how synthetic data generation integrates with perception systems and how to validate the effectiveness of synthetic training data for real-world deployment.