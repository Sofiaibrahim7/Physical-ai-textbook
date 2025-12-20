import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import {useColorMode} from '@docusaurus/theme-common';
import styles from './HomepageFeatures.module.css';

const FeatureList = [
  {
    title: 'Master Physical AI',
    Svg: require('../../static/img/undraw_docusaurus_mountain.svg').default,
    description: (
      <>
        Build a strong foundation in Physical AI and embodied intelligence. Learn how AI systems sense, understand, and act in the real world using ROS 2, sensors, and humanoid control architectures.
      </>
    ),
    video: '/videos/digital-twin.mp4'
  },
  {
    title: 'Simulate Humanoids in 3D Worlds',
    Svg: require('../../static/img/undraw_docusaurus_react.svg').default,
    description: (
      <>
        Create high-fidelity digital twins using Gazebo and Unity. Simulate physics, collisions, LiDAR, depth cameras, and IMU sensors to test robots in realistic environments before deployment.
      </>
    ),
    video: '/videos/vla.mp4'
  },
  {
    title: 'Build Intelligent Robot Brains',
    Svg: require('../../static/img/undraw_docusaurus_tree.svg').default,
    description: (
      <>
        Integrate advanced AI using NVIDIA Isaac, VSLAM, and navigation systems. Combine Vision-Language-Action models with ROS 2 to enable natural language commands, perception, and autonomous decision-making.
      </>
    ),
    video: '/videos/robot-brain.mp4'
  },
];

function Feature({Svg, video, title, description}) {
  const {colorMode} = useColorMode();

  return (
    <div className={clsx('col col--4')}>
      <div className="text--center">
        {video ? (
          <video className="featureVideo_iBaM" autoPlay loop muted playsInline>
            <source src={video} type="video/mp4" />
            Your browser does not support the video tag.
          </video>
        ) : (
          <Svg className={styles.featureSvg} alt={title} />
        )}
      </div>
      <div className="padding-horiz--md">
        <h3>{title}</h3>
        <p>{description}</p>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}