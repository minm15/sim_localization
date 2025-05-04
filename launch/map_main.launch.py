# launch/map.launch.py
import os

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sim_local',
            executable='map_combined_node',  
            name='map_combined',
            output='screen',
            parameters=[
                {'dataset': 'nuscenes'},
            ],
            remappings=[
                # ('/LIDAR_TOP','/your_lidar_topic')
            ]
        )
    ])
