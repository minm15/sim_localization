# launch/map.launch.py
import os

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sim_local',
            executable='map_main',  
            name='map_main',
            output='screen',
            parameters=[
                {'dataset': 'nclt'},
            ],
            remappings=[
                #('/LIDAR_TOP','/your_lidar_topic')
            ]
        )
    ])
