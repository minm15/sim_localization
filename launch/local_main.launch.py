# launch/localization.launch.py

from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
        Node(
            package='sim_local',
            executable='localization_main',
            name='localization_main',
            output='screen',
            parameters=[{
                'dataset': 'nuscenes',   # or override on the command line
            }],
        ),
    ])
