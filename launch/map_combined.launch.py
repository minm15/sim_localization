# launch/map_combined.launch.py
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    return LaunchDescription([
      Node(
        package='sim_local',
        executable='map_combined_node',
        name='map_combined',
        output='screen',
        # no parameters needed any more
      )
    ])
