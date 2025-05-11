from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

def generate_launch_description():
    root_arg = DeclareLaunchArgument(
        'root_path',
        default_value='./nclt_frame_desc/2012-01-15',
        description='Root directory under which a \"frames\" folder will be created'
    )

    return LaunchDescription([
        root_arg,
        Node(
            package='sim_local',
            executable='frame_dumper',
            name='frame_dumper',
            output='screen',
            parameters=[{'root_path': LaunchConfiguration('root_path')}]
        )
    ])
