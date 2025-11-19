# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import sys
import numpy as np
from isaacsim import SimulationApp

# --- 1. ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument("--num-robots", type=int, default=1, help="Number of robots")
parser.add_argument("--num-gpus", type=int, default=None, help="Number of GPUs on machine.")
parser.add_argument("--non-headless", action="store_false", help="Run with GUI - nonheadless mode")
parser.add_argument("--viewport-updates", action="store_false", help="Enable viewport updates when headless")

args, unknown = parser.parse_known_args()

n_robot = args.num_robots
n_gpu = args.num_gpus
headless = args.non_headless
viewport_updates = args.viewport_updates


# --- 2. LAUNCH ISAAC SIM ---
simulation_app = SimulationApp(
    {"headless": headless, "max_gpu_count": n_gpu, "disable_viewport_updates": viewport_updates}
)

import carb
import omni
import omni.graph.core as og
from isaacsim.core.api import PhysicsContext
from isaacsim.core.utils.extensions import enable_extension
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.robot.wheeled_robots.robots import WheeledRobot
from pxr import Usd
from isaacsim.sensors.rtx import LidarRtx
from isaacsim.storage.native import get_assets_root_path # Needed to find assets
from isaacsim.core.utils.prims import get_prim_at_path

# --- 3. ROS 2 SETUP ---
# This line is critical and requires the environment fix
enable_extension("isaacsim.ros2.bridge")

# These imports are why the environment fix is necessary
import rclpy
from geometry_msgs.msg import Twist

# Generate Twist message
def move_cmd_msg(x, y, z, ax, ay, az):
    msg = Twist()
    msg.linear.x = x
    msg.linear.y = y
    msg.linear.z = z
    msg.angular.x = ax
    msg.angular.y = ay
    msg.angular.z = az
    return msg

# Wait for extensions to load
omni.kit.app.get_app().update()

# Create publisher for move commands
rclpy.init()
node = rclpy.create_node("cmd_vel_publisher")
cmd_vel_pub = node.create_publisher(Twist, "cmd_vel", 1)


# --- 4. SCENE SETUP ---
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    carb.log_error("Could not find Isaac Sim assets folder. Check Nucleus connection.")
    simulation_app.close()
    sys.exit()

robot_path = "/Isaac/Samples/ROS2/Robots/Nova_Carter_ROS.usd"
scene_path = "/Isaac/Environments/Simple_Warehouse/full_warehouse.usd"

# Load the stage
omni.usd.get_context().open_stage(assets_root_path + scene_path)

# Wait for stage to load
omni.kit.app.get_app().update()

with Usd.EditContext(get_current_stage(), get_current_stage().GetRootLayer()):
    get_current_stage().SetEndTimeCode(1000000.0)

stage = omni.usd.get_context().get_stage()
PhysicsContext(physics_dt=1.0 / 60.0)
set_camera_view(eye=[-6, -15.5, 6.5], target=[-6, 10.5, -1], camera_prim_path="/OmniverseKit_Persp")


robot_prim_path = "/World/Nova_Carter"
robot_usd_path = assets_root_path + robot_path
robot_position = np.array([0, 0, 0.1]) # Spawn at origin

my_carter = WheeledRobot(
    prim_path=robot_prim_path,
    name="my_carter",
    wheel_dof_names=["joint_wheel_left", "joint_wheel_right"],
    create_robot=True,
    usd_path=robot_usd_path,
    position=robot_position,
)

# This prim path exists inside the 'Nova_Carter_ROS.usd'
lidar_prim_path = robot_prim_path + "/chassis_link/sensors/XT_32/PandarXT_32_10hz"
my_lidar_vis = LidarRtx(
    prim_path=lidar_prim_path, 
    name="lidar_visualizer" # Simple name
)

omni.kit.app.get_app().update()

carb.settings.get_settings().set_bool("/exts/isaacsim.ros2.bridge/publish_without_verification", True)


# --- 6. START SIMULATION & CONTROL ---
timeline = omni.timeline.get_timeline_interface()
timeline.play()

# Wait a few frames for the world to be ready
for _ in range(10):
    omni.kit.app.get_app().update()

print("Initializing Lidar visualizer...")
my_lidar_vis.initialize()
my_lidar_vis.enable_visualization()
print("Lidar visualizer enabled.")

# Initialize the single robot
my_carter.initialize()
move_cmd = move_cmd_msg(1.0, 0.0, 0.0, 0.0, 0.0, 0.0)
cmd_vel_pub.publish(move_cmd)

omni.kit.app.get_app().update()

print("\n--- Simulation is running. Robot is spawned. Press Ctrl+C to exit. ---")

# --- 7. MAIN SIMULATION LOOP ---
try:
    while simulation_app.is_running():
        omni.kit.app.get_app().update()

except KeyboardInterrupt:
    print("Caught interrupt. Shutting down simulation.")


# --- 8. CLEANUP ---
node.destroy_node()
rclpy.shutdown()
timeline.stop()
simulation_app.close()
sys.exit()