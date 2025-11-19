# In: main.py (at the root of your project)

import sys
import os
import argparse
import numpy as np
import math
from isaacsim import SimulationApp
import torch
import numpy as np


# --- 1. ARGUMENT PARSING ---
parser = argparse.ArgumentParser()
parser.add_argument("--headless", action="store_true", help="Run in headless mode (no GUI)")
args, unknown = parser.parse_known_args()

# --- 2. LAUNCH ISAAC SIM ---
simulation_app = SimulationApp({"headless": args.headless, "enable_motion_bvh": True})

# --- 3. IMPORTS ---
import carb
import omni
from isaacsim.core.api import PhysicsContext
from isaacsim.core.utils.stage import get_current_stage
from isaacsim.core.utils.viewports import set_camera_view
from isaacsim.storage.native import get_assets_root_path
from pxr import Usd, Gf, UsdGeom


# --- 4. IMPORTS ---
from navigation_stack.robot.nova_carter import NovaCarter
from navigation_stack.controllers.mppi_controller import MPPIController
from navigation_stack.controllers.simple_controller import SimpleController
from navigation_stack.planner.global_planner import GlobalPlanner
from navigation_stack.perception.stvl_stem import STVL_System
from navigation_stack.utils.height_map_processor import HeightMapProcessor
from isaacsim.core.utils.rotations import quat_to_euler_angles
from omni.isaac.core.utils.extensions import enable_extension
from isaacsim.core.api import World

from omni.isaac.core.objects import VisualCuboid
enable_extension("omni.isaac.debug_draw")
from isaacsim.util.debug_draw import _debug_draw


def get_yaw_from_quat(quat_xyzw):
    """
    Converts a quaternion (x, y, z, w) to a 2D yaw angle (theta).
    Uses Isaac Sim's official conversion function.
    """
    # Isaac Sim's function returns [roll, pitch, yaw] in radians
    euler_angles = quat_to_euler_angles(quat_xyzw)
    yaw = euler_angles[2]  # Extract yaw (Z-axis rotation)
    
    return yaw




class NavigationSimulator:
    """
    The main class that orchestrates the entire simulation.
    """
    def __init__(self, simulation_context):
        self.simulation_app = simulation_context
        self.world = None
        
        self.stage = None 
        self.physics_context = PhysicsContext()

        # --- Define Path ---
        self.START_POS = np.array([0, 0, 0.1])
        self.GOAL_POS = (5.0, 5.0)
        self.waypoint_list = []
        self.current_waypoint_idx = 0
        self.goal_threshold = 0.5  # (meters) How close to get to a waypoint

        # --- Get Lidar Transform ---
        # We will get this after the robot is spawned
        self.lidar_to_base_pos = None
        self.lidar_to_base_quat = None


        # --- Initialize all our modules ---
        print("Initializing navigation modules...")
        self.global_planner = GlobalPlanner()
        # Define our grid properties
        self.height_map_processor = HeightMapProcessor(
            grid_resolution=0.05,  # 5cm per cell
            grid_width_m=30.0,     # 40m wide map
            grid_height_m=30.0,    # 40m tall map
            robot_height=2.5,      # Max obstacle height to care about (2m)
            robot_ground_clearance=0.1 # Ignore floor (5cm)
        )
        # self.mppi_controller = MPPIController()
        self.mppi_controller = SimpleController()

        self.stvl = STVL_System()
        
        self.robot = NovaCarter(
            robot_prim_path="/World/Nova_Carter",
            robot_name="my_carter",
            position=self.START_POS
        )
        
        
        print("Initialization complete.")

    def create_debug_cube(self, prim_path, position, size, color):
        """Helper to create a single, colored, transformed cube prim."""

        # 1. Define a Cube prim at the specified path
        # This single prim will have its own shape and transform
        cube_prim = UsdGeom.Cube.Define(self.stage, prim_path)

        # 2. Set its size
        cube_prim.CreateSizeAttr(size)

        # 3. Set its color
        # This fixes the Vec3f(tuple) bug by "unpacking" the tuple
        r, g, b = color
        cube_prim.GetDisplayColorAttr().Set([Gf.Vec3f(r, g, b)])

        # 4. Set its position
        # We add a transform operation (Translate) to this prim
        # This also fixes the Vec3f(numpy) bug by "unpacking" the array
        pos_tuple = (position[0], position[1], position[2])
        cube_prim.AddTranslateOp().Set(Gf.Vec3f(*pos_tuple))

    def visualize_lidar_points(self, points_world, color=(0.0, 1.0, 0.0)):
        """
        Visualizes points that are ALREADY in the world frame.
        """
        if points_world.size == 0:
            return

        draw = _debug_draw.acquire_debug_draw_interface()

        step = 1 # Sample ~2000 points
        sampled_points = points_world[::step]
        point_list = [tuple(point) for point in sampled_points]

        colors = [(color[0], color[1], color[2], 1.0)] * len(point_list)
        sizes = [5] * len(point_list)

        draw.draw_points(point_list, colors, sizes)
        print(f"   ✓ Drew {len(point_list)} points (sampled from {len(points_world)})")

    def clear_debug_drawing(self):
        """Clear all debug draw visualizations"""
        draw = _debug_draw.acquire_debug_draw_interface()
        draw.clear_points()
        draw.clear_lines()

    def setup_simulation(self):
        """Spawns all assets, generates the path, and initializes the robot."""
        print("Setting up simulation scene...")
        
        # --- 5. SCENE SETUP: ALIGN WAREHOUSE TO MAP ORIGIN ---
        assets_root_path = get_assets_root_path()
        if assets_root_path is None:
            carb.log_error("Could not find Isaac Sim assets folder.")
            return

       
            
        scene_path = assets_root_path + "/Isaac/Environments/Simple_Warehouse/warehouse.usd"
        omni.usd.get_context().open_stage(scene_path)
        self.simulation_app.update()
        # Get the map's origin from the planner
        # This is the coordinate system our planner expects.
        
        omni.usd.get_context().open_stage(scene_path)
        self.simulation_app.update()
        
        self.world = World(stage_units_in_meters=1.0)
        
        # 3. Get the stage and context from the new, valid world
        self.stage = self.world.stage
        self.physics_context = self.world.get_physics_context()
        
        # 4. Reset the new world
        self.world.reset()

        if not self.stage:
            carb.log_error("Failed to get stage after opening. Exiting.")
            self.simulation_app.close()
            return
        
        
        # --- 6. GLOBAL PLANNER ---
        print(f"Planning global path from {self.START_POS[:2]} to {self.GOAL_POS}...")
        path, grid_path = self.global_planner.plan_path(self.START_POS[:2], self.GOAL_POS)
        # if not path:
        #     carb.log_error("Failed to generate a global path. Exiting.")
        #     self.simulation_app.close()
        #     return
        self.waypoint_list = path
        # # print("Using SIMPLE STRAIGHT PATH for testing...")
        # # Create a straight line from (0,0) to (5,5)
        # num_waypoints = 11
        # self.waypoint_list = []
        # for i in range(num_waypoints):
        #     t = i / (num_waypoints - 1)
        #     x = 0.0 + t * 5.0  # 0 to 5
        #     y = 0.0 + t * 5.0  # 0 to 5
        #     self.waypoint_list.append((x, y))

        print(f"Created {len(self.waypoint_list)} waypoints:")
        for i, wp in enumerate(self.waypoint_list):
            print(f"  [{i}]: {wp}")
        print(f"Global path found with {len(self.waypoint_list)} waypoints.")
        print("\n=== PATH DEBUG ===")
        print(f"Start pose: {self.START_POS[:2]}")
        print(f"Goal pose: {self.GOAL_POS}")
        print(f"First 3 waypoints from A*:")
        for i in range(min(3, len(self.waypoint_list))):
            print(f"  [{i}]: {self.waypoint_list[i]}")
        print(f"Last 3 waypoints:")
        for i in range(max(0, len(self.waypoint_list)-3), len(self.waypoint_list)):
            print(f"  [{i}]: {self.waypoint_list[i]}")
        print("==================\n")

        # --- SAVE PATH VISUALIZATION ---
        # (This is optional, but great for debugging)
        try:
            output_dir = os.path.join(os.getcwd(), "maps")
            os.makedirs(output_dir, exist_ok=True)
            output_image_path = os.path.join(output_dir, "path_visualization.png")
            self.global_planner.save_path_image(grid_path, output_image_path)
            print(f"Saved path visualization to {output_image_path}")
        except Exception as e:
            print(f"Could not save path visualization: {e}")
        # -------------------------------
        
        print("Visualizing waypoints in simulator...")
        UsdGeom.Xform.Define(self.stage, "/World/Debug")
        self.simulation_app.update()
        
        # # 1. Giant GREEN Cube at the START
        # start_pos_3d = np.array([self.START_POS[0], self.START_POS[1], 1.0])
        # self.create_debug_cube("/World/Debug/Waypoint_Start", 
        #                        start_pos_3d, 
        #                        size=2.0, 
        #                        color=(0.0, 1.0, 0.0)) # Green

        # # 2. Giant RED Cube at the FINAL GOAL
        # goal_pos_3d = np.array([self.GOAL_POS[0], self.GOAL_POS[1], 1.0])
        # self.create_debug_cube("/World/Debug/Waypoint_Goal", 
        #                        goal_pos_3d, 
        #                        size=2.0, 
        #                        color=(1.0, 0.0, 0.0)) # Red
        
        # # 3. Spawn a small blue cube at EACH waypoint
        # print(f"Spawning {len(self.waypoint_list)} blue waypoint cubes...")
        # for i, waypoint in enumerate(self.waypoint_list):
        #     waypoint_pos_3d = np.array([waypoint[0], waypoint[1], 0.2]) # 20cm high
        #     prim_path = f"/World/Debug/Waypoint_Cube_{i}"
        #     self.create_debug_cube(prim_path, 
        #                            waypoint_pos_3d, 
        #                            size=0.1, # 10cm cube
        #                            color=(0.0, 0.5, 1.0)) # Blue
                               
        # --- 7. CREATE ROBOT ---
        # Note: The robot's START_POS is already in the correct map coordinates
        self.robot.spawn()
        self.simulation_app.update()
        
        # --- 8. START SIMULATION & INITIALIZE ---
        self.timeline = omni.timeline.get_timeline_interface()
        self.timeline.play()
        
        # Short pause to let physics settle
        for _ in range(10):
            self.simulation_app.update()
            
        self.robot.initialize()
        self.simulation_app.update()
        
        # --- 9. GET LIDAR TRANSFORM ---
        print("Fetching Lidar-to-Base transform...")
        transform = self.robot.get_lidar_to_base_transform()
        if transform:
            self.lidar_to_base_pos = transform[0]
            self.lidar_to_base_quat = transform[1]
            print(f"  Lidar Pos: {self.lidar_to_base_pos}")
            print(f"  Lidar Quat: {self.lidar_to_base_quat}")
        else:
            carb.log_error("Could not get Lidar transform. Obstacle avoidance will fail.")
        
        
        # New camera view: Set to look at the START_POS
        set_camera_view(
    eye=[6.0, -12.0, 8.5],       # Corner view, elevated
    target=[2.5, 2.5, 0.0],      
    camera_prim_path="/OmniverseKit_Persp"
)
        
        print("Spawning test cube at [3, 0, 0.5]...")
        self.world.scene.add(
            VisualCuboid(
                prim_path="/World/TestCube",
                name="test_cube",
                position=np.array([-3.0, 0.0, 0.5]), # 3m in front, 0.5m up
                size=1.0, # 1-meter cube
                color=np.array([1.0, 0.0, 0.0]) # Red
            )
        )


        print("\n--- Simulation is running. Robot and Humans are spawned. ---")


    def get_current_target(self, current_pose_2d):
        """
        Manages the list of waypoints.
        Returns the current target and advances to the next if we're close.
        """
        if self.current_waypoint_idx >= len(self.waypoint_list):
            return None # We have reached the final goal

        target_waypoint = self.waypoint_list[self.current_waypoint_idx]
        
        # Check distance to the current target
        dist_to_target = np.linalg.norm(current_pose_2d[:2] - target_waypoint)
        
        if dist_to_target < self.goal_threshold:
            # print(f"Reached waypoint {self.current_waypoint_idx}: {target_waypoint}")
            self.current_waypoint_idx += 1
            
            if self.current_waypoint_idx >= len(self.waypoint_list):
                print("--- FINAL GOAL REACHED! ---")
                return None
            
            # Get the *new* next waypoint
            target_waypoint = self.waypoint_list[self.current_waypoint_idx]
            
        return target_waypoint
        

    def run_simulation_loop(self):
        """The main simulation loop where SENSE-THINK-ACT happens."""
        i = 0
        try:
            while self.simulation_app.is_running():
                dt = self.physics_context.get_physics_dt()
                if dt < 1e-6: 
                    dt = 1.0 / 60.0 
                
                # Update simulation
                self.simulation_app.update()
                
                # Update human animations
                # self.task.update_behaviors(dt)
                
                # --- SENSE ---
                # Get 3D pose [pos(x,y,z), quat(x,y,z,w)]
                position_3d, orientation_quat = self.robot.get_world_pose()

                if i == 100:  # Run debug on frame 100
                    print("\n" + "🔍"*35)
                    print("RUNNING COMPREHENSIVE LIDAR DEBUG")
                    print("🔍"*35)
                    self.robot.debug_lidar_transform()
                    self.robot.debug_transformation_comparison() 
                    self.robot.debug_find_axis_transform()
                
                # Get raw 3D Lidar data
                # lidar_raw_data = self.robot.get_lidar_world_points()

                # --- PROCESS (The "Glue") ---
                
                # 1. Convert 3D pose to 2D [x, y, theta]
                yaw = get_yaw_from_quat(orientation_quat)
                current_pose_2d = np.array([position_3d[0], position_3d[1], yaw])
                
                
                # 2. Get the current target waypoint
                target_waypoint = self.get_current_target(current_pose_2d)
                
                if target_waypoint is None:
                    # We are done! Stop the robot.
                    cmd_vel = [0.0, 0.0]
                else:
                # 3. Process 3D Lidar into 2.5D Height Map

                    # Default to an empty map (or the last known map)
                    height_map = self.height_map_processor.height_map


                    
                    
                

                    points_in_world_frame = self.robot.get_lidar_world_points()

                    if points_in_world_frame.size > 0 and i % 10 == 0:
                        self.clear_debug_drawing()
                        
                        # Get robot and lidar positions for reference
                        robot_pos, _ = self.robot.get_world_pose()
                        lidar_pos, _ = self.robot._lidar_vis.get_world_pose()
                        
                        # Draw the points (green)
                        self.visualize_lidar_points(
                            points_in_world_frame,
                            color=(1.0, 0.0, 0.0)  # Green points
                        )
                        
                        # Draw reference markers
                        draw = _debug_draw.acquire_debug_draw_interface()
                        
                        # 1. Big RED sphere at robot base
                        draw.draw_points([tuple(robot_pos)], [(1.0, 0.0, 0.0, 1.0)], [0.3])
                        
                        # 2. Big YELLOW sphere at lidar sensor
                        draw.draw_points([tuple(lidar_pos)], [(1.0, 1.0, 0.0, 1.0)], [0.2])
                        
                        # 3. Draw axes at lidar position
                        axis_length = 1.0
                        start_points = [tuple(lidar_pos)] * 3
                        end_points = [
                            tuple(lidar_pos + np.array([axis_length, 0, 0])),  # X red
                            tuple(lidar_pos + np.array([0, axis_length, 0])),  # Y green
                            tuple(lidar_pos + np.array([0, 0, axis_length]))   # Z blue
                        ]
                        line_colors = [
                            (1.0, 0.0, 0.0, 1.0),
                            (0.0, 1.0, 0.0, 1.0),
                            (0.0, 0.0, 1.0, 1.0)
                        ]
                        line_sizes = [5.0, 5.0, 5.0]
                        
                        draw.draw_lines(start_points, end_points, line_colors, line_sizes)
                        
                        # 4. Print stats every 100 frames
                        if i % 100 == 0:
                        #     print(f"\n📊 Frame {i} Lidar Stats:")
                        #     print(f"   Points visualized: {len(points_in_world_frame)}")
                        #     print(f"   Robot at: [{robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}]")
                        #     print(f"   Lidar at: [{lidar_pos[0]:.2f}, {lidar_pos[1]:.2f}, {lidar_pos[2]:.2f}]")
                            
                            # Check if test cube is in view
                            test_cube_pos = np.array([-3.0, 0.0, 0.5])
                            dist_to_cube = np.linalg.norm(points_in_world_frame - test_cube_pos, axis=1)
                            points_near_cube = np.sum(dist_to_cube < 1.0)
                            # print(f"   Points near test cube (<1m): {points_near_cube}")
                

                    # --- THINK ---
                    # Get the entire path that's left to follow

                    raw_points_np = self.robot.get_lidar_points_in_sensor_frame()
                    if raw_points_np.size > 0:
                        # Get sensor pose and robot pose
                        sensor_pose_np = self.robot.get_sensor_pose_matrix()
                        robot_pose_np = self.robot.get_robot_pose_vector()

                        raw_points = torch.from_numpy(raw_points_np).float().cuda()
                        sensor_pose = torch.from_numpy(sensor_pose_np).float().cuda()
                        robot_pose = torch.from_numpy(robot_pose_np).float().cuda()

                        costmap_2d = self.stvl.update(raw_points, sensor_pose, robot_pose)
                        costmap_np = costmap_2d.cpu().numpy()
                    else:
                        # No lidar data, use empty costmap
                        costmap_np = np.zeros((128, 128), dtype=np.float32)    
                    path_to_follow = self.waypoint_list[self.current_waypoint_idx:]

                    # The "Brain" (MPPI) computes the command
                   
                    cmd_vel = self.mppi_controller.compute_control_command(
                    current_pose_2d, 
                    path_to_follow,
                    height_map
                )
                # --- ACT ---
                # The "Driver" (DifferentialController) applies the command
                self.robot.apply_drive_commands(cmd_vel[0], cmd_vel[1])
                
                # Print for debugging
                if i % 100 == 0: 
                    print(f"\n{'='*60}")
                    print(f"--- Frame {i} ---")
                    print(f"{'='*60}")
                    print(f"  Robot Pose: x={current_pose_2d[0]:.2f}, y={current_pose_2d[1]:.2f}, θ={math.degrees(yaw):.1f}°")
                    
                    if target_waypoint is not None:
                        dist_to_target = np.linalg.norm(current_pose_2d[:2] - np.array(target_waypoint))
                        print(f"  Target Waypoint: {target_waypoint} (dist={dist_to_target:.2f}m)")
                        # print(f"  Progress: {self.current_waypoint_idx}/{len(self.waypoint_list)}")
                    else:
                        print(f"  ** GOAL REACHED **")
                    
    
                    
                    # Control command
                    print(f"  MPPI Command: v={cmd_vel[0]:.3f} m/s, ω={cmd_vel[1]:.3f} rad/s")
                    print(f"{'='*60}\n")
                    
                i += 1

        except KeyboardInterrupt:
            print("Caught interrupt. Shutting down simulation.")
        
        self.timeline.stop()

def run_full_simulation():
    """
    Manages the simulation setup and loop.
    """
    sim = NavigationSimulator(simulation_app)
    sim.setup_simulation()
    sim.run_simulation_loop()


if __name__ == "__main__":
    try:
        # Run the setup and main loop
        run_full_simulation()
            
    except Exception as e:
        print(f"An error occurred: {e}")
        
    finally:
        # This will shut down the app after the simulation finishes
        print("Closing simulation app...")
        simulation_app.close()
        sys.exit()
