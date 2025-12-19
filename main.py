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
from navigation_stack.robot.h1_humanoid import H1Humanoid

# from navigation_stack.controllers.mppi_controller import MPPIController
from navigation_stack.controllers.mppi_controller import MPPIController
from navigation_stack.controllers.mppi_types import MPPIConfig
# from navigation_stack.controllers.simple_controller import SimpleController
from navigation_stack.planner.global_planner import GlobalPlanner
from navigation_stack.perception.stvl_stem import STVL_System
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
        self.START_POS = np.array([-1.0,-1.0, 0.1])
        self.GOAL_POS = (5.0, 5.0)
        self.waypoint_list = []
        self.current_waypoint_idx = 0
        self.goal_threshold = 0.5  # (meters) How close to get to a waypoint

        # --- Get Lidar Transform ---
        # We will get this after the robot is spawned
        self.lidar_to_base_pos = None
        self.lidar_to_base_quat = None
        
        # --- Human actor ---
        self.h1_humanoid = None
        self.h1_timeline_sub = None

        # --- Initialize all our modules ---
        print("Initializing navigation modules...")
        self.global_planner = GlobalPlanner()
        # Define our grid properties
        # self.height_map_processor = HeightMapProcessor(
        #     grid_resolution=0.05,  # 5cm per cell
        #     grid_width_m=30.0,     # 40m wide map
        #     grid_height_m=30.0,    # 40m tall map
        #     robot_height=2.5,      # Max obstacle height to care about (2m)
        #     robot_ground_clearance=0.1 # Ignore floor (5cm)
        # )
        # self.mppi_controller = MPPIController()
        # self.mppi_controller = SimpleController()

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

    def setup_h1_humanoid(self):
        """Spawn H1 humanoid robot that walks straight."""
        try:
            print("Spawning H1 humanoid robot...")
            
            # Spawn position: left side of warehouse, facing +Y direction
            spawn_pos = [0, 7.0, 1.05]  # Near left wall
            walk_distance = 5.0  # Walk 3.5 meters (from y=1.0 to y=4.5)
            
            self.h1_humanoid = H1Humanoid(
                world=self.world,
                spawn_position=spawn_pos,
                walk_distance=walk_distance
            )
            
            if self.h1_humanoid.spawn():
                # Add physics callback
                self.world.add_physics_callback(
                    "h1_physics_step",
                    callback_fn=self.h1_humanoid.on_physics_step
                )
                
                # Add timeline event callback
                timeline = omni.timeline.get_timeline_interface()
                self.h1_timeline_sub = timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
                    int(omni.timeline.TimelineEventType.PLAY),
                    self.h1_humanoid.on_timeline_event
                )
                
                print("✓ H1 humanoid ready to walk")
            else:
                self.h1_humanoid = None
        except Exception as e:
            print(f" Failed to spawn H1: {e}")
            self.h1_humanoid = None


    def visualize_3d_voxel_grid(self, robot_pose_vector):
        """Visualize grid as vertical bars: XY = position, Z = occupancy height."""
        draw = _debug_draw.acquire_debug_draw_interface()
        
        # Get 3D grid and project to 2D (max over Z)
        stvl_grid_3d = self.stvl.stvl_grid.cpu().numpy()  # [W, H, D]
        costmap_2d = np.max(stvl_grid_3d, axis=2)  # [W, H] - max occupancy per column
        
        grid_dims = self.stvl.grid_dims
        voxel_size = self.stvl.voxel_size
        robot_centric_offset = self.stvl.robot_centric_offset.cpu().numpy()
        grid_origin = robot_pose_vector + robot_centric_offset
        
        # Find occupied cells
        occupied_indices = np.argwhere(costmap_2d > 0.1)
        
        if len(occupied_indices) == 0:
            return
        
        # print(f"   🧊 Drawing {len(occupied_indices)} grid bars")
        
        max_height = 1.0  # Maximum bar height in meters
        
        for idx in occupied_indices:
            i, j = idx
            occupancy = costmap_2d[i, j]
            
            # XY position (center of grid cell)
            x = grid_origin[0] + (i + 0.5) * voxel_size
            y = grid_origin[1] + (j + 0.5) * voxel_size
            z_base = robot_pose_vector[2]  # Ground level
            
            # Height = occupancy (0.1 to 1.0 → 0.2m to 2.0m)
            height = occupancy * max_height
            z_top = z_base + height
            
            # Color: yellow if occupancy=1.0, blue gradient if <1.0
            if occupancy >= 0.99:
                color = (1.0, 1.0, 0.0, 0.8)  # Yellow
            else:
                # Blue gradient: darker = lower occupancy
                brightness = occupancy
                color = (0.0, 0.0, brightness, 0.6)  # Blue
            
            # Draw vertical bar (4 vertical edges + top/bottom squares)
            half = voxel_size * 0.4
            corners_bottom = [
                (x - half, y - half, z_base),
                (x + half, y - half, z_base),
                (x + half, y + half, z_base),
                (x - half, y + half, z_base),
            ]
            
            corners_top = [
                (x - half, y - half, z_top),
                (x + half, y - half, z_top),
                (x + half, y + half, z_top),
                (x - half, y + half, z_top),
            ]
            
            # Bottom square
            for i in range(4):
                draw.draw_lines([corners_bottom[i]], [corners_bottom[(i+1)%4]], [color], [2.0])
            
            # Top square
            for i in range(4):
                draw.draw_lines([corners_top[i]], [corners_top[(i+1)%4]], [color], [2.0])
            
            # 4 vertical edges
            for i in range(4):
                draw.draw_lines([corners_bottom[i]], [corners_top[i]], [color], [2.0])
    def visualize_costmap(self, costmap_2d, robot_pose_vector):
        """Visualize the 2D costmap in Isaac Sim using debug draw."""
        draw = _debug_draw.acquire_debug_draw_interface()
        
        grid_dims = self.stvl.grid_dims
        voxel_size = self.stvl.voxel_size
        robot_centric_offset = self.stvl.robot_centric_offset.cpu().numpy()
        grid_origin = robot_pose_vector + robot_centric_offset
        
        # Lower threshold to see more cells
        occupied_indices = np.argwhere(costmap_2d > 0.1)
        
        # Debug print
        print(f"   🗺️ Costmap: {len(occupied_indices)} cells, max={costmap_2d.max():.3f}, grid_origin={grid_origin}")
        
        if occupied_indices.shape[0] == 0:
            return
        
        world_points = []
        for idx in occupied_indices:
            i, j = idx
            world_x = grid_origin[0] + (i + 0.5) * voxel_size
            world_y = grid_origin[1] + (j + 0.5) * voxel_size
            world_z = robot_pose_vector[2] + 0.5  # Higher for visibility
            world_points.append((world_x, world_y, world_z))
        
        if len(world_points) > 0:
            # Brighter magenta color, bigger size
            colors = [(1.0, 0.0, 1.0, 1.0)] * len(world_points)
            sizes = [voxel_size * 150] * len(world_points)
            draw.draw_points(world_points, colors, sizes)
            
            # Thicker green grid outline
            corners = [
                (grid_origin[0], grid_origin[1], robot_pose_vector[2] + 0.05),
                (grid_origin[0] + grid_dims[0] * voxel_size, grid_origin[1], robot_pose_vector[2] + 0.05),
                (grid_origin[0] + grid_dims[0] * voxel_size, grid_origin[1] + grid_dims[1] * voxel_size, robot_pose_vector[2] + 0.05),
                (grid_origin[0], grid_origin[1] + grid_dims[1] * voxel_size, robot_pose_vector[2] + 0.05),
            ]
            
            start_points = [corners[0], corners[1], corners[2], corners[3]]
            end_points = [corners[1], corners[2], corners[3], corners[0]]
            draw.draw_lines(start_points, end_points, [(0.0, 1.0, 0.0, 1.0)] * 4, [5.0] * 4)
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
        
        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0/200.0)
        # self.world.reset()
        
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

        path_with_heading, grid_path = self.global_planner.plan_path(self.START_POS[:2], self.GOAL_POS)        # if not path:
        
        if path_with_heading is None:
            carb.log_error("Failed to generate a global path. Exiting.")
            self.simulation_app.close()
            return

        self.waypoint_list = path_with_heading


        print(f"Created {len(self.waypoint_list)} waypoints with heading:")
        for i in range(min(5, len(self.waypoint_list))):  # Print first 5
            x, y, theta = self.waypoint_list[i]
            print(f"  [{i}]: ({x:.2f}, {y:.2f}) theta={math.degrees(theta):.1f}°")
        if len(self.waypoint_list) > 5:
            print(f"  ... and {len(self.waypoint_list) - 5} more waypoints")


        print("\n=== PATH DEBUG ===")
        print(f"Start pose: {self.START_POS[:2]}")
        print(f"Goal pose: {self.GOAL_POS}")
        print(f"First 3 waypoints:")
        for i in range(min(3, len(self.waypoint_list))):
            x, y, theta = self.waypoint_list[i]
            print(f"  [{i}]: ({x:.2f}, {y:.2f}) theta={math.degrees(theta):.1f}°")
        print(f"Last 3 waypoints:")
        for i in range(max(0, len(self.waypoint_list)-3), len(self.waypoint_list)):
            x, y, theta = self.waypoint_list[i]
            print(f"  [{i}]: ({x:.2f}, {y:.2f}) theta={math.degrees(theta):.1f}°")
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
        

        print("INITIALIZING MPPI CONTROLLER")
        # Path is already numpy array [N, 3] with heading
        full_path_array = self.waypoint_list
        goal_array = np.array(self.GOAL_POS)

        # Create MPPI config
        mppi_config = MPPIConfig()
        # Initialize MPPI with full path
        self.mppi_controller = MPPIController(
            config=mppi_config,
            full_path=full_path_array,
            goal=goal_array
        )
        print(f" MPPI Controller initialized!")
        print(f"   Path: {len(full_path_array)} waypoints")
        print(f"   Horizon: {mppi_config.planning_horizon_seconds:.2f}s")
        print(f"   Samples: {mppi_config.num_samples} trajectories")

                               
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

        self.setup_h1_humanoid()
        

        print("\n--- Simulation is running. Robot and Humans are spawned. ---")


    def get_current_target(self, current_pose_2d):
        """
        Manages the list of waypoints.
        Returns the current target and advances to the next if we're close.
        """
        if self.current_waypoint_idx >= len(self.waypoint_list):
            return None  # We have reached the final goal

        target_waypoint = self.waypoint_list[self.current_waypoint_idx]
        
        # Check distance to the current target (only x, y)
        dist_to_target = np.linalg.norm(current_pose_2d[:2] - target_waypoint[:2])
        
        if dist_to_target < self.goal_threshold:
            x, y, theta = target_waypoint
            print(f"Reached waypoint {self.current_waypoint_idx}: ({x:.2f}, {y:.2f}) theta={math.degrees(theta):.1f}°")
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
                    dt = 1.0 / 200.0 
                
                # Update simulation
                self.simulation_app.update()
                
                
                # --- SENSE ---
                # Get 3D pose [pos(x,y,z), quat(x,y,z,w)]
                position_3d, orientation_quat = self.robot.get_world_pose()

            

                # --- PROCESS (The "Glue") ---
                
                # 1. Convert 3D pose to 2D [x, y, theta]
                yaw = get_yaw_from_quat(orientation_quat)
                current_pose_2d = np.array([position_3d[0], position_3d[1], yaw])

                current_pose_tensor = torch.tensor(
                current_pose_2d,
                dtype=torch.float32,
                device='cuda'
                )
    


                

                    # points_in_world_frame = self.robot.get_lidar_world_points()

                    # if points_in_world_frame.size > 0 and i % 10 == 0:
                    #     self.clear_debug_drawing()
                        
                        # # Get robot and lidar positions for reference
                        # robot_pos, _ = self.robot.get_world_pose()
                        # lidar_pos, _ = self.robot._lidar_vis.get_world_pose()
                        
                        # # Draw the points (green)
                        # self.visualize_lidar_points(
                        #     points_in_world_frame,
                        #     color=(1.0, 0.0, 0.0)  # Green points
                        # )
                        
                        # # Draw reference markers
                        # draw = _debug_draw.acquire_debug_draw_interface()
                        
                        # 1. Big RED sphere at robot base
                        # draw.draw_points([tuple(robot_pos)], [(1.0, 0.0, 0.0, 1.0)], [0.3])
                        
                        # 2. Big YELLOW sphere at lidar sensor
                        # draw.draw_points([tuple(lidar_pos)], [(1.0, 1.0, 0.0, 1.0)], [0.2])
                        
                        # 3. Draw axes at lidar position
                        # axis_length = 1.0
                        # start_points = [tuple(lidar_pos)] * 3
                        # end_points = [
                        #     tuple(lidar_pos + np.array([axis_length, 0, 0])),  # X red
                        #     tuple(lidar_pos + np.array([0, axis_length, 0])),  # Y green
                        #     tuple(lidar_pos + np.array([0, 0, axis_length]))   # Z blue
                        # ]
                        # line_colors = [
                        #     (1.0, 0.0, 0.0, 1.0),
                        #     (0.0, 1.0, 0.0, 1.0),
                        #     (0.0, 0.0, 1.0, 1.0)
                        # ]
                        # line_sizes = [5.0, 5.0, 5.0]
                        
                        # draw.draw_lines(start_points, end_points, line_colors, line_sizes)
                        
                        # # 4. Print stats every 100 frames
                        # if i % 100 == 0:
                        # #     print(f"\n📊 Frame {i} Lidar Stats:")
                        # #     print(f"   Points visualized: {len(points_in_world_frame)}")
                        # #     print(f"   Robot at: [{robot_pos[0]:.2f}, {robot_pos[1]:.2f}, {robot_pos[2]:.2f}]")
                        # #     print(f"   Lidar at: [{lidar_pos[0]:.2f}, {lidar_pos[1]:.2f}, {lidar_pos[2]:.2f}]")
                            
                        #     # Check if test cube is in view
                        #     test_cube_pos = np.array([-3.0, 0.0, 0.5])
                        #     dist_to_cube = np.linalg.norm(points_in_world_frame - test_cube_pos, axis=1)
                        #     points_near_cube = np.sum(dist_to_cube < 1.0)
                        #     # print(f"   Points near test cube (<1m): {points_near_cube}")
                

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
                    # costmap_np = costmap_2d.cpu().numpy()
                    costmap_tensor = costmap_2d
                    
                    # if i % 5 == 0:  # Update visualization every 10 frames
                        # self.visualize_costmap(costmap_np, robot_pose_np)
                        # self.visualize_3d_voxel_grid(robot_pose_np)
                else:
                    # No lidar data, use empty costmap
                    costmap_tensor = torch.zeros((128, 128), dtype=torch.float32, device='cuda')
                    robot_pose_np = np.array([position_3d[0], position_3d[1], position_3d[2]])

                # path_to_follow = self.waypoint_list[self.current_waypoint_idx:]
                robot_centric_offset = self.stvl.robot_centric_offset[:2].cpu().numpy()
                # The "Brain" (MPPI) computes the command
                grid_origin = robot_pose_np[:2] + robot_centric_offset
                grid_origin_tensor = torch.tensor(
                    grid_origin,
                    dtype=torch.float32,
                    device='cuda'
                )

                v, w = self.mppi_controller.compute_control_command(
                    current_pose=current_pose_tensor,
                    costmap=costmap_tensor,
                    grid_origin=grid_origin_tensor
                )
                    # if i % 100 == 0:
                    #     robot_x = robot_pose_np[0]
                    #     robot_y = robot_pose_np[1]
                        
                    #     print(f"\n🎯 Path Verification:")
                    #     print(f"   Robot at: ({robot_x:.2f}, {robot_y:.2f})")
                    #     print(f"   Should be near X=1.0 (error: {abs(robot_x - 1.0):.2f}m)")
                        
                    #     # Print path chunk
                    #     chunk = self.mppi_controller.current_path_chunk
                    #     if chunk is not None and len(chunk) > 0:
                    #         print(f"   Path chunk[0]: ({chunk[0][0]:.2f}, {chunk[0][1]:.2f})")
                    #         if len(chunk) > 5:
                    #             print(f"   Path chunk[5]: ({chunk[5][0]:.2f}, {chunk[5][1]:.2f})")
                    #         print(f"   Chunk length: {len(chunk)} waypoints")
                            
                    #         # CRITICAL DEBUG: Check if costmap has high costs at path location
                    #         # This helps identify if obstacle avoidance is fighting path following
                    #         if costmap_tensor is not None:
                    #             print(f"   Costmap stats: min={costmap_tensor.min().item():.3f}, "
                    #                 f"max={costmap_tensor.max().item():.3f}, "
                    #                 f"mean={costmap_tensor.mean().item():.3f}")
    
                
                # --- ACT ---
                # The "Driver" (DifferentialController) applies the command
                self.robot.apply_drive_commands(v, w)
                
                if i % 100 == 0:
                    progress_info = self.mppi_controller.get_progress()
                    
                    print(f"\n{'='*60}")
                    print(f"--- Frame {i} ---")
                    print(f"{'='*60}")
                    print(f"  Robot Pose: x={current_pose_2d[0]:.2f}, y={current_pose_2d[1]:.2f}, "
                        f"θ={math.degrees(yaw):.1f}°")
                    print(f"  Progress: {progress_info['progress_pct']:.1f}%")
                    print(f"  Distance to goal: {progress_info['remaining_distance']:.2f}m")
                    print(f"  MPPI Command: v={v:.3f} m/s, ω={w:.3f} rad/s")
                    print(f"  Costmap stats: min={costmap_tensor.min().item():.3f}, "
                        f"max={costmap_tensor.max().item():.3f}, "
                        f"mean={costmap_tensor.mean().item():.3f}")
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
