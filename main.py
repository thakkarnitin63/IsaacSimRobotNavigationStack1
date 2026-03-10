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
        self.h1_humanoids = []  # List to store multiple humanoids
        self.h1_timeline_subs = []

        # --- Initialize all our modules ---
        print("Initializing navigation modules...")
        self.global_planner = GlobalPlanner(debug= True)
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



    def setup_h1_humanoids(self):
        """
        Spawn 10 H1 humanoid robots along the diagonal path.
        
        Strategy:
        - Place humanoids at intervals along diagonal from (-1,-1) to (5,5)
        - Give them different walking patterns:
          1. Cross path (perpendicular)
          2. Walk along path (same direction)
          3. Walk against path (opposite direction)
          4. Stand still (stationary obstacles)
        """
        try:
            print("\n" + "="*70)
            print("Spawning 10 H1 Humanoid Robots Along Path")
            print("="*70)
            
            # Define spawn configurations
            # Format: (position, walk_direction, walk_distance, description)
            humanoid_configs = [
    # Simple, clear 3-crossing scenario
    
    # Crossing 1: Early, slow crosser
    {
        'position': [1.5, 0.5, 1.05],
        'walk_direction': np.array([-1, 1, 0]) / np.sqrt(2),
        'walk_distance': 1.8,
        'start_delay': 0.0,
        'description': 'Crossing 1/3: Early section'
    },
    
    {
        'position': [1.5, 2.8, 1.05],  # ← MOVED: Start further back
        'walk_direction': np.array([1, -1, 0]) / np.sqrt(2),  # SE
        'walk_distance': 1.3,  # ← INCREASED: Longer walk to cross later
        'start_delay': 3.0,
        'description': 'Crossing 2/3: Mid section'
    },
    
    # ────────────────────────────────────────────────────────────
    # CROSSING 3: Late (robot reaches [4,4] at ~18 sec)
    # FIXED: Start BEHIND and to the side
    # ────────────────────────────────────────────────────────────
    {
        'position': [2.3, 4.5, 1.05],  # ← MOVED: Start further back
        'walk_direction': np.array([1, -1, 0]) / np.sqrt(2),  # SE
        'walk_distance': 2.0,  # ← INCREASED: Longer walk
        'start_delay':5.2,
        'description': 'Crossing 3/3: Late section'
    },
]
            
            timeline = omni.timeline.get_timeline_interface()
            
            for i, config in enumerate(humanoid_configs):
                print(f"\n Spawning Humanoid {i+1}/10:")
                print(f"   Position: {config['position']}")
                print(f"   Behavior: {config['description']}")
                
                delay = config.get('start_delay', 0.0)

                humanoid = H1Humanoid(
                    world=self.world,
                    spawn_position=config['position'],
                    walk_distance=config['walk_distance'],
                    walk_direction=config['walk_direction'],
                    start_delay=delay  # <--- Pass the delay here
                )
                
                if humanoid.spawn():
                    # Add physics callback
                    self.world.add_physics_callback(
                        f"h1_physics_step_{i}",
                        callback_fn=humanoid.on_physics_step
                    )
                    
                    # Add timeline event callback
                    sub = timeline.get_timeline_event_stream().create_subscription_to_pop_by_type(
                        int(omni.timeline.TimelineEventType.PLAY),
                        humanoid.on_timeline_event
                    )
                    
                    self.h1_humanoids.append(humanoid)
                    self.h1_timeline_subs.append(sub)
                    
                    print(f"    Spawned successfully")
                else:
                    print(f"    Failed to spawn")
            
            print("\n" + "="*70)
            print(f" Successfully spawned {len(self.h1_humanoids)}/10 humanoids")
            print("="*70)
            
            if len(self.h1_humanoids) < 10:
                print(f"  WARNING: Only {len(self.h1_humanoids)} humanoids spawned!")
            
        except Exception as e:
            print(f" Failed to spawn humanoids: {e}")
            import traceback
            traceback.print_exc()

    def visualize_global_path(self, height_offset: float = 0.15):
        """
        Visualize the global path as a line in Isaac Sim.
        
        Args:
            height_offset: Height above ground to draw the path (meters)
        """
        if self.waypoint_list is None or len(self.waypoint_list) < 2:
            print("No path to visualize")
            return
        
        draw = _debug_draw.acquire_debug_draw_interface()
        
        # Build lists of start and end points for each line segment
        start_points = []
        end_points = []
        colors = []
        sizes = []
        num_waypoints = len(self.waypoint_list)

        for i in range(num_waypoints - 1):
            # Current waypoint
            # Current waypoint - convert to float explicitly
            x1 = float(self.waypoint_list[i][0])
            y1 = float(self.waypoint_list[i][1])
            # Next waypoint
            x2 = float(self.waypoint_list[i + 1][0])
            y2 = float(self.waypoint_list[i + 1][1])
            
            start_points.append((x1, y1, height_offset))
            end_points.append((x2, y2, height_offset))
            
            # Gradient color: green at start → cyan at end
            progress = i / len(self.waypoint_list)
            color = (0.0, 1.0, progress, 1.0)  # Green to cyan
            colors.append(color)
            sizes.append(3.0)  # Line thickness
        
        # Draw all line segments
        draw.draw_lines(start_points, end_points, colors, sizes)
        
         # Draw start marker (green sphere)
        start_x = float(self.waypoint_list[0][0])
        start_y = float(self.waypoint_list[0][1])
        draw.draw_points(
            [(start_x, start_y, height_offset + 0.1)],
            [(0.0, 1.0, 0.0, 1.0)],  # Green
            [15.0]  # Size
        )
        
        # Draw goal marker (red sphere)
        goal_x = float(self.waypoint_list[-1][0])
        goal_y = float(self.waypoint_list[-1][1])
        draw.draw_points(
            [(goal_x, goal_y, height_offset + 0.1)],
            [(1.0, 0.0, 0.0, 1.0)],  # Red
            [15.0]  # Size
        )
        
        print(f"✓ Visualized global path: {num_waypoints} waypoints")
        
        


    def visualize_3d_voxel_grid(self, robot_pose_vector):
        """Visualize grid as vertical bars: XY = position, Z = occupancy height."""
        draw = _debug_draw.acquire_debug_draw_interface()
        
        # Get 3D grid and project to 2D (max over Z)
        stvl_grid_3d = self.stvl.stvl_grid.cpu().numpy()  # [W, H, D]
        costmap_2d,_ = np.max(stvl_grid_3d, axis=2)  # [W, H] - max occupancy per column
        
        grid_dims = self.stvl.grid_dims
        voxel_size = self.stvl.voxel_size
        robot_centric_offset = self.stvl.robot_centric_offset.cpu().numpy()
        grid_origin = robot_pose_vector + robot_centric_offset
        
        # Find occupied cells
        occupied_indices = np.argwhere(costmap_2d > 0.1)
        
        if len(occupied_indices) == 0:
            return
        
        # print(f"    Drawing {len(occupied_indices)} grid bars")
        
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
        print(f"    Costmap: {len(occupied_indices)} cells, max={costmap_2d.max():.3f}, grid_origin={grid_origin}")
        
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
        
        self.world = World(stage_units_in_meters=1.0, physics_dt=1.0/200.0, rendering_dt = 1.0/60.0)
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
        
        if path_with_heading is not None:
            # This is the line that saves the image
            self.global_planner.save_path_image_with_heading(
                path_with_heading=path_with_heading,
                output_path="my_path_visualization.png",
                arrow_spacing=10  # Draw an arrow every 10 points to avoid clutter
            )
            print("Image saved!")

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

        self.visualize_global_path(height_offset=0.15)
                               
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

        self.setup_h1_humanoids()

        self.cached_costmap = torch.zeros((256, 256), dtype=torch.float32, device='cuda')
        self.cached_grid_origin = torch.zeros(2, dtype=torch.float32, device='cuda')
        self.cached_distance_field = torch.full(  
        (256, 256), float('inf'), dtype=torch.float32, device='cuda'
    )
        

        print("\n--- Simulation is running. Robot and Humans are spawned. ---")

        

    def run_simulation_loop(self):
        """The main simulation loop where SENSE-THINK-ACT happens."""
        i = 0
        
        try:
            while self.simulation_app.is_running():
                dt = self.physics_context.get_physics_dt()
                if dt < 1e-6: 
                    dt = 1.0 / 200.0 
                
                # Update simulation
                self.world.step(render=True)
                
                
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

                    self.cached_costmap, self.cached_distance_field = self.stvl.update(raw_points, sensor_pose, robot_pose)

                    robot_centric_offset = self.stvl.robot_centric_offset[:2].cpu().numpy()
               
                    grid_origin = robot_pose_np[:2] + robot_centric_offset
                    self.cached_grid_origin = torch.tensor(
                        grid_origin, dtype=torch.float32, device='cuda'
                    )
                    self.cached_grid_origin = torch.tensor(grid_origin, dtype=torch.float32, device='cuda')

               
                v, w = self.mppi_controller.compute_control_command(
                    current_pose=current_pose_tensor,
                    costmap=self.cached_costmap,
                    distance_field=self.cached_distance_field, 
                    grid_origin=self.cached_grid_origin
                )
    
                
                # --- ACT ---
                # The "Driver" (DifferentialController) applies the command
                self.robot.apply_drive_commands(v, w)
                
                if i % 100 == 0:
                    #progress_info = self.mppi_controller.get_progress()
                    
                    print(f"\n{'='*60}")
                    print(f"--- Frame {i} ---")
                    print(f"{'='*60}")
                    print(f"  Robot Pose: x={current_pose_2d[0]:.2f}, y={current_pose_2d[1]:.2f}, "
                        f"θ={math.degrees(yaw):.1f}°")
                    #print(f"  Progress: {progress_info['progress_pct']:.1f}%")
                    #print(f"  Distance to goal: {progress_info['remaining_distance']:.2f}m")
                    print(f"  MPPI Command: v={v:.3f} m/s, ω={w:.3f} rad/s")
                    print(f"  Costmap stats: min={self.cached_costmap.min().item():.3f}, "
                        f"max={self.cached_costmap.max().item():.3f}, "
                        f"mean={self.cached_costmap.mean().item():.3f}")
                    print(f"{'='*60}\n")

                    # Print humanoid status
                    active_humanoids = sum(1 for h in self.h1_humanoids if not h.is_stationary)
                    print(f"  Active humanoids: {active_humanoids}/{len(self.h1_humanoids)}")


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
