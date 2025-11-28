# navigation_stack/perception/stvl_system.py
import torch
import kaolin
import time
from pathlib import Path
import math
import yaml

# Import our custom files
from . import numba_kernel
import kaolin.math.quat as Kq
from .frustum_model import ThreeDimensionalLidarFrustum 

class STVL_System:
    """
    A fully GPU-native Spatio-Temporal Voxel Layer (STVL) system.
    
    This class orchestrates the 5-step process:
    1. Pre-Processing (Frustum + Height)
    2. Fading (Temporal Decay)
    3. Clearing (Numba DDA "Active Clearing")
    4. Marking (Kaolin Voxelization)
    5. Flattening (to 2D Costmap)
    """
    
    def __init__(self): 
        """ Initialize STVL from config file in same directory."""

        # Load config from same folder as this file
        config_path = Path(__file__).parent / "stvl_config.yaml"
        config = self._load_config(config_path)

        print("Initializing STVL System...")

        self.device = torch.device(config['device'])

        # --- Grid Parameters ---
        # IMPORTANT: We assume grid_dims is [W, H, D] (e.g., [128, 128, 64])
        self.grid_dims = config['grid']['dimensions']
        self.grid_dims_tensor = torch.tensor(
            self.grid_dims, device=self.device, dtype=torch.int32
        )
        self.voxel_size = config['grid']['voxel_size']
        self.robot_centric_offset = torch.tensor(
            config['grid']['robot_centric_offset'],
            device=self.device, dtype=torch.float32
        )
        self.grid_dims_meters = self.grid_dims_tensor * self.voxel_size

        # --- The Core Data Structure ---
        # 3D occupancy grid (values: 0.0 = free, 1.0 = occupied)
        self.stvl_grid = torch.zeros(
            self.grid_dims, device=self.device, dtype=torch.float32
        )
        # --- STVL Logic Parameters ---
        # CRITICAL: Aggressive decay for dynamic obstacle handling
        # At 10Hz update rate:
        #   - 0.35 decay → obstacle fades to <0.1 in ~6 frames (~0.6s)
        #   - 0.40 decay → obstacle fades to <0.1 in ~10 frames (~1.0s)
        #   - 0.70 decay → obstacle fades to <0.1 in ~24 frames (~2.4s) [OLD]

        # --- STVL Logic Parameters ---
        self.DECAY_RATE = config['stvl']['decay_rate']
        self.MIN_OCCUPANCY = config['stvl']['min_occupancy']
        self.MAX_OCCUPANCY = config['stvl']['max_occupancy']
        self.COSTMAP_THRESHOLD = config['stvl']['costmap_threshold']
        
        # --- Pre-processing Parameters ---
        self.LIDAR_MIN_HEIGHT = config['lidar']['min_height_m']
        self.LIDAR_MAX_HEIGHT = config['lidar']['max_height_m']

        # Frustum Model
        frustum_params = {
            'v_fov': math.radians(config['lidar']['v_fov_deg']),
            'v_fov_padding': config['lidar']['v_fov_padding_m'],
            'h_fov': math.radians(config['lidar']['h_fov_deg']),
            'min_d': config['lidar']['min_range_m'],
            'max_d': config['lidar']['max_range_m']
        }
        self.frustum = ThreeDimensionalLidarFrustum(
            **frustum_params, device=self.device
        )

        
        print("STVL System Initialized!")
        print(f"  Grid: {self.grid_dims[0]}×{self.grid_dims[1]}×{self.grid_dims[2]} voxels")
        print(f"  Physical: {self.grid_dims_meters[0].item():.1f}m × "
              f"{self.grid_dims_meters[1].item():.1f}m × {self.grid_dims_meters[2].item():.1f}m")
        print(f"  Decay: {self.DECAY_RATE}, Threshold: {self.COSTMAP_THRESHOLD}")
        print(f"  Device: {self.device}")
        print()
    
    def _load_config(self, config_path):
        """Load configuration from YAML file."""
        if not config_path.exists():
            raise FileNotFoundError(
                f"Config file not found: {config_path}\n"
                f"Create 'stvl_config.yaml' in the same folder as stvl_system.py"
            )
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"✓ Loaded config from: {config_path}")
        return config
        
    def update(self, 
               raw_points: torch.Tensor, 
               sensor_pose_matrix: torch.Tensor, 
               robot_pose_vector: torch.Tensor):
        """
        Args:
            raw_points: [N, 3] tensor of lidar points in sensor frame
            sensor_pose_matrix: [4, 4] sensor-to-world transform
            robot_pose_vector: [3] robot position in world frame
        
        Returns:
            costmap_2d: [W, H] binary costmap for MPPI
        """
        
        # === STEP 1: TEMPORAL DECAY ===
        # All old data fades - prioritizes recent observations
        # Static obstacles are re-marked every frame, so they stay at 1.0
        # Dynamic obstacles that moved leave a fading "trail"
        self.stvl_grid *= self.DECAY_RATE
        
        # === STEP 2: PRE-PROCESSING ===
        # Filter points by frustum FOV and height
        valid_points, sensor_origin = self.pre_process_points(
            raw_points, sensor_pose_matrix, robot_pose_vector
        )
        
        # === STEP 3 & 4: CLEAR + MARK ===
        if valid_points.shape[0] > 0:
            # Calculate grid origin (stays robot-centric)
            # No rolling needed - we just recompute coordinates each frame
            grid_origin = robot_pose_vector + self.robot_centric_offset
            
            # Active clearing: Ray trace from sensor to obstacles
            # Sets all voxels along rays to MIN_OCCUPANCY (0.0)
            self.clear_free_space(valid_points, sensor_origin, grid_origin)

            # Mark obstacles: Set hit voxels to MAX_OCCUPANCY (1.0)
            self.mark_obstacles(valid_points, grid_origin)
        
        # === STEP 5: FLATTEN TO 2D COSTMAP ===
        costmap_2d = self.get_costmap()

        return costmap_2d

    def pre_process_points(self, 
                           raw_points: torch.Tensor, 
                           sensor_pose: torch.Tensor, 
                           robot_pose: torch.Tensor):
        
        # 1a. Transform points from Lidar frame to World frame
        points_world = self._transform_points(raw_points, sensor_pose)
        
        # 1b. Get sensor origin in World frame
        sensor_origin_world = sensor_pose[0:3, 3]

        # 1c. Update the frustum's position
        # We derive the quaternion from the 3x3 rotation matrix
        rot_mat_batched = sensor_pose[:3, :3].unsqueeze(0)
        sensor_quat_xyzw = Kq.quat_from_rot33(rot_mat_batched).squeeze(0)
        self.frustum.set_transform(sensor_origin_world, sensor_quat_xyzw)
        
        # 1d. Get the frustum mask
        frustum_mask = self.frustum.is_inside(points_world)
        # frustum_mask = torch.ones(points_world.shape[0], dtype=torch.bool, device=self.device)

        # 1e. Get height filter mask
        z_world = points_world[:, 2]
        robot_z = robot_pose[2]
        height_mask = (z_world > (robot_z + self.LIDAR_MIN_HEIGHT)) & \
                      (z_world < (robot_z + self.LIDAR_MAX_HEIGHT))
        
        # 1f. Combine masks
        final_mask = frustum_mask & height_mask
        
        # FIXED: Ensure mask has correct shape
        if final_mask.dim() == 0 or final_mask.shape[0] != points_world.shape[0]:
            # Mask is malformed, return empty points
            return torch.empty((0, 3), device=self.device), sensor_origin_world
            
        valid_points = points_world[final_mask]
        
        return valid_points, sensor_origin_world

    def clear_free_space(self, 
                         valid_points: torch.Tensor, 
                         sensor_origin: torch.Tensor, 
                         grid_origin: torch.Tensor):
        """
        Use DDA ray tracing to clear free space.
        
        For each valid point:
        - Trace ray from sensor to obstacle
        - Set all voxels along ray to MIN_OCCUPANCY (0.0)
        
        This actively removes "ghost" obstacles when sensor
        can see through previous obstacle positions.
        """

        
        # Convert world coordinates to grid index coordinates
        start_index = (sensor_origin - grid_origin) / self.voxel_size
        end_indices = (valid_points - grid_origin) / self.voxel_size
        
        # Launch the Numba DDA kernel
        numba_kernel.clear_free_space_kernel(
            self.stvl_grid,
            start_index,
            end_indices,
            self.grid_dims_tensor,
            self.MIN_OCCUPANCY
        )

    def mark_obstacles(self, 
                   valid_points: torch.Tensor, 
                   grid_origin: torch.Tensor):
        """
        Mark obstacle voxels using optimized manual voxelization.
        
        """
        # Convert world coordinate to grid indices (vectorized)
        if valid_points.shape[0] == 0:
            return 

        # Convert world coordinates to grid indices
        grid_indices = ((valid_points - grid_origin) / self.voxel_size).long()
        
        valid_mask = (
            (grid_indices[:, 0] >= 0) & (grid_indices[:, 0] < self.grid_dims[0]) &
            (grid_indices[:, 1] >= 0) & (grid_indices[:, 1] < self.grid_dims[1]) &
            (grid_indices[:, 2] >= 0) & (grid_indices[:, 2] < self.grid_dims[2])
        )
        grid_indices = grid_indices[valid_mask]
        
        # Remove duplicates (multiple points in same voxel)
        grid_indices_unique = torch.unique(grid_indices, dim=0)
        
        # Direct indexing (fastest approach)
        self.stvl_grid[
            grid_indices_unique[:, 0],
            grid_indices_unique[:, 1],
            grid_indices_unique[:, 2]
        ] = self.MAX_OCCUPANCY


    def get_costmap(self):
        """
        Generate 2D costmap for MPPI local planner.
        
        Projects 3D grid down to 2D by taking MAX value in Z-axis.
        Returns binary costmap (1=Lethal, 0=Free).
        """
        # Project 3D → 2D (max occupancy in vertical column)
        costmap_2d, _ = torch.max(self.stvl_grid, dim=2)
        
        # Threshold to binary (for MPPI cost function)
        return (costmap_2d > self.COSTMAP_THRESHOLD).float()

    def _transform_points(self, 
                          points: torch.Tensor, 
                          pose_matrix: torch.Tensor):
        
        """
        Transform points using 4x4 homogeneous transformation matrix.
        
        Args:
            points: [N, 3] points in source frame
            pose_matrix: [4, 4] transformation matrix
            
        Returns:
            [N, 3] points in target frame
        """
        # Ensure dtype consistency (convert to float32 if needed)  ← NEW!
        if points.dtype != pose_matrix.dtype:                      
            points = points.to(pose_matrix.dtype)                  
        # Convert to homogeneous coordinates [N, 4]
        points_h = torch.nn.functional.pad(points, (0, 1), 'constant', 1.0)
        
        # Apply transformation
        points_transformed_h = torch.matmul(pose_matrix, points_h.T).T
        
        # Convert back to 3D [N, 3]
        return points_transformed_h[:, 0:3]
